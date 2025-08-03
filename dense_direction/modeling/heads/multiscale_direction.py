"""
Multiscale Loss Direction Head class for multiscale loss computation.

This module provides a MultiscaleLossDirectionHead class. This class is a wrapper that wraps a "real"
direction head and any number of "dummy" direction heads. The "real" head performs like all other
direction heads, while the "dummy" heads enable loss computation with different gt_scale_factor
values.
"""

from typing import Sequence

from torch import Tensor
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList

from .dummy_direction import DummyDirectionHead


@MODELS.register_module()
class MultiscaleLossDirectionHead(nn.Module):
    """
    Multiscale loss direction head for multiscale loss computation.

    This class is a wrapper that wraps a "real" direction head and any number of "dummy" heads.

    Args:
        real_head (ConfigType): The config for the "real" direction head.
        dummy_heads (ConfigType | list[ConfigType]): The config or list of configs for the "dummy"
            direction heads. If not all keys are provided, they will be filled with the "real" values.
    """

    DEFAULT_DUMMY_HEAD = DummyDirectionHead.__name__

    def __init__(
        self,
        real_head: ConfigType,
        dummy_heads: ConfigType | list[ConfigType],
    ) -> None:
        """
        Initializes MultiscaleLossDirectionHead class.

        This class is a wrapper that wraps a "real" direction head and any number of "dummy" heads.

        Args:
            real_head (ConfigType): The config for the "real" direction head.
            dummy_heads (ConfigType | list[ConfigType]): The config or list of configs for the "dummy"
                direction heads. If not all keys are provided, they will be filled with the "real" values.
        """
        super().__init__()
        self.real_head = MODELS.build(real_head)
        self._init_dummy_heads(real_head, dummy_heads)

        self.align_corners = self.real_head.align_corners
        self.num_classes = self.real_head.num_classes
        self.out_channels = self.real_head.out_channels
        self.dir_classes = self.real_head.dir_classes

    def _init_dummy_heads(
        self, real_head: ConfigType, dummy_heads: ConfigType | list[ConfigType]
    ) -> None:
        """
        Builds dummy heads from config.
        """
        base_config: ConfigType = real_head.copy()
        base_config.type = self.DEFAULT_DUMMY_HEAD

        heads: list[nn.Module] = []
        if isinstance(dummy_heads, list):
            for dummy_config in dummy_heads:
                dummy_config = base_config.merge(dummy_config)
                dummy_head = MODELS.build(dummy_config)
                heads.append(dummy_head)
        else:
            dummy_config = base_config.merge(dummy_heads)
            dummy_head = MODELS.build(dummy_config)
            heads.append(dummy_head)

        self.dummy_heads: nn.ModuleList = nn.ModuleList(heads)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        """
        Forward pass through the real decode head.

        This method wraps real head forward call and returns outputs.

        Args:
            inputs (Sequence[Tensor]): Input tensors of shape (N, C, H, W).

        Returns:
            outputs (Tensor): Output direction vector field for each class (N, K, 2, H, W).
        """
        outputs: Tensor = self.real_head(inputs)

        return outputs

    def loss(
        self, inputs: tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType
    ) -> dict:
        """
        Forward function for training.

        This method wraps all direction head loss calls and returns a merged loss dictionary.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg data samples. It usually
                includes information such as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        dir_vectors: Tensor = self.real_head.forward(inputs)
        losses: dict = self.real_head.loss_by_feat(dir_vectors, batch_data_samples)
        losses = {f"dir1.{key}": value for key, value in losses.items()}

        for head_idx, dummy_head in enumerate(self.dummy_heads):
            dummy_losses: dict = dummy_head.loss_by_feat(dir_vectors, batch_data_samples)
            dummy_losses = {f"dir{head_idx+2}.{key}": value for key, value in dummy_losses.items()}
            losses.update(dummy_losses)

        return losses

    def predict(
            self, inputs: tuple[Tensor], batch_img_metas: list[dict], test_cfg: ConfigType
    ) -> Tensor:
        """
        Forward function for prediction.

        This method wraps real head predict call and returns outputs.

        Args:
            inputs (tuple[Tensor]): List of multi-level img features.
            batch_img_metas (list[dict]): List Image info where each dict may also contain: 'img_shape',
                'scale_factor', 'flip', 'img_path', 'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs direction vector field for each class.
        """

        return self.real_head.predict(inputs, batch_img_metas, test_cfg)
