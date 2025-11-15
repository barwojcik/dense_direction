"""
Dummy Direction Decode Head Class.

A dummy decode head class used for direction estimation tasks. This class does not contain any
layers, it simply returns the input tensor. It is meant to be used for multiscale loss computation.
"""

from typing import Sequence

from torch import Tensor
from torch import nn

from mmseg.registry import MODELS
from mmseg.utils import ConfigType

from .base_direction import BaseDirectionDecodeHead


@MODELS.register_module()
class DummyDirectionHead(BaseDirectionDecodeHead):
    """
    Dummy decode head for a direction estimation task.

    Dummy head that is meant to be used for multiscale loss computation.

    Args:
        **kwargs: Additional arguments to pass to BaseDirectionDecodeHead.
    """

    def __init__(
        self,
        dir_classes: Sequence[int] = None,
        loss_decode: ConfigType = None,
        ignore_index: int = 255,
        align_corners: bool = False,
        pre_norm_vectors: bool = False,
        post_norm_vectors: bool = False,
        gt_scale_factor: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initializes DummyDirectionHead class.

        DummyDirectionHead class is meant to be used for multiscale loss computation.

        Args:
            **kwargs: Additional arguments to pass to BaseDirectionDecodeHead.
        """
        nn.Module.__init__(self)
        self.dir_classes: Sequence[int] = dir_classes or (1,)
        self.num_classes: int = len(self.dir_classes)
        self.ignore_index: int = ignore_index
        self.align_corners: bool = align_corners
        self.pre_norm_vectors: bool = pre_norm_vectors
        self.post_norm_vectors: bool = post_norm_vectors
        self.gt_scale_factor: float = gt_scale_factor
        self.sampler = None

        # 2 vector components per class
        self.in_channels: int = 2 * self.num_classes
        self.out_channels: int = 2 * self.num_classes

        loss_decode = loss_decode or self.DEFAULT_LOSS
        self._init_loss(loss_decode)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the dummy head.

        Args:
            inputs (Tensor): Input direction vector field for each class (N, K, 2, H, W).

        Returns:
            inputs (Tensor): Input direction vector field for each class (N, K, 2, H, W).
        """
        assert (
            len(inputs.shape) == 4
        ), f"Input tensor shape does not match. Got {inputs.shape} should be (N, K, 2, H, W)."
        assert inputs.shape[-4] == self.num_classes, (
            f"Input tensor shape does not match. Got {inputs.shape} (N, K, 2, H, W). K should be "
            f"{self.num_classes} instead got {inputs.shape[-4]}."
        )
        return inputs
