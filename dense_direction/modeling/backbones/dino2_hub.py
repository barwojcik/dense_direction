"""
Dinov2 model wrapper for PyTorch Hub.

This module provides a wrapper for the Dinov2 model from PyTorch Hub, that can be loaded and used
as a backbone.
"""

from typing import Sequence
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class Dino2TorchHub(BaseModule):
    """
    Wrapper for the Dinov2 model from PyTorch Hub.

    Model source:
        https://github.com/facebookresearch/dinov2

    This class provides a wrapper for PyTorch Hub distribution of the Dinov2 model, that can be
    loaded and used as a backbone.

    Args:
        model_size (str): The size of the model to be used. It can be one of: 'small', 'baseline',
            'large', 'xlarge'.
        with_registers (bool): Whether to use model trained with registers or not. Default True.
        return_intermediate_layers (bool): Whether to return intermediate layers. Default True.
        layers_to_extract (int | Sequence[int]): The number of last layers or the indices of the
            layers to extract. Default True. Applies only when return_intermediate_layers argument
            is set to True.
        reshape_output (bool): Whether to reshape the output. Default True. Applies only when
            return_intermediate_layers argument is set to True.
        return_class_token (bool): Whether to return the class token. Default False. Applies only
            when return_intermediate_layers argument is set to True.
        norm_output (bool): Whether to normalize the output. Default True. Applies only when
            return_intermediate_layers argument is set to True.
    """

    REPO_NAME: str = "facebookresearch/dinov2"
    MODEL_NAMES: dict[str, str] = dict(
        small="dinov2_vits14",
        baseline="dinov2_vitb14",
        large="dinov2_vitl14",
        xlarge="dinov2_vitg14",
    )
    MODEL_REG_NAMES: dict[str, str] = dict(
        small="dinov2_vits14_reg",
        baseline="dinov2_vitb14_reg",
        large="dinov2_vitl14_reg",
        xlarge="dinov2_vitg14_reg",
    )

    def __init__(
        self,
        model_size: str = "small",
        with_registers: bool = True,
        return_intermediate_layers: bool = True,
        layers_to_extract: int | Sequence[int] = 1,
        reshape_output: bool = True,
        return_class_token: bool = False,
        norm_output: bool = True,
        frozen: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the Dinov2 model from PyTorch Hub.

        Args:
            model_size (str): The size of the model to be used. It can be one of: 'small',
                'baseline', 'large', 'xlarge'.
            with_registers (bool): Whether to use model trained with registers or not.
                Default True.
            return_intermediate_layers (bool): Whether to return intermediate layers.
                Default True.
            layers_to_extract (int | Sequence[int]): The number of last layers or the indices of
                the layers to extract. Default True. Applies only when return_intermediate_layers
                argument is set to True.
            reshape_output (bool): Whether to reshape the output. Default True. Applies only when
                return_intermediate_layers argument is set to True.
            return_class_token (bool): Whether to return the class token. Default False. Applies
                only when return_intermediate_layers argument is set to True.
            norm_output (bool): Whether to normalize the output. Default True. Applies only when
                return_intermediate_layers argument is set to True.
            frozen (bool): Whether to freeze the model weights. Default False.
        """

        super().__init__(**kwargs)
        assert (
            model_size in self.MODEL_NAMES.keys()
        ), f"Model size should be one of {self.MODEL_NAMES.keys()}"
        if with_registers:
            model_name = self.MODEL_REG_NAMES[model_size]
        else:
            model_name = self.MODEL_NAMES[model_size]
        self.layers: nn.Module = torch.hub.load(self.REPO_NAME, model_name)
        self.return_intermediate_layers: bool = return_intermediate_layers
        self.layers_to_extract: int | Sequence[int] = layers_to_extract
        self.reshape_output: bool = reshape_output
        self.return_class_token: bool = return_class_token
        self.norm_output: bool = norm_output
        self.frozen: bool = frozen
        self._freeze()

    def _freeze(self) -> None:
        if self.frozen:
            self.layers.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
        x (Tensor): Input tensor to be processed by the model.

        Returns:
        Tensor: Output tensor after processing by the model.
        """

        if self.return_intermediate_layers:
            x = self.layers.get_intermediate_layers(
                x,
                self.layers_to_extract,
                reshape=self.reshape_output,
                return_class_token=self.return_class_token,
                norm=self.norm_output,
            )
        else:
            x = self.layers(x)

        return x
