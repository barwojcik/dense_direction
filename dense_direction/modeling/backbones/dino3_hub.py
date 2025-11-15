"""
Dinov3 model wrapper for PyTorch Hub.

This module provides a wrapper for the Dinov3 model from PyTorch Hub, that can be loaded and used
as a backbone.
"""
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
import torch.nn as nn

from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class Dino3TorchHub(BaseModule):
    """
    Wrapper for the Dinov3 model from PyTorch Hub.

    Model source:
        https://github.com/facebookresearch/dinov3

    This class provides a wrapper for PyTorch Hub distribution of the Dinov3 model, that can be
    loaded and used as a backbone.

    Args:
        repo_url (str): Directory where the repository is cloned.
        weights_path (str): Path to the weights file.
        model_size (str): The size of the model to be used. It can be one of: 'small',
            'small_plus', 'baseline', 'large', 'huge_plus', 'full'.
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
        **kwargs: Additional arguments to pass to BaseModule.
    """

    MODEL_NAMES: dict[str, str] = dict(
        small='dinov3_vits16',
        small_plus='dinov3_vits16plus',
        baseline='dinov3_vitb16',
        large='dinov3_vitl16',
        huge_plus='dinov3_vith16plus',
        full='dinov3_vit7b16',
    )
    _DEFAULT_REPO_URL: str = "facebookresearch/dinov3"

    def __init__(
        self,
        weights_path: str,
        repo_url: str | None = None,
        model_size: str = "small",
        return_intermediate_layers: bool = True,
        layers_to_extract: int | Sequence[int] = 1,
        reshape_output: bool = True,
        return_class_token: bool = False,
        norm_output: bool = True,
        frozen: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the Dinov3 model from PyTorch Hub.

        Args:
            weights_path (str): Path to the weight .pt file.
            repo_url (str | None): Optional link to the git repository, when not provided, it
                defaults to `facebookresearch/dinov3`. Default None.
            model_size (str): The size of the model to be used. It can be one of: 'small',
                'small_plus', 'baseline', 'large', 'huge_plus', 'full'.
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
        assert weights_path is not None, "Please provide a valid weights path"
        assert Path(weights_path).exists(), "Weights path does not exist"

        model_name = self.MODEL_NAMES[model_size]

        self.layers: nn.Module = torch.hub.load(
            repo_url or self._DEFAULT_REPO_URL,
            model_name,
            pretrained=False,
        )
        self.layers.load_state_dict(torch.load(weights_path, weights_only=False), strict=False)
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

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor to be processed by the model.

        Returns:
            list[Tensor]: Output tensors after processing by the model.
        """

        if self.return_intermediate_layers:
            x = self.layers.get_intermediate_layers(
                x,
                n=self.layers_to_extract,
                reshape=self.reshape_output,
                return_class_token=self.return_class_token,
                norm=self.norm_output,
            )
            x = list(x)
        else:
            x = [
                self.layers(x),
            ]

        return x
