"""
Dummy Direction Decode Head Class.

A dummy decode head class used for direction estimation tasks. This class does not contain any
layers, it simply returns the input tensor. It is meant to be used for multiscale loss computation.
"""

from torch import Tensor

from mmseg.registry import MODELS

from .base_direction import BaseDirectionDecodeHead


@MODELS.register_module()
class DummyDirectionHead(BaseDirectionDecodeHead):
    """
    Dummy decode head for a direction estimation task.

    Dummy head that is meant to be used for multiscale loss computation.

    Args:
        **kwargs: Additional arguments to pass to BaseDirectionDecodeHead.
    """

    def __init__(self, **kwargs) -> None:
        """
        Dummy decode head for a direction estimation task.

        Dummy head that is meant to be used for multiscale loss computation.

        Args:
            **kwargs: Additional arguments to pass to BaseDirectionDecodeHead.
        """
        super().__init__(**kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the dummy head.

        Args:
            inputs (Tensor): Input direction vector field for each class (N, K, 2, H, W).

        Returns:
            inputs (Tensor): Input direction vector field for each class (N, K, 2, H, W).
        """
        return inputs
