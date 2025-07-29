"""
Module containing model head classes.
"""

from .dpt_direction import DPTDirectionHead
from .dual_head import DualDecodeHead
from .linear_direction import LinearDirectionHead
from .linear_segmentation import LinearHead

__all__ = [
    DPTDirectionHead.__name__,
    DualDecodeHead.__name__,
    LinearDirectionHead.__name__,
    LinearHead.__name__,
]
