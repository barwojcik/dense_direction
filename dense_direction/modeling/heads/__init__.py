"""
Module containing model head classes.
"""

from .dpt_direction_head import DPTDirectionHead
from .linear_direction_head import LinearDirectionHead
from .linear_segmentation_head import LinearHead

__all__ = [DPTDirectionHead.__name__, LinearDirectionHead.__name__, LinearHead.__name__]
