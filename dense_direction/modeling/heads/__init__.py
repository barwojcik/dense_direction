"""
Module containing model head classes.
"""

from .dpt_direction import DPTDirectionHead
from .dual_head import DualDecodeHead
from .dummy_direction import DummyDirectionHead
from .linear_direction import LinearDirectionHead
from .linear_segmentation import LinearHead
from .multiscale_direction import MultiscaleLossDirectionHead

__all__ = [
    DPTDirectionHead.__name__,
    DualDecodeHead.__name__,
    DummyDirectionHead.__name__,
    LinearDirectionHead.__name__,
    LinearHead.__name__,
    MultiscaleLossDirectionHead.__name__,
]
