"""
Module containing model backbone classes.
"""

from .dino2_hub import Dino2TorchHub
from .dino3_hub import Dino3TorchHub

__all__ = [Dino2TorchHub.__name__, Dino3TorchHub.__name__]
