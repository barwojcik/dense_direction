"""
Module containing loss classes.
"""

from .base_circular_directional import (
    BaseCircularDirectionalLoss,
    EfficientBaseCircularDirectionalLoss,
)
from .cosine_directional import CosineDirectionalLoss
from .directional import DirectionalLoss
from .efficient_cosine_directional import EfficientCosineDirectionalLoss
from .efficient_directional import EfficientDirectionalLoss
from .efficient_kl_directional import EfficientKLDirectionalLoss
from .efficient_von_mises_directional import EfficientVonMisesDirectionalLoss
from .kernels import *
from .kl_directional import KLDirectionalLoss
from .smoothness import SmoothnessLoss
from .von_mises_directional import VonMisesDirectionalLoss

__all__: list[str] = [
    BaseCircularDirectionalLoss.__name__,
    CosineDirectionalLoss.__name__,
    DirectionalLoss.__name__,
    EfficientBaseCircularDirectionalLoss.__name__,
    EfficientCosineDirectionalLoss.__name__,
    EfficientDirectionalLoss.__name__,
    EfficientKLDirectionalLoss.__name__,
    EfficientVonMisesDirectionalLoss.__name__,
    KLDirectionalLoss.__name__,
    SmoothnessLoss.__name__,
    VonMisesDirectionalLoss.__name__,
]
