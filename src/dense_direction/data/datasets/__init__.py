"""
Module containing datasets classes.
"""

from .concrete_cracks import ConcreteCracksDataset
from .ottawa_roads import OttawaRoadsDataset

__all__: list[str] = [ConcreteCracksDataset.__name__, OttawaRoadsDataset.__name__]
