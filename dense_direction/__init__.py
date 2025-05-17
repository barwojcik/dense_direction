"""
Dense direction framework.

The `dense_direction` is a framework for dense direction estimation of linear objects in images.
It is built on top of popular OpenMMLab's libraries (`mmengine`, `mmcv`, and `mmseg`), and utilizes
loss-based algorithmic weak-supervision to learn the direction estimation of linear objects from
semantic segmentation maps.
"""

import mmengine
import mmcv
import mmseg

from .data import *
from .evaluation import *
from .modeling import *
