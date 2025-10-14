from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from matplotlib.path import Path

Point = Tuple[float, float]
Polygon = NDArray[np.float64]   # (N,2)

def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Return True if (x,y) lies inside polygon."""
    return Path(polygon).contains_point((float(point[0]), float(point[1])))
