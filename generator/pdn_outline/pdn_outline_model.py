from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

# Each polygon is (N,2) float array; bxy is object array of polygons
Polygon = NDArray[np.float64]
BxyArray = np.ndarray  # dtype=object, each item is a Polygon


@dataclass(slots=True)
class BoardOutlineModel:
    """
    Pure data container for board outline geometry and segmentation.
    - bxy: object array of per-layer polygons in meters, shape (num_layers,)
    - sxy: concatenated boundary segments (total_segs, 4): [x1, y1, x2, y2]
    """
    bxy: Optional[BxyArray] = None

    # Segmentation cache
    sxy: Optional[NDArray[np.float64]] = None
    seg_len: Optional[float] = None
    sxy_index_ranges: Optional[List[Tuple[int, int]]] = None
    sxy_list: Optional[List[NDArray[np.float64]]] = None

    # ---- helpers ----
    def num_layers(self) -> int:
        return 0 if self.bxy is None else int(self.bxy.shape[0])

    def summary(self) -> None:
        print("=== BoardOutline Summary ===")
        print(f"Layers (polygons): {self.num_layers()}")
        if self.sxy is not None:
            print(f"Segments array shape: {self.sxy.shape}")
        print("============================")
