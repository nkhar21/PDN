from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

# Alias for readability: each polygon is (N,2) float array; bxy is object array of polygons
Polygon = NDArray[np.float64]
BxyArray = np.ndarray  # dtype=object, each item is a Polygon


@dataclass
class BoardOutlineModel:
    """
    Pure data container for board outline geometry and segmentation.
    No external calls or file I/O here—just state.
    """
    # Outline polygons (N_layers,), each entry is (M,2) ndarray in meters
    bxy: Optional[BxyArray] = None

    # Segmentation results
    sxy: Optional[NDArray[np.float64]] = None           # (total_segs, 4): [x1,y1,x2,y2]
    seg_len: Optional[float] = None
    sxy_index_ranges: Optional[List[Tuple[int, int]]] = None
    sxy_list: Optional[List[NDArray[np.float64]]] = None

    # ---- helpers ----
    def num_layers(self) -> int:
        return 0 if self.bxy is None else int(self.bxy.shape[0])

    def is_collapsed(self) -> bool:
        n = self.num_layers()
        return n in (0, 1)

    def summary(self) -> None:
        print("=== BoardOutline Summary ===")
        print(f"Polygons: {self.num_layers() if self.bxy is not None else 0}")
        if self.sxy is not None:
            print(f"Segments: {self.sxy.shape}")
        print("============================")
