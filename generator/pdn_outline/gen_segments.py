from __future__ import annotations

from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray

Polygon = NDArray[np.float64]  # (N,2)
BxyArray = np.ndarray          # dtype=object, each item is a Polygon


def _segment_polygon(poly: Polygon, dl: float) -> NDArray[np.float64]:
    """
    Subdivide a *closed* polygon into ~uniform segments of target length dl ClockWise Direction (CW).
    Returns (M,4) array of segments [x1, y1, x2, y2].

    Assumes poly is (N,2) with poly[0] == poly[-1] (closure). If not closed,
    this function will treat the last->first edge anyway.
    """
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"Polygon must be (N,2); got {poly.shape}")
    if poly.shape[0] < 3:
        # Minimum 3 points (including closure) to form an area; still handle edges
        pass

    # Ensure we iterate every edge, including last->first
    pts = poly
    if not np.allclose(pts[0], pts[-1], atol=0.0, rtol=0.0):
        pts = np.vstack([pts, pts[0]])

    segs: List[NDArray[np.float64]] = []
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        L = (dx * dx + dy * dy) ** 0.5
        if L == 0.0:
            continue

        # number of subdivisions along this edge
        n = max(1, int(np.ceil(L / dl)))
        t = np.linspace(0.0, 1.0, n + 1)
        xs = p0[0] + t * dx
        ys = p0[1] + t * dy

        # build [x1,y1,x2,y2] for each subsegment
        seg_xy = np.stack([xs[:-1], ys[:-1], xs[1:], ys[1:]], axis=1)
        segs.append(seg_xy.astype(np.float64, copy=False))

    if segs:
        return np.vstack(segs)
    return np.zeros((0, 4), dtype=np.float64)


def generate_segments(
    bxy: BxyArray,
    dl: float
) -> Tuple[NDArray[np.float64], List[Tuple[int, int]], List[NDArray[np.float64]]]:
    """
    Generate segmented boundary attributes from per-layer outlines.

    Parameters
    ----------
    bxy : np.ndarray[dtype=object]
        Board outline polygons (num_layers,), each entry (Mi,2) in meters.
    dl : float
        Target segment length (meters).

    Returns
    -------
    sxy : (total_segs, 4) float64
        Concatenated segments: [x1, y1, x2, y2].
    sxy_index_ranges : list[(start, stop)]
        Index slices into sxy for each layer's polygon.
    sxy_list : list[np.ndarray]
        Per-polygon segment arrays, same format as sxy rows.
    """
    if not isinstance(bxy, np.ndarray) or bxy.dtype != object:
        raise TypeError("bxy must be a numpy.ndarray[dtype=object] of polygons")
    if dl <= 0:
        raise ValueError("dl must be > 0")

    sxy_list: List[NDArray[np.float64]] = []
    sxy_index_ranges: List[Tuple[int, int]] = []
    offset = 0

    for poly in bxy:
        if not isinstance(poly, np.ndarray):
            raise TypeError("Each bxy entry must be an ndarray polygon")
        segs = _segment_polygon(poly, dl)
        sxy_list.append(segs)
        n_seg = int(segs.shape[0])
        sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg

    sxy = np.concatenate(sxy_list, axis=0) if sxy_list else np.zeros((0, 4), dtype=np.float64)
    return sxy, sxy_index_ranges, sxy_list
