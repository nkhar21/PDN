# utils/geometry.py
from __future__ import annotations

from copy import deepcopy
from math import sqrt, pi, sin, cos
import numpy as np

# Vacuum permittivity
EPS0 = 8.854187817e-12


# --------------------------------------------------------------------------
# Areas
# --------------------------------------------------------------------------
def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the area of a polygon by the shoelace formula.

    Parameters
    ----------
    x, y : 1D arrays
        Vertex coordinates. The polygon may be open or closed; the
        formula is robust to either. Orientation (CW/CCW) does not
        affect the returned absolute area.

    Returns
    -------
    float
        Absolute polygon area in m².
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# --------------------------------------------------------------------------
# Boundary segmentation (outer board outline)
# --------------------------------------------------------------------------
def segment_boundary(bxy: np.ndarray, dl: float) -> np.ndarray:
    """
    Discretize the **outer board boundary** polygon into straight segments
    of ~length `dl`, ordered **counter-clockwise (CCW)** for CIM sign consistency.

    Parameters
    ----------
    bxy : (..., N, 2)
        Boundary vertices (meters). Must trace the outline **CCW** and
        the path must end at the start point (closed). If not closed,
        this function appends the first point to the end.
        If shape is (1, N, 2), unwrap to (N, 2).
    dl : float
        Target segment length (meters). Each polygon edge is split into
        roughly floor(edge_len/dl) segments; a small remainder creates
        one last short segment if > 1% of dl.

    Returns
    -------
    segments : (S, 4) array
        Segment endpoints as rows [x1, y1, x2, y2] in meters, CCW order.
    """
    bxy_old = deepcopy(bxy)
    if bxy_old.ndim == 3 and bxy_old.shape[0] == 1:
        bxy_old = bxy_old[0]

    # Ensure closed polygon
    if bxy_old[-1, 0] != bxy_old[0, 0] or bxy_old[-1, 1] != bxy_old[0, 1]:
        closed = np.zeros((bxy_old.shape[0] + 1, bxy_old.shape[1]))
        closed[0:-1, :] = bxy_old
        closed[-1, :]   = bxy_old[0, :]
    else:
        closed = bxy_old

    # First pass: count segments
    nseg = 0
    for i in range(closed.shape[0] - 1):
        dx = closed[i + 1, 0] - closed[i, 0]
        dy = closed[i + 1, 1] - closed[i, 1]
        L  = sqrt(dx*dx + dy*dy)
        if dl <= L:
            ne = np.floor(L / dl)
            nseg += ne + 1 if (L - ne*dl) > (0.01*dl) else ne
        else:
            nseg += 1
    nseg = int(nseg)

    # Second pass: generate segments
    segments = np.empty((nseg, 4), dtype=float)
    s = 0
    for i in range(closed.shape[0] - 1):
        x0, y0 = closed[i, 0],     closed[i, 1]
        x1, y1 = closed[i + 1, 0], closed[i + 1, 1]
        dx, dy = x1 - x0, y1 - y0
        L = sqrt(dx*dx + dy*dy)

        if dl <= L:
            ne = int(np.floor(L / dl))
            # Uniform sub-segments of length ~dl
            for j in range(ne):
                t0 = j      * dl / L
                t1 = (j + 1)* dl / L
                segments[s, 0] = x0 + t0 * dx
                segments[s, 1] = y0 + t0 * dy
                segments[s, 2] = x0 + t1 * dx
                segments[s, 3] = y0 + t1 * dy
                s += 1
            # Trailing remainder segment (if > 1% of dl)
            if (L - ne*dl) > (0.01*dl):
                t = (j + 1) * dl / L
                segments[s, 0] = x0 + t * dx
                segments[s, 1] = y0 + t * dy
                segments[s, 2] = x1
                segments[s, 3] = y1
                s += 1
        else:
            # Edge shorter than dl → keep as one segment
            segments[s, 0] = x0; segments[s, 1] = y0
            segments[s, 2] = x1; segments[s, 3] = y1
            s += 1

    # NOTE:
    # - **Orientation**: This function does not re-order vertices; the caller must supply
    #   the outline in **CCW** order (CIM sign convention). If your polygon is CW, reverse it.
    # - The via rims (ports) are segmented **clockwise** (segment_port); opposite orientation
    #   between inner (CW) and outer (CCW) contours is intentional in CIM to keep signs consistent.
    return segments


# --------------------------------------------------------------------------
# Port (via rim) segmentation
# --------------------------------------------------------------------------
def segment_port(x0: float, y0: float, r: float, n: int = 6) -> np.ndarray:
    """
    Discretize a circular via rim (port) into **n straight segments**,
    ordered **clockwise (CW)**.

    Parameters
    ----------
    x0, y0 : float
        Center of the via (meters).
    r : float
        Via rim radius (meters).
    n : int, default 6
        Number of segments.

    Returns
    -------
    segments : (n, 4) ndarray
        Rows are [x1, y1, x2, y2] segment endpoints, CW order.
    """
    n = int(n)
    dtheta = 2 * pi / n
    segments = np.empty((n, 4), dtype=float)
    for i in range(n):
        # CW orientation by negating the angle
        segments[i, 0] = x0 + r * cos(-(i)     * dtheta)
        segments[i, 1] = y0 + r * sin(-(i)     * dtheta)
        segments[i, 2] = x0 + r * cos(-(i + 1) * dtheta)
        segments[i, 3] = y0 + r * sin(-(i + 1) * dtheta)
    return segments


# --------------------------------------------------------------------------
# Whole-outline segmentation helper (refactor of PDN.seg_bd)
# --------------------------------------------------------------------------
def segment_outline(
    outer_boundary: np.ndarray,
    seg_len: float,
    inner_boundary: np.ndarray | None = None,
    er_list: list[float] | np.ndarray | None = None,
):
    """
    Segment outer/inner board boundaries and compute net copper area.
    (Pure function replacement for the old `PDN.seg_bd`.)

    Parameters
    ----------
    outer_boundary : (N1, 2) ndarray
        Outer boundary vertices (meters). Supply **CCW** ordering.
    seg_len : float
        Target segment length (meters) for polygon discretization.
    inner_boundary : (N2, 2) ndarray or None, optional
        Optional inner void/ring vertices (meters). If following CIM sign
        conventions strictly, supply **CW** ordering (not strictly required here).
    er_list : list[float] or (L,) ndarray, optional
        Relative permittivity per dielectric cavity. If provided, a parallel-plate
        `C_pul`-like array is returned as `EPS0 * er * area` (no thickness division).
        The caller may divide by thickness where appropriate.

    Returns
    -------
    outer_segments : (S1, 4) ndarray
        Segments of the outer boundary (CCW).
    inner_segments : (S2, 4) ndarray or None
        Segments of the inner boundary (CW) if provided, else None.
    all_segments : (S, 4) ndarray
        Concatenation of outer (and inner if present) segments.
    area : float
        Net copper area (outer minus inner) in m².
    c_pul_like : (L,) ndarray or None
        If `er_list` is given, returns `EPS0 * er * area` per layer; else None.

    Notes
    -----
    - This routine is purely geometric; it does not handle per-layer polygons.
      If you have multiple layers, compute each layer’s area separately and
      use overlap as needed at call sites.
    - We intentionally do **not** divide by thickness here.
    """
    outer_segments = segment_boundary(outer_boundary, seg_len)
    inner_segments = None

    # For area, use the vertex arrays (not the segments) to avoid any small
    # numerical differences introduced by discretization.
    if inner_boundary is not None and len(inner_boundary) != 0:
        inner_segments = segment_boundary(inner_boundary, seg_len)
        area = polygon_area(outer_boundary[:, 0], outer_boundary[:, 1]) \
             - polygon_area(inner_boundary[:, 0], inner_boundary[:, 1])
        all_segments = np.concatenate((outer_segments, inner_segments), axis=0)
    else:
        area = polygon_area(outer_boundary[:, 0], outer_boundary[:, 1])
        all_segments = outer_segments

    c_pul_like = None
    if er_list is not None:
        er_arr = np.asarray(er_list, dtype=float).ravel()
        c_pul_like = EPS0 * er_arr * float(area)

    return outer_segments, inner_segments, all_segments, float(area), c_pul_like


__all__ = [
    "EPS0",
    "polygon_area",
    "segment_boundary",
    "segment_port",
    "segment_outline",
]
