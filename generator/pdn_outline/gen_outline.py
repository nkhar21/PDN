from __future__ import annotations

from typing import Iterable, Literal
import numpy as np
from numpy.typing import NDArray

Polygon = NDArray[np.float64]   # (N, 2)
BxyArray = np.ndarray           # dtype=object, each item is a Polygon


def _to_meters_and_close(
    poly: np.ndarray,
    *,
    units: Literal["mm", "m"] = "mm",
    auto_close: bool = True,
    tol: float = 1e-9,
) -> Polygon:
    """
    Normalize one polygon:
      - cast to float64
      - convert units to meters
      - optionally close the polygon (append first point to the end if needed)
    """
    if not isinstance(poly, np.ndarray):
        raise TypeError("Each shape must be a numpy.ndarray")

    a = np.asarray(poly, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"Polygon must be (N,2); got shape {a.shape}")

    scale = 1e-3 if units == "mm" else 1.0
    a = a * scale

    if auto_close and a.shape[0] >= 2:
        if not np.allclose(a[0], a[-1], atol=tol, rtol=0.0):
            a = np.vstack([a, a[0]])

    return a


def generate_bxy(
    shapes: Iterable[np.ndarray],
    *,
    units: Literal["mm", "m"] = "mm",
    auto_close: bool = True,
    tol: float = 1e-7,
) -> BxyArray:
    """
    Build the outline array bxy with one polygon **per layer**.

    Parameters
    ----------
    shapes : iterable of (N,2) ndarrays
        One polygon per layer. Provide them in layer order (0..N-1).
    units : "mm" | "m", default "mm"
        Units of input coordinates.
    auto_close : bool, default True
        If True, ensure each polygon is closed (first point repeated at end).
    tol : float, default 1e-9
        Tolerance used to decide if the polygon is already closed.

    Returns
    -------
    bxy : np.ndarray[dtype=object]
        Object array of polygons in meters: shape (num_layers,), each entry (Mi,2).

    Notes
    -----
    This replaces the previous “collapsed/noncollapsed” generator logic.
    """
    shapes_list = list(shapes)
    if len(shapes_list) == 0:
        raise ValueError("At least one outline polygon must be provided (one per layer).")

    out = np.empty(len(shapes_list), dtype=object)
    for i, s in enumerate(shapes_list):
        out[i] = _to_meters_and_close(s, units=units, auto_close=auto_close, tol=tol)
    return out


def validate_bxy(bxy: BxyArray) -> None:
    """
    Lightweight validator for bxy produced by generate_bxy.
    Raises on structural issues; returns None if OK.
    """
    if not isinstance(bxy, np.ndarray) or bxy.dtype != object:
        raise TypeError("bxy must be a numpy.ndarray with dtype=object")

    if bxy.size == 0:
        raise ValueError("bxy has no layers")

    for i, poly in enumerate(bxy):
        if not isinstance(poly, np.ndarray):
            raise TypeError(f"bxy[{i}] is not an ndarray")
        if poly.ndim != 2 or poly.shape[1] != 2:
            raise ValueError(f"bxy[{i}] must be (N,2); got shape {poly.shape}")
        if poly.shape[0] < 3:
            raise ValueError(f"bxy[{i}] must have at least 3 points (including closure)")
