# generator/pdn_outline/pdn_outline_policy.py
from __future__ import annotations

from typing import List, Literal
import numpy as np
from numpy.typing import NDArray

from generator.pdn_outline.gen_outline import generate_bxy, validate_bxy
from generator.pdn_outline.gen_segments import generate_segments
from .pdn_outline_model import BoardOutlineModel, BxyArray


def set_outline(
    model: BoardOutlineModel,
    shapes: List[NDArray[np.float64]],
    *,
    units: Literal["mm", "m"] = "mm",
    auto_close: bool = True,
    tol: float = 1e-9,
) -> None:
    """
    Build bxy with exactly one polygon per layer.
    - Converts to meters
    - Optionally closes polygons
    - Resets any existing segmentation state
    """
    if not shapes:
        raise ValueError("At least one outline polygon must be provided (one per layer).")

    bxy: BxyArray = generate_bxy(shapes, units=units, auto_close=auto_close, tol=tol)
    validate_bxy(bxy)

    # Commit
    model.bxy = bxy

    # Invalidate segmentation whenever outline changes
    model.sxy = None
    model.seg_len = None
    model.sxy_index_ranges = None
    model.sxy_list = None


def set_segmentation(model: BoardOutlineModel, seg_len: float) -> None:
    """
    Segment bxy with a target segment length (meters) and store:
      - concatenated segments (sxy)
      - per-polygon index ranges (sxy_index_ranges)
      - per-polygon segment lists (sxy_list)
    """
    if model.bxy is None:
        raise ValueError("bxy must be set before segmentation")
    if seg_len <= 0:
        raise ValueError("seg_len must be > 0")

    sxy, ranges, sxy_list = generate_segments(model.bxy, seg_len)

    model.seg_len = float(seg_len)
    model.sxy = sxy
    model.sxy_index_ranges = ranges
    model.sxy_list = sxy_list
