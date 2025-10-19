from __future__ import annotations

from typing import List, Literal
import numpy as np
from numpy.typing import NDArray

from generator.pdn_outline.gen_outline import OutlineGeneratorFactory, OutlineMode
from generator.pdn_outline.gen_segments import generate_segments
from .pdn_outline_model import BoardOutlineModel, BxyArray


def set_outline(model: BoardOutlineModel,
                shapes: List[NDArray[np.float64]],
                *,
                mode: OutlineMode,
                units: Literal["mm", "m"] = "mm") -> None:
    """
    Validates and converts incoming shapes to meters, sets model.bxy.
    Resets any existing segmentation state.
    """
    if not shapes:
        raise ValueError("At least one outline polygon must be provided")

    gen = OutlineGeneratorFactory.create(mode, units=units)
    bxy: BxyArray = gen.generate(shapes)

    if not isinstance(bxy, np.ndarray) or bxy.dtype != object:
        raise TypeError("Generated bxy must be a numpy.ndarray[dtype=object] of polygons")

    # commit
    model.bxy = bxy
    # invalidate segmentation when outline changes
    model.sxy = None
    model.seg_len = None
    model.sxy_index_ranges = None
    model.sxy_list = None


def set_segmentation(model: BoardOutlineModel, seg_len: float) -> None:
    """
    Segments model.bxy with a target segment length (meters) and stores
    concatenated segments + per-polygon slices back into the model.
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
