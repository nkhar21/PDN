from __future__ import annotations

import math
import numpy as np

from BEM_AC_NVM_PDN import PDN
from generator.pdn_board import PDNBoard

def _assert_outline_ready(board: PDNBoard) -> None:
    bo = board.outline
    if bo is None:
        raise ValueError("PDNBoard.outline is not initialized. Call PDNBoard.set_outline(...).")

    # bxy (per-cavity polygons) must exist
    if getattr(bo, "bxy", None) is None:
        raise ValueError("BoardOutline.bxy is None. Call PDNBoard.set_outline(...).")

    # segmentation (sxy & sxy_list) must exist
    if getattr(bo, "sxy", None) is None or getattr(bo, "sxy_list", None) is None:
        raise ValueError(
            "BoardOutline segmentation is missing. "
            "Call PDNBoard.outline.set_segmentation(seg_len=...)."
        )

    # seg_len must be present and valid (required, not optional)
    seg_len = getattr(bo, "seg_len", None)
    if seg_len is None or not np.isfinite(seg_len) or seg_len <= 0:
        raise ValueError(
            f"Invalid BoardOutline.seg_len={seg_len!r}. "
            "You must call set_segmentation(seg_len>0) before mapping."
        )


def apply_outline(board: PDNBoard, pdn: PDN) -> None:
    """
    Map BoardOutline ➜ PDN fields used by `calc_z_fast`.

    Sets (all required):
      • pdn.bxy       : object array of (Nix2) float polygons
      • pdn.sxy       : (Sx4) segments (concatenated)
      • pdn.sxy_list  : list of per-cavity segment arrays
      • pdn.seg_len   : positive float (segment length)

    Note: `area` is NOT set here; `calc_z_fast` computes it from `sxy_list`.
    """
    _assert_outline_ready(board)
    bo = board.outline

    # Polygons (object array preserves per-cavity shapes)
    pdn.bxy = np.asarray(bo.bxy, dtype=object)

    # Segments (concatenated and per-cavity)
    pdn.sxy = bo.sxy.copy()
    pdn.sxy_list = [seg.copy() for seg in bo.sxy_list]  # type: ignore[attr-defined]

    # Required segment length
    pdn.seg_len = float(bo.seg_len)
