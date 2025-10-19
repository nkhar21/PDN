# pdn_bridge/map_stackup.py
from __future__ import annotations

import numpy as np

from BEM_AC_NVM_PDN import PDN
from generator.pdn_board import PDNBoard
from generator.pdn_enums import NetType


def _as_int_mask(mask_like) -> np.ndarray:
    """
    Convert PDNBoard.stackup.stackup_mask (NetType or ints) to int numpy array
    expected by PDN: 1=PWR, 0=GND, 2=FLOAT (reserved).
    """
    mask_like = np.asarray(mask_like, dtype=object)
    out = np.empty(mask_like.shape, dtype=int)
    for idx, v in np.ndenumerate(mask_like):
        if isinstance(v, NetType):
            out[idx] = 1 if v == NetType.PWR else 0
        elif isinstance(v, (int, np.integer)):
            if v in (0, 1, 2):
                out[idx] = int(v)
            else:
                raise ValueError(f"Unsupported stackup mask value {v!r} at index {idx}.")
        else:
            raise TypeError(f"Unsupported stackup mask element type {type(v)} at index {idx}.")
    return out.astype(int)


def _assert_outline_and_stackup(board: PDNBoard, pdn: PDN) -> None:
    if board.stackup is None:
        raise ValueError("PDNBoard.stackup is None. Set a valid Stackup on the board first.")
    if getattr(pdn, "bxy", None) is None:
        raise ValueError("PDN.bxy must be set before stackup mapping (call apply_outline first).")
    if getattr(pdn, "sxy_list", None) is None or getattr(pdn, "sxy", None) is None:
        raise ValueError("PDN segmentation missing (sxy/sxy_list). Run apply_outline after set_segmentation().")
    if not hasattr(pdn, "seg_len") or not np.isfinite(pdn.seg_len) or pdn.seg_len <= 0:
        raise ValueError(f"Invalid pdn.seg_len={getattr(pdn, 'seg_len', None)!r}. Segmentation must set a positive seg_len.")


def apply_stackup(board: PDNBoard, pdn: PDN) -> None:
    """
    Map PDNBoard.stackup ➜ PDN fields used by `calc_z_fast`.

    Sets:
      • pdn.stackup : int array, per *signal* layer (1=PWR, 0=GND, 2=FLOAT)
      • pdn.die_t   : float array, per dielectric (meters)
      • pdn.er_list : float array, per dielectric
      • pdn.d_r     : float array, per *signal* layer

    Notes:
      • We DO NOT broadcast outline/segments. A collapsed outline with len(bxy)==1
        is valid; calc_z_fast handles sxy_list length 1 by reusing geometry for all cavities.
    """
    _assert_outline_and_stackup(board, pdn)

    st = board.stackup

    # Copy stackup arrays (avoid aliasing)
    pdn.stackup = _as_int_mask(st.stackup_mask)
    pdn.die_t   = np.asarray(st.die_t,   dtype=float).copy()
    pdn.er_list = np.asarray(st.er_list, dtype=float).copy()
    pdn.d_r     = np.asarray(st.d_r,     dtype=float).copy()

    # Basic sanity
    if pdn.stackup.ndim != 1 or pdn.d_r.ndim != 1:
        raise ValueError("stackup and d_r must be 1-D arrays.")
    if len(pdn.stackup) != len(pdn.d_r):
        raise ValueError(
            f"stackup (len={len(pdn.stackup)}) and d_r (len={len(pdn.d_r)}) must have equal length."
        )
    if pdn.die_t.ndim != 1 or pdn.er_list.ndim != 1:
        raise ValueError("die_t and er_list must be 1-D arrays.")
    if len(pdn.die_t) != len(pdn.er_list):
        raise ValueError("die_t and er_list length mismatch.")
    if np.any(~np.isfinite(pdn.die_t)) or np.any(pdn.die_t <= 0):
        raise ValueError("All dielectric thickness values (die_t) must be finite and > 0.")
    if np.any(~np.isfinite(pdn.er_list)) or np.any(pdn.er_list <= 0):
        raise ValueError("All relative permittivities (er_list) must be finite and > 0.")

    # Outline/cavity compatibility policy:
    # - Collapsed outline (len(bxy)==1) is OK with any number of dielectrics.
    # - Non-collapsed outline (len(bxy)>1) should match number of dielectrics.
    num_cavities = len(pdn.die_t)
    num_polys = len(pdn.bxy)
    if num_polys > 1 and num_polys != num_cavities:
        raise ValueError(
            f"Non-collapsed outline expects one polygon per cavity: "
            f"dielectrics={num_cavities}, bxy entries={num_polys}."
        )
