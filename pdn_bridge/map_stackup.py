# pdn_bridge/map_stackup.py
from __future__ import annotations

import numpy as np

from BEM_AC_NVM_PDN import PDN
from generator.pdn_board import PDNBoard
from generator.pdn_enums import NetType


def _as_int_mask(mask_like) -> np.ndarray:
    """
    Convert PDNBoard.stackup.stackup_mask (NetType or ints) to int numpy array
    expected by PDN: 1=PWR, 0=GND, 2=FLOAT (if ever used).
    """
    mask_like = np.asarray(mask_like, dtype=object)
    out = np.empty(mask_like.shape, dtype=int)

    for idx, v in np.ndenumerate(mask_like):
        # Enum NetType -> int
        if isinstance(v, NetType):
            if v == NetType.PWR:
                out[idx] = 1
            elif v == NetType.GND:
                out[idx] = 0
            else:
                # If you someday add NetType.FLOAT, map to 2 here
                out[idx] = 2
        # Already an int
        elif isinstance(v, (int, np.integer)):
            if v in (0, 1, 2):
                out[idx] = int(v)
            else:
                raise ValueError(f"Unsupported stackup mask value {v!r} at index {idx}.")
        else:
            raise TypeError(f"Unsupported stackup mask element type {type(v)} at index {idx}.")

    return out.astype(int)


def _assert_stackup_ready(board: PDNBoard, pdn: PDN) -> None:
    if board.stackup is None:
        raise ValueError("PDNBoard.stackup is None. Initialize PDNBoard with a valid Stackup.")
    if getattr(pdn, "bxy", None) is None:
        raise ValueError("PDN.bxy must be set before stackup mapping (call apply_outline first).")

    # Basic shape checks pulled from solver expectations:
    # For N signal layers, there are N-1 dielectrics (cavities).
    # We also require bxy per cavity (len(bxy) == number of cavities).
    die_t = np.asarray(board.stackup.die_t, dtype=float)
    er_list = np.asarray(board.stackup.er_list, dtype=float)

    if die_t.ndim != 1 or er_list.ndim != 1:
        raise ValueError("die_t and er_list must be 1-D arrays.")

    if len(die_t) != len(er_list):
        raise ValueError(f"die_t (len={len(die_t)}) and er_list (len={len(er_list)}) must have equal length.")

    # bxy is an object array of per-cavity polygons
    num_cavities = len(pdn.bxy)  # set by apply_outline
    if len(die_t) != num_cavities:
        raise ValueError(
            f"Mismatch: dielectrics={len(die_t)} but bxy cavities={num_cavities}. "
            "Your outline and stackup must describe the same number of cavities."
        )

    if np.any(~np.isfinite(die_t)) or np.any(die_t <= 0):
        raise ValueError("All dielectric thickness values (die_t) must be finite and > 0.")

    if np.any(~np.isfinite(er_list)) or np.any(er_list <= 0):
        raise ValueError("All relative permittivities (er_list) must be finite and > 0.")


def apply_stackup(board: PDNBoard, pdn: PDN) -> None:
    """
    Map PDNBoard.stackup ➜ PDN fields used by `calc_z_fast`.

    Sets:
      • pdn.stackup : int array, per *signal* layer (1=PWR, 0=GND, 2=FLOAT)
      • pdn.die_t   : float array, per dielectric (meters), length = num_cavities
      • pdn.er_list : float array, per dielectric (relative permittivity), length = num_cavities
      • pdn.d_r     : float array, per *signal* layer (meters)

    Notes:
      • Consistency with outline is enforced: len(die_t) == len(er_list) == len(pdn.bxy)
      • This mapper assumes `apply_outline` already ran.
    """
    _assert_stackup_ready(board, pdn)

    st = board.stackup

    # Mask conversion (enums -> ints)
    pdn.stackup = _as_int_mask(st.stackup_mask)

    # Copy arrays (avoid aliasing)
    pdn.die_t = np.asarray(st.die_t, dtype=float).copy()
    pdn.er_list = np.asarray(st.er_list, dtype=float).copy()
    pdn.d_r = np.asarray(st.d_r, dtype=float).copy()

    # Additional sanity checks:
    if pdn.stackup.ndim != 1 or pdn.d_r.ndim != 1:
        raise ValueError("stackup and d_r must be 1-D arrays.")
    if len(pdn.stackup) != len(pdn.d_r):
        raise ValueError(
            f"stackup (len={len(pdn.stackup)}) and d_r (len={len(pdn.d_r)}) must match the number of signal layers."
        )
