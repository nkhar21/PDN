# pdn_stackup_policy.py
from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

from generator.pdn_enums import NetType
from .pdn_stackup_model import Stackup

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]

# ---------- Builders / normalizers ----------

def make_stackup(
    *,
    num_layers: int,
    stackup_mask: Sequence[int],
    die_t: Sequence[float],
    er_list: Sequence[float],
    d_r: Sequence[float],
) -> Stackup:
    """
    Convenience builder that simply forwards to the immutable Stackup model.
    Kept here for symmetry with your 'policy + model' pattern.
    """
    return Stackup(
        num_layers=num_layers,
        stackup_mask=np.asarray(stackup_mask, dtype=np.int_),
        die_t=np.asarray(die_t, dtype=np.float64),
        er_list=np.asarray(er_list, dtype=np.float64),
        d_r=np.asarray(d_r, dtype=np.float64),
    )

# ---------- Lightweight helpers (used by policies elsewhere) ----------

def top_layer(s: Stackup) -> int:
    return 0

def bottom_layer(s: Stackup) -> int:
    return s.num_layers - 1

def is_outer_layer(s: Stackup, layer: int) -> bool:
    return layer in (0, s.num_layers - 1)

def layer_net(s: Stackup, layer: int) -> NetType:
    try:
        return NetType(int(s.stackup_mask[layer]))
    except Exception as e:
        raise IndexError(f"Layer index {layer} out of bounds for stackup with {s.num_layers} layers") from e

def touches_side(s: Stackup, start_layer: int, stop_layer: int, *, side: str) -> bool:
    """
    side: "top" or "bottom"
    """
    if side not in ("top", "bottom"):
        raise ValueError("side must be 'top' or 'bottom'")
    if side == "top":
        return start_layer == 0 or stop_layer == 0
    return start_layer == s.num_layers - 1 or stop_layer == s.num_layers - 1

def require_valid_layer_range(s: Stackup, start_layer: int, stop_layer: int) -> None:
    if start_layer < 0 or stop_layer >= s.num_layers:
        raise ValueError(
            f"Layer range {start_layer}->{stop_layer} out of bounds for stackup with {s.num_layers} layers"
        )

def require_via_net_match(s: Stackup, start_layer: int, stop_layer: int, via_net: NetType) -> None:
    """
    Ensure at least one of the two layers has the via's net type (PWR or GND).
    """
    nets = {int(s.stackup_mask[start_layer]), int(s.stackup_mask[stop_layer])}
    if via_net.value not in nets:
        raise ValueError(
            f"Via net {via_net.name} must land on at least one {via_net.name} layer "
            f"(layers {start_layer}->{stop_layer}, mask={s.stackup_mask})"
        )
