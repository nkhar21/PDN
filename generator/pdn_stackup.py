from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from generator.pdn_enums import NetType


IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Stackup:
    """
    PCB stackup description.
    - stackup_mask: per-layer NetType mask (0=GND, 1=PWR) of shape (num_layers,)
    - die_t: dielectric thicknesses (m) of shape (num_layers-1,)
    - er_list: relative permittivities of shape (num_layers-1,)
    - d_r: conductor thicknesses (m) of shape (num_layers,)
    """
    num_layers: int
    stackup_mask: IntArray
    die_t: FloatArray
    er_list: FloatArray
    d_r: FloatArray

    def __post_init__(self) -> None:
        # Normalize to numpy arrays with exact dtypes
        mask = np.asarray(self.stackup_mask, dtype=np.int_)
        die_t = np.asarray(self.die_t, dtype=np.float64)
        er    = np.asarray(self.er_list, dtype=np.float64)
        d_r   = np.asarray(self.d_r, dtype=np.float64)

        # --- Validation ---
        nl = int(self.num_layers)

        if mask.shape != (nl,):
            raise ValueError(f"stackup_mask shape {mask.shape} must be ({nl},)")
        if d_r.shape != (nl,):
            raise ValueError(f"d_r shape {d_r.shape} must be ({nl},)")
        if die_t.shape != (nl - 1,):
            raise ValueError(f"die_t shape {die_t.shape} must be ({nl-1},)")
        if er.shape != (nl - 1,):
            raise ValueError(f"er_list shape {er.shape} must be ({nl-1},)")

        # Optional: ensure mask only contains valid NetType values
        if not np.isin(mask, (NetType.GND.value, NetType.PWR.value)).all():
            raise ValueError("stackup_mask must contain only NetType values (0=GND, 1=PWR).")

        # Because the dataclass is frozen, set normalized arrays via object.__setattr__
        object.__setattr__(self, "stackup_mask", mask)
        object.__setattr__(self, "die_t", die_t)
        object.__setattr__(self, "er_list", er)
        object.__setattr__(self, "d_r", d_r)

    def __repr__(self) -> str:
        return (
            f"Stackup(num_layers={self.num_layers}, "
            f"mask={self.stackup_mask}, "
            f"die_t={self.die_t}, "
            f"er_list={self.er_list}, "
            f"d_r={self.d_r})"
        )

    # -------- Convenience properties --------
    @property
    def num_dielectrics(self) -> int:
        return self.num_layers - 1

    # -------- Debug helpers --------
    def summary(self) -> None:
        print("=== Stackup Summary ===")
        print(f"Layers: {self.num_layers}")
        print(f"Stackup mask: {self.stackup_mask}")
        print(f"Dielectric layers: {self.num_dielectrics} → {self.die_t}")
        print(f"εr values: {self.er_list}")
        print(f"Conductor thicknesses: {self.d_r}")
        print("=======================")
