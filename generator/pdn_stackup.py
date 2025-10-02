import numpy as np
from typing import Optional
from generator.pdn_enums import NetType

class Stackup:
    """
    PCB stackup description.
    Holds mask of conductor layers (NetType), dielectric thicknesses,
    permittivities, and conductor thicknesses.
    """

    def __init__(self, num_layers: int,
                 stackup_mask,
                 die_t,
                 er_list,
                 d_r):
        self.num_layers: int = num_layers

        # --- Convert to arrays ---
        self.stackup_mask: np.ndarray = np.asarray(stackup_mask, dtype=int) # NetType mask, shape (num_layers,), 0=GND, 1=PWR
        self.die_t: np.ndarray = np.asarray(die_t, dtype=float)             # dielectric thicknesses, shape (num_layers-1,), in [m]
        self.er_list: np.ndarray = np.asarray(er_list, dtype=float)         # relative permittivities, shape (num_layers-1,)
        self.d_r: np.ndarray = np.asarray(d_r, dtype=float)                 # conductor thicknesses, shape (num_layers,), in [m]

        # --- Validation ---
        if len(self.stackup_mask) != num_layers:
            raise ValueError(f"stackup_mask length {len(self.stackup_mask)} "
                             f"must equal num_layers {num_layers}")
        if len(self.d_r) != num_layers:
            raise ValueError(f"d_r length {len(self.d_r)} must equal num_layers {num_layers}")
        if len(self.die_t) != len(self.er_list):
            raise ValueError(f"die_t length {len(self.die_t)} must equal er_list length {len(self.er_list)}")
        if len(self.die_t) != num_layers - 1:
            raise ValueError(f"die_t length {len(self.die_t)} must equal num_layers - 1 ({num_layers-1})")

    def __repr__(self):
        return (f"Stackup(num_layers={self.num_layers}, "
                f"mask={self.stackup_mask}, "
                f"die_t={self.die_t}, "
                f"er_list={self.er_list}, "
                f"d_r={self.d_r})")

    def summary(self):
        """Print debug info about the stackup."""
        print("=== Stackup Summary ===")
        print(f"Layers: {self.num_layers}")
        print(f"Stackup mask: {self.stackup_mask}")
        print(f"Dielectric layers: {len(self.die_t)} → {self.die_t}")
        print(f"εr values: {self.er_list}")
        print(f"Conductor thicknesses: {self.d_r}")
        print("=======================")
