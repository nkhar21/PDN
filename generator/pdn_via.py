import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List

from generator.pdn_enums import NetType, ViaRole

@dataclass
class Via:
    xy: Tuple[float, float]         
    start_layer: int                
    stop_layer: int                 
    via_type: NetType               # GND or PWR
    role: ViaRole = ViaRole.UNSET   # default UNSET, will be set by PDNBoard
    
    def __post_init__(self):
        if self.start_layer < 0 or self.stop_layer < 0:
            raise ValueError("Layer indices must be >= 0")
        if self.stop_layer < self.start_layer:
            # Auto-fix: swap them
            print(f"[WARN] Via at {self.xy}: start_layer={self.start_layer} > stop_layer={self.stop_layer}. Reversing.")
            self.start_layer, self.stop_layer = self.stop_layer, self.start_layer

    def as_array(self) -> np.ndarray:
        return np.array([
            self.xy[0], self.xy[1],
            self.start_layer, self.stop_layer,
            self.via_type.value,
            self.role.value
        ], dtype=float)



class ViaCollection:
    """
    Container for multiple vias.
    Provides methods to add, query, and export via data.
    """

    def __init__(self):
        self.vias: List[Via] = []

    def add_via(self, via: Via):
        self.vias.append(via)

    def add_vias(self, vias: List[Via]):
        self.vias.extend(vias)

    def to_numpy(self) -> np.ndarray:
        """Return all vias as (N,6) array: x, y, start, stop, type, role."""
        if not self.vias:
            return np.zeros((0, 6))
        return np.vstack([v.as_array() for v in self.vias])

    def filter_by_type(self, via_type: NetType) -> List[Via]:
        """Return all vias of a given type."""
        return [v for v in self.vias if v.via_type == via_type]

    def dedupe(self, rdec: int = 9, warn: bool = True):
        """
        Remove exact duplicates (same xy, start, stop, type).
        """
        unique = []
        seen = set()

        for v in self.vias:
            key = (round(v.xy[0], rdec),
                   round(v.xy[1], rdec),
                   v.start_layer,
                   v.stop_layer,
                   v.via_type.value)

            if key in seen:
                if warn:
                    print(f"[DEDUP] Dropped duplicate via: xy={v.xy}, "
                          f"layers {v.start_layer}->{v.stop_layer}, type={v.via_type.name}")
                continue  # drop duplicate
            seen.add(key)
            unique.append(v)

        self.vias = unique

    def nudge(self, eps: float = 1e-7, rdec: int = 9) -> List[Tuple[int, Tuple[float, float], Tuple[float, float]]]:
        """
        Nudge stacked vias (same xy but different vertical spans)
        so they don't collide in BEM solvers.

        Returns
        -------
        nudged : list of (index, old_xy, new_xy)
            - index : int, index of via in self.vias
            - old_xy : (x, y) before nudging
            - new_xy : (x, y) after nudging
        """
        seen = set()
        nudged = []

        for i, v in enumerate(self.vias):
            x, y = float(v.xy[0]), float(v.xy[1])
            old_xy = (x, y)
            key = (round(x, rdec), round(y, rdec))

            while key in seen:
                x += eps
                y += eps
                key = (round(x, rdec), round(y, rdec))

            if (x, y) != old_xy:  # if nudged
                nudged.append((i, old_xy, (x, y)))

            seen.add(key)
            v.xy = (x, y)

        return nudged


    def summary(self):
        print("=== ViaCollection Summary ===")
        print(f"Total vias: {len(self.vias)}")
        n_gnd = len(self.filter_by_type(NetType.GND))
        n_pwr = len(self.filter_by_type(NetType.PWR))
        print(f"  Ground vias: {n_gnd}")
        print(f"  Power vias:  {n_pwr}")
        for i, v in enumerate(self.vias):
            print(f"  {i}: xy=({v.xy[0]:.4f},{v.xy[1]:.4f}), "
                f"layers {v.start_layer}->{v.stop_layer}, "
                f"type={v.via_type.name}, role={v.role.name}")
        print("=============================")


