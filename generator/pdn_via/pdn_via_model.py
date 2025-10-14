from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Sequence, Dict, Optional, Set
import numpy as np
from numpy.typing import NDArray
from generator.pdn_enums import NetType, ViaRole

Point = Tuple[float, float]
ViaKey = Tuple[float, float, int, int, int]  # rounded x, rounded y, start, stop, net

@dataclass
class Via:
    xy: Point
    start_layer: int
    stop_layer: int
    via_type: NetType
    role: ViaRole = ViaRole.UNSET
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.start_layer < 0 or self.stop_layer < 0:
            raise ValueError("Layer indices must be >= 0")
        if self.stop_layer < self.start_layer:
            print(f"[WARN] Via at {self.xy}: start_layer={self.start_layer} > stop_layer={self.stop_layer}. Reversing.")
            self.start_layer, self.stop_layer = self.stop_layer, self.start_layer

    def as_array(self) -> NDArray[np.float64]:
        vid = -1.0 if self.id is None else float(self.id)
        # id, x, y, start, stop, type, role
        return np.array([
            vid,
            float(self.xy[0]), float(self.xy[1]),
            float(self.start_layer), float(self.stop_layer),
            float(self.via_type.value), float(self.role.value)
        ], dtype=np.float64)

class ViaCollection:
    """
    Pure container:
      - Assigns unique IDs
      - O(1) id→index
      - Basic accessors
    """
    def __init__(self) -> None:
        self.vias: List[Via] = []
        self._id_to_index: Dict[int, int] = {}
        self._next_id: int = 0

    def __len__(self) -> int: return len(self.vias)
    def __iter__(self) -> Iterable[Via]: return iter(self.vias)

    # --- IDs ---
    def _assign_id(self, via: Via) -> int:
        if via.id is None:
            vid = self._next_id; self._next_id += 1
            via.id = vid
            return vid
        if via.id in self._id_to_index:
            raise ValueError(f"Via id {via.id} already exists in collection")
        if via.id >= self._next_id:
            self._next_id = via.id + 1
        return via.id

    def add_via(self, via: Via) -> int:
        vid = self._assign_id(via)
        self.vias.append(via)
        self._id_to_index[vid] = len(self.vias) - 1
        return vid

    def add_vias(self, vias: List[Via]) -> List[int]:
        return [self.add_via(v) for v in vias]

    def has_id(self, vid: int) -> bool:
        return vid in self._id_to_index

    def index_of_id(self, vid: int) -> int:
        if vid not in self._id_to_index:
            raise KeyError(f"Via id {vid} not found")
        return self._id_to_index[vid]

    def get_by_id(self, vid: int) -> Via:
        return self.vias[self.index_of_id(vid)]

    def ids_to_indices(self, ids: Sequence[int]) -> NDArray[np.int_]:
        return np.asarray([self.index_of_id(vid) for vid in ids], dtype=np.int_)

    # --- queries/exports (no policy logic here) ---
    def to_numpy(self) -> NDArray[np.float64]:
        """
        Return vias as (N,7): id, x, y, start, stop, type, role
        """
        if not self.vias:
            return np.zeros((0, 7), dtype=np.float64)
        rows = [v.as_array() for v in self.vias]  # <-- now always includes id
        return np.vstack(rows).astype(np.float64, copy=False)

    def filter_by_type(self, via_type: NetType) -> List[Via]:
        return [v for v in self.vias if v.via_type == via_type]

    def summary(self) -> None:
        print("=== ViaCollection Summary ===")
        print(f"Total vias: {len(self.vias)}")
        n_gnd = len(self.filter_by_type(NetType.GND))
        n_pwr = len(self.filter_by_type(NetType.PWR))
        print(f"  Ground vias: {n_gnd}")
        print(f"  Power vias:  {n_pwr}")
        for v in self.vias:
            vid = -1 if v.id is None else v.id
            print(f"  id={vid}: xy=({v.xy[0]:.4f},{v.xy[1]:.4f}), layers {v.start_layer}->{v.stop_layer}, type={v.via_type.name}, role={v.role.name}")
        print("=============================")
