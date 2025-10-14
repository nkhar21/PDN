from __future__ import annotations
from typing import Tuple, List, Set
from generator.pdn_via.pdn_via_model import ViaCollection, Via, Point, ViaKey

def dedupe_vias(vias: ViaCollection, rdec: int = 9, warn: bool = True) -> None:
    unique: list[Via] = []
    seen: Set[ViaKey] = set()
    for v in vias.vias:
        key: ViaKey = (
            round(float(v.xy[0]), rdec),
            round(float(v.xy[1]), rdec),
            int(v.start_layer),
            int(v.stop_layer),
            int(v.via_type.value),
        )
        if key in seen:
            if warn:
                print(f"[DEDUP] Dropped duplicate via: id={v.id}, xy={v.xy}, layers {v.start_layer}->{v.stop_layer}, type={v.via_type.name}")
            continue
        seen.add(key)
        unique.append(v)

    vias.vias = unique
    # rebuild id map
    vias._id_to_index.clear()
    for idx, v in enumerate(vias.vias):
        if v.id is None:
            v.id = vias._assign_id(v)
        vias._id_to_index[v.id] = idx

def nudge_vias(vias: ViaCollection, eps: float = 1e-7, rdec: int = 9) -> List[Tuple[int, Point, Point]]:
    seen_xy: Set[Tuple[float, float]] = set()
    nudged: List[Tuple[int, Point, Point]] = []
    for v in vias.vias:
        x, y = float(v.xy[0]), float(v.xy[1])
        old_xy: Point = (x, y)
        key_xy = (round(x, rdec), round(y, rdec))
        while key_xy in seen_xy:
            x += eps; y += eps
            key_xy = (round(x, rdec), round(y, rdec))
        if (x, y) != old_xy:
            nudged.append((int(v.id) if v.id is not None else -1, old_xy, (x, y)))
        seen_xy.add(key_xy)
        v.xy = (x, y)
    return nudged
