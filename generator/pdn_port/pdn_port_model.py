from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Optional
from generator.pdn_enums import PortRole, PortSide

@dataclass(frozen=True)
class Terminal:
    via_ids: List[int]

@dataclass(frozen=True)
class Port:
    name: str
    role: PortRole      # IC or DECAP
    side: PortSide      # TOP or BOTTOM
    positive: Terminal
    negative: Terminal

class PortCollection:
    def __init__(self) -> None:
        self._ports: list[Port] = []

    def __len__(self) -> int: return len(self._ports)
    def __iter__(self) -> Iterable[Port]: return iter(self._ports)
    def add(self, port: Port) -> None: self._ports.append(port)

    def get(self, name: str) -> Optional[Port]:
        for p in self._ports:
            if p.name == name:
                return p
        return None

    def names(self) -> list[str]:
        return [p.name for p in self._ports]

    def summary(self) -> None:
        print("=== Port Summary ===")
        if not self._ports:
            print("No ports defined.")
        else:
            for p in self._ports:
                print(f"  {p.role.name}:{p.name} [{p.side.name}]")
                print(f"    + {p.positive.via_ids}")
                print(f"    - {p.negative.via_ids}")
        print("====================")
