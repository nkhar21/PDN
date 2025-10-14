from __future__ import annotations
from typing import Protocol
from generator.pdn_stackup import Stackup
from generator.pdn_via.pdn_via_model import ViaCollection




class BoardView(Protocol):
    stackup: Stackup
    vias: ViaCollection
