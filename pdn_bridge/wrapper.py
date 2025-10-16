# pdn_bridge/wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Upstream classes from your repo
from BEM_AC_NVM_PDN import PDN
from generator.pdn_board import PDNBoard

# Local mappers
from .map_outline import apply_outline
from .map_stackup import apply_stackup   # <-- add this


@dataclass
class PDNFromBoardOptions:
    """Config for building a PDN object from a PDNBoard snapshot."""
    via_radius: float = 0.2e-3  # meters


class PDNFromBoard:
    """
    Wrapper that takes a **PDNBoard** (geometry-only) and populates a legacy **PDN**
    with the attributes required by `PDN.calc_z_fast(...)`.

    This class does *no* physics; it only maps fields and ensures the
    ordering/shape conventions expected by the PDN solver.
    """

    def __init__(self, board: PDNBoard, opts: Optional[PDNFromBoardOptions] = None):
        self.board = board
        self.opts = opts or PDNFromBoardOptions()

    def build(self) -> PDN:
        """
        Create and return a populated PDN instance ready for `calc_z_fast`.

        Implemented now:
          • Outline mapping (bxy, sxy, sxy_list, seg_len)
          • Stackup mapping (stackup, die_t, er_list, d_r)

        Stubs left for upcoming steps:
          • Vias, Ports
        """
        pdn = PDN()

        # Ensure attributes that calc_z_fast expects exist, even if the
        # original PDN.__init__ didn't declare them (safe in Python).
        if not hasattr(pdn, "sxy_list"):
            pdn.sxy_list = []  # type: ignore[attr-defined]

        # Global options
        pdn.via_r = float(self.opts.via_radius)

        # --- Outline / segmentation ---
        apply_outline(self.board, pdn)

        # --- Stackup ---
        apply_stackup(self.board, pdn)

        # --- Vias, Ports (to be added next) ---
        # apply_vias(self.board, pdn)
        # apply_ports(self.board, pdn)

        return pdn
