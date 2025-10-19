from __future__ import annotations

from typing import Any, Optional, Tuple, Sequence, Literal
import numpy as np
from numpy.typing import NDArray

from matplotlib.path import Path

from generator.pdn_enums import NetType, PortRole, ViaRole, PortSide


from generator.pdn_outline import BoardOutline
from generator.pdn_stackup import Stackup

from generator.pdn_via.pdn_via_model import Via, ViaCollection
from generator.pdn_via.pdn_via_policy import validate_via_placement, assign_via_role

from generator.pdn_port.pdn_port_model import Port, PortCollection, Terminal
from generator.pdn_port.pdn_port_policy import validate_terminal_ids




# ----- Type aliases for clarity -----
Point = Tuple[float, float]               # (x, y)
Polygon = NDArray[np.float64]             # shape (N, 2), closed or open
Polygons = list[Polygon]


class PDNBoard:
    """
    High-level PDN board representation.
    Holds outline, stackup, and via attributes.
    """

    def __init__(self) -> None:
        # --- Outline / Geometry ---
        self.outline: BoardOutline = BoardOutline()

        # --- Stackup ---
        self.stackup: Optional[Stackup] = None

        # --- Vias ---
        self.vias: ViaCollection = ViaCollection()

        # --- Ports ---
        self.ports: PortCollection = PortCollection()

    # --------------------------
    # Delegation helpers
    # --------------------------
    def set_outline(self, *args: Any, **kwargs: Any) -> None:
        """Delegate to BoardOutline.set_outline"""
        self.outline.set_outline(*args, **kwargs)

    def set_segmentation(self, *args: Any, **kwargs: Any) -> None:
        """Delegate to BoardOutline.set_segmentation"""
        self.outline.set_segmentation(*args, **kwargs)

    def set_stackup(self, stackup: Stackup) -> None:
        """Attach a pre-constructed Stackup object."""
        if not isinstance(stackup, Stackup):
            raise TypeError("stackup must be a Stackup object")
        self.stackup = stackup

    def _point_in_polygon(self, point: Point | NDArray[np.floating], polygon: Polygon) -> bool:
        """Return True if (x,y) lies inside polygon."""
        # Normalize point to a 2-tuple[float, float] for Path.contains_point
        px: float = float(point[0])
        py: float = float(point[1])
        return Path(polygon).contains_point((px, py))

    def add_via(self, via: Via) -> int:
        """
        Validate, assign role, then add to collection. Returns the via id.
        """
        if self.stackup is None:
            raise ValueError("Stackup must be defined before adding vias")

        validate_via_placement(self, via)
        via.role = assign_via_role(via, self.stackup.num_layers)

        vid = self.vias.add_via(via)
        if via.role == ViaRole.UNSET:
            raise RuntimeError(f"Via {via.xy} could not be assigned a role")
        return vid

    def add_port_by_ids(
        self,
        name: str,
        role: PortRole,
        side: PortSide,
        pos_via_ids: Sequence[int],
        neg_via_ids: Sequence[int],
    ) -> Port:
        if self.stackup is None:
            raise ValueError("Stackup must be defined before port checks")

        validate_terminal_ids(self, pos_via_ids, NetType.PWR, side)
        validate_terminal_ids(self, neg_via_ids, NetType.GND, side)

        port = Port(
            name=name,
            role=role,
            side=side,
            positive=Terminal(list(pos_via_ids)),
            negative=Terminal(list(neg_via_ids)),
        )
        self.ports.add(port)
        return port

    def export_port_map(
        self,
        by: Literal["index", "id"] = "index",
    ) -> dict[str, dict[str, NDArray[np.int_]]]:
        """
        Build a mapping for solver wiring.
        Key: "{ROLE}:{NAME}" → {"positive": array([...]), "negative": array([...])}

        - by="index": returns via indices (good for matrix ops)
        - by="id":    returns via IDs (traceable in logs)
        """
        port_map: dict[str, dict[str, NDArray[np.int_]]] = {}
        for p in self.ports:  # PortCollection is iterable
            key = f"{p.role.name}:{p.name}"
            if by == "index":
                pos = self.vias.ids_to_indices(p.positive.via_ids)
                neg = self.vias.ids_to_indices(p.negative.via_ids)
            else:
                pos = np.asarray(p.positive.via_ids, dtype=np.int_)
                neg = np.asarray(p.negative.via_ids, dtype=np.int_)
            port_map[key] = {"positive": pos, "negative": neg}
        return port_map


    # --------------------------
    # Debug helpers
    # --------------------------
    def summary(self) -> None:
        """Print a quick summary of the board."""
        print("=== PDNBoard Summary ===")

        # Outline / Geometry
        if self.outline is not None:
            self.outline.summary()
        else:
            print("Outline: not defined")

        # Stackup
        if self.stackup is not None:
            self.stackup.summary()
        else:
            print("Stackup: not defined")

        print("========================")

        # Vias
        if len(self.vias.vias) > 0:
            self.vias.summary()
        else:
            print("No vias defined.")

        print("========================")

        # Ports
        if len(self.ports) > 0:
            self.ports.summary()
        else:
            print("No ports defined.")

        print("========================")
