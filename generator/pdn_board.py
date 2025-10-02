import numpy as np
from typing import Optional
from matplotlib.path import Path 

from generator.pdn_board_outline import BoardOutline
from generator.pdn_stackup import Stackup
from generator.plot_board import (
    plot_board_outline, plot_stackup,
    plot_vias_on_layer, plot_vias_cross_section, plot_layers
)
from generator.pdn_via import Via, ViaCollection, ViaRole
from generator.pdn_enums import NetType


class PDNBoard:
    """
    High-level PDN board representation.
    Holds outline, stackup, and via attributes.
    """

    def __init__(self):
        # --- Outline / Geometry ---
        self.outline: BoardOutline = BoardOutline()

        # --- Stackup ---
        self.stackup: Optional[Stackup] = None

        # --- Vias ---
        self.vias: ViaCollection = ViaCollection()

    # --------------------------
    # Delegation helpers
    # --------------------------
    def set_outline(self, *args, **kwargs):
        """Delegate to BoardOutline.set_outline"""
        return self.outline.set_outline(*args, **kwargs)

    def set_segmentation(self, *args, **kwargs):
        """Delegate to BoardOutline.set_segmentation"""
        return self.outline.set_segmentation(*args, **kwargs)

    def set_stackup(self, stackup: Stackup):
        """Attach a pre-constructed Stackup object."""
        if not isinstance(stackup, Stackup):
            raise TypeError("stackup must be a Stackup object")
        self.stackup = stackup

    def _point_in_polygon(self, point, polygon) -> bool:
        """Return True if (x,y) lies inside polygon."""
        return Path(polygon).contains_point(point)

    def add_via(self, via: Via):
        """Add a single via to the board with validity + stackup checks, auto-assign role."""
        if self.stackup is None:
            raise ValueError("Stackup must be defined before adding vias")

        num_layers = self.stackup.num_layers
        stackup_mask = self.stackup.stackup_mask

        # --- Vertical check ---
        if via.start_layer < 0 or via.stop_layer >= num_layers:
            raise ValueError(
                f"Via layers {via.start_layer}->{via.stop_layer} out of bounds "
                f"(board has {num_layers} layers)"
            )

        # --- Horizontal check (geometry containment) ---
        if self.outline.bxy is None:
            raise ValueError("Board outline must be set before adding vias")

        if len(self.outline.bxy) == 1:
            # Collapsed: same polygon applies to all layers
            poly = self.outline.bxy[0]
            for layer in (via.start_layer, via.stop_layer):
                if not self._point_in_polygon(via.xy, poly):
                    raise ValueError(
                        f"Via at {via.xy} is outside board outline on layer {layer}"
                    )
        else:
            # Per-layer outlines
            for layer in (via.start_layer, via.stop_layer):
                poly = self.outline.bxy[layer]
                if not self._point_in_polygon(via.xy, poly):
                    raise ValueError(
                        f"Via at {via.xy} is outside board outline on layer {layer}"
                    )

        # --- Stackup mask check ---
        layer_types = {stackup_mask[via.start_layer], stackup_mask[via.stop_layer]}
        if via.via_type == NetType.PWR:
            if NetType.PWR.value not in layer_types:
                raise ValueError(
                    f"Power via at {via.xy} must connect to at least one PWR layer "
                    f"(layers {via.start_layer}->{via.stop_layer})"
                )
        elif via.via_type == NetType.GND:
            if NetType.GND.value not in layer_types:
                raise ValueError(
                    f"Ground via at {via.xy} must connect to at least one GND layer "
                    f"(layers {via.start_layer}->{via.stop_layer})"
                )

        # --- Auto-assign role ---
        if via.start_layer == 0 and via.stop_layer == num_layers - 1:
            via.role = ViaRole.THROUGH
        elif via.start_layer == 0 or via.stop_layer == num_layers - 1:
            via.role = ViaRole.BLIND
        else:
            via.role = ViaRole.BURIED

        # Add via to collection
        self.vias.add_via(via)
        if via.role == ViaRole.UNSET:
            raise RuntimeError(f"Via {via.xy} could not be assigned a role")


    # --------------------------
    # Debug helpers
    # --------------------------
    def summary(self):
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
        if self.vias and len(self.vias.vias) > 0:
            self.vias.summary()
        else:
            print("No vias defined.")

        print("========================")
