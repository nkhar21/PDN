from generator.pdn_visaul.plot_board import (plot_board_outline, plot_stackup,
                                    plot_vias_on_layer, plot_vias_cross_section, plot_all_layers)
import matplotlib.pyplot as plt
from generator.pdn_enums import PortSide
from generator.pdn_visaul.plot_board import plot_ports_on_layer

class PDNPlotter:
    """
    Visualization utilities for a PDNBoard.
    Holds a reference to a PDNBoard and provides plotting helpers.
    """

    def __init__(self, board):
        self.board = board

    def plot_outline(self, layer: int = 0, **kwargs):
        if self.board.outline.bxy is None:
            raise ValueError("Outline must be defined before plotting")
        if self.board.stackup is None:
            raise ValueError("Stackup must be defined before plotting")
        return plot_board_outline(
            self.board.outline.bxy,
            stackup_mask=self.board.stackup.stackup_mask,
            layer=layer,
            **kwargs
        )

    def plot_stackup(self, **kwargs):
        if self.board.stackup is None:
            raise ValueError("Stackup must be defined before plotting")
        return plot_stackup(
            self.board.stackup,
            stackup_mask=self.board.stackup.stackup_mask,
            **kwargs
        )

    def plot_vias_on_layer(self, layer: int = 0, **kwargs):
        if self.board.outline.bxy is None:
            raise ValueError("Outline must be defined before plotting")
        if not self.board.vias.vias:
            raise ValueError("No vias defined for this board")
        if self.board.stackup is None:
            raise ValueError("Stackup must be defined before plotting")
        return plot_vias_on_layer(
            self.board.outline.bxy,
            self.board.vias,
            layer=layer,
            stackup_mask=self.board.stackup.stackup_mask,
            **kwargs
        )

    def plot_vias_cross_section(self, vias=None, **kwargs):
        if self.board.stackup is None:
            raise ValueError("Stackup must be defined before plotting")
        vias_to_plot = vias if vias is not None else self.board.vias
        return plot_vias_cross_section(
            self.board.stackup,
            vias=vias_to_plot,
            stackup_mask=self.board.stackup.stackup_mask,
            **kwargs
        )

    def plot_every_layer(self, layers=None, show_ports: bool = True, **kwargs):
        if self.board.outline.bxy is None:
            raise ValueError("Outline must be defined before plotting")
        if not self.board.vias.vias:
            raise ValueError("No vias defined for this board")
        if self.board.stackup is None:
            raise ValueError("Stackup must be defined before plotting")

        return plot_all_layers(
            self.board.outline.bxy,
            self.board.vias,
            layers=layers,
            ports=(self.board.ports if show_ports else None),
            overlay_ports_on_outer=True,
            stackup_mask=self.board.stackup.stackup_mask,
            **kwargs
        )



    def plot_ports_on_side(self, side: PortSide, ax: plt.Axes | None = None, **kwargs):
        if self.board.outline.bxy is None:
            raise ValueError("Outline must be defined before plotting")
        if self.board.stackup is None:
            raise ValueError("Stackup must be defined before plotting")
        return plot_ports_on_layer(
            self.board.outline.bxy,
            self.board.vias,
            self.board.ports,
            stackup_mask=self.board.stackup.stackup_mask,
            side=side,
            ax=ax,
            **kwargs
        )
