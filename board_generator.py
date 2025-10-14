# board_example.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from generator.gen_outline import OutlineMode
from generator.pdn_board import PDNBoard
from generator.pdn_via.pdn_via_model import Via
from generator.pdn_enums import NetType, PortRole, PortSide
from generator.pdn_stackup import Stackup
from generator.pdn_visaul.pdn_plotter import PDNPlotter

if __name__ == "__main__":
    # 1) Board shell
    board = PDNBoard()

    # 2) Outline (mm → meters inside outline code)
    square = np.array([[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]], dtype=float)
    board.set_outline([square], mode=OutlineMode.COLLAPSED, units="mm")

    # 3) Segment outline (meters)
    board.set_segmentation(seg_len=0.001)  # 1 mm

    # 4) Stackup (4 signal layers): 0=GND, 1=PWR, 2=GND, 3=GND
    stackup_mask = [0, 1, 0, 0]
    die_t        = [0.0005, 0.00043, 0.00047]   # m
    er_list      = [4.0, 3.43, 3.8]
    d_r          = [0.000035] * 4               # m

    stackup = Stackup(
        num_layers=4,
        stackup_mask=stackup_mask,
        die_t=die_t,
        er_list=er_list,
        d_r=d_r,
    )
    board.set_stackup(stackup)
    print(f"Stackup: {stackup}")

    # 5) Define vias (meters)
    #   For ports: + must be PWR and touch the chosen outer side; - must be GND and touch that side.
    v1 = Via(xy=(0.010, 0.020), start_layer=3, stop_layer=0, via_type=NetType.GND)  # THROUGH → touches TOP & BOTTOM
    v2 = Via(xy=(0.030, 0.040), start_layer=1, stop_layer=2, via_type=NetType.PWR)  # BURIED → not usable for ports
    v3 = Via(xy=(0.020, 0.030), start_layer=0, stop_layer=1, via_type=NetType.PWR)  # BLIND TOP → usable for TOP
    v4 = Via(xy=(0.040, 0.010), start_layer=2, stop_layer=3, via_type=NetType.GND)  # BLIND BOTTOM → usable for BOTTOM
    v5 = Via(xy=(0.035, 0.012), start_layer=1, stop_layer=3, via_type=NetType.PWR)  # BLIND BOTTOM → usable for BOTTOM

    print("\n=== Vias BEFORE adding to board (roles unset) ===")
    for v in [v1, v2, v3, v4, v5]:
        print(v)

    # 6) Add vias (returns ids)
    vid1 = board.add_via(v1)
    vid2 = board.add_via(v2)
    vid3 = board.add_via(v3)
    vid4 = board.add_via(v4)
    vid5 = board.add_via(v5)

    print("\n=== Vias AFTER adding to board (roles assigned) ===")
    board.vias.summary()

    # 7) Ports (by via ids, side-aware validation)
    board.add_port_by_ids(
        name="ic_port",
        role=PortRole.IC,
        side=PortSide.TOP,
        pos_via_ids=[vid3],  # PWR touching TOP
        neg_via_ids=[vid1],  # GND touching TOP (through)
    )
    board.add_port_by_ids(
        name="decap1",
        role=PortRole.DECAP,
        side=PortSide.BOTTOM,
        pos_via_ids=[vid5],  # PWR touching BOTTOM
        neg_via_ids=[vid4],  # GND touching BOTTOM
    )

    # 8) Summaries
    board.summary()

    # 9) Exports (do these BEFORE plotting per your request)
    print("\nFinal via array for solver (id,x,y,start,stop,type,role):")
    print(board.vias.to_numpy())

    print("\nPort map by indices (good for matrix ops):")
    print(board.export_port_map(by="index"))

    print("\nPort map by IDs (good for traceability/logging):")
    print(board.export_port_map(by="id"))

    # 10) Plot LAST: 2×3 grid: 4 layers (ports overlaid on top/bottom) + stackup cross-section
    plotter = PDNPlotter(board)
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1.2], figure=fig)

    axes_layers = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    top_idx = 0
    bot_idx = board.stackup.num_layers - 1

    for i, ax in enumerate(axes_layers):
        if i == top_idx:
            # overlay ports on TOP
            plotter.plot_ports_on_side(PortSide.TOP, ax=ax, show_ids=True)
        elif i == bot_idx:
            # overlay ports on BOTTOM
            plotter.plot_ports_on_side(PortSide.BOTTOM, ax=ax, show_ids=True)
        else:
            # interior layers: vias only
            plotter.plot_vias_on_layer(layer=i, ax=ax)
        ax.set_title(f"Layer {i}")

    ax_stackup = fig.add_subplot(gs[:, 2])
    plotter.plot_vias_cross_section(ax=ax_stackup)
    ax_stackup.set_title("Stackup Cross-Section")

    fig.suptitle("Board Outlines with Vias + Stackup + Ports", fontsize=14)
    plt.tight_layout()
    plt.show()
