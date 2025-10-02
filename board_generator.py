import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from generator.gen_outline import OutlineMode
from generator.pdn_board import PDNBoard
from generator.pdn_via import Via
from generator.pdn_enums import NetType
from generator.pdn_stackup import Stackup
from generator.pdn_plotter import PDNPlotter   # <-- new import

if __name__ == "__main__":

    # -----------------------------
    # 1. Create empty board (4 signal layers)
    # -----------------------------
    board = PDNBoard()

    # -----------------------------
    # 2. Define outline
    # -----------------------------
    square = np.array([[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]], dtype=float)
    board.set_outline([square], mode=OutlineMode.COLLAPSED, units="mm")

    # -----------------------------
    # 3. Segment outline
    # -----------------------------
    board.set_segmentation(seg_len=0.001)  # 1 mm segments

    # -----------------------------
    # 4. Define stackup (4 signal, 3 dielectrics)
    # -----------------------------
    stackup_mask = [0, 1, 0, 0]  # GND-PWR-GND-GND
    die_t   = [0.0005, 0.00043, 0.00047]
    er_list = [4.0, 3.43, 3.8]
    d_r     = [0.000035]*4

    stackup = Stackup(num_layers=4,
                  stackup_mask=stackup_mask,
                  die_t=die_t,
                  er_list=er_list,
                  d_r=d_r)

    board.set_stackup(stackup)
    print(f"Stackup: {stackup}")

    # -----------------------------
    # 5. Create vias manually
    # -----------------------------
    v1 = Via(xy=(0.01, 0.02), start_layer=3, stop_layer=0, via_type=NetType.GND)
    v2 = Via(xy=(0.03, 0.04), start_layer=1, stop_layer=2, via_type=NetType.PWR)
    v3 = Via(xy=(0.02, 0.03), start_layer=0, stop_layer=1, via_type=NetType.PWR)
    v4 = Via(xy=(0.04, 0.01), start_layer=2, stop_layer=3, via_type=NetType.GND)

    print("\n=== Vias BEFORE adding to board (roles unset) ===")
    for v in [v1, v2, v3, v4]:
        print(v)

    # -----------------------------
    # 6. Add vias to board (auto-assigns role)
    # -----------------------------
    for v in [v1, v2, v3, v4]:
        board.add_via(v)

    print("\n=== Vias AFTER adding to board (roles assigned) ===")
    board.vias.summary()

    # -----------------------------
    # 7. Show board summary
    # -----------------------------
    board.summary()

    # -----------------------------
    # 8. Plots using PDNPlotter
    # -----------------------------
    plotter = PDNPlotter(board)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1.2], figure=fig)

    # --- Plot 4 layers on left (2x2 grid) ---
    axes_layers = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])
    ]

    for i, ax in enumerate(axes_layers):
        plotter.plot_vias_on_layer(layer=i, ax=ax)
        ax.set_title(f"Layer {i}")

    # --- Plot stackup cross-section on the right (spans both rows) ---
    ax_stackup = fig.add_subplot(gs[:, 2])
    plotter.plot_vias_cross_section(ax=ax_stackup)
    ax_stackup.set_title("Stackup Cross-Section")

    fig.suptitle("Board Outlines with Vias + Stackup", fontsize=14)
    plt.tight_layout()
    plt.show()


    # -----------------------------
    # 9. Export vias for solver
    # -----------------------------
    print("\nFinal via array for solver:")
    print(board.vias.to_numpy())
