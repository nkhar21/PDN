import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from generator.pdn_stackup import Stackup
from generator.pdn_via import Via, ViaCollection
import math


import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from generator.pdn_stackup import Stackup
from generator.pdn_via import Via, ViaCollection
import math


def plot_board_outline(
    bxy: np.ndarray,
    *,
    layer: int = 0,
    ax: plt.Axes = None,
    stackup_mask: np.ndarray,
    **kwargs
):
    """
    Plot a single board outline polygon from PDN().bxy, 
    with color chosen by stackup_mask (GND=black, PWR=red).

    Parameters
    ----------
    bxy : np.ndarray[dtype=object]
        Board outlines, shape (N_layers,). Each entry is (M,2) polygon in meters.
    layer : int, default=0
        Index of the layer to plot. Must be < len(bxy).
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure/axes is created.
    stackup_mask : np.ndarray
        Array of NetType values per layer (0=GND, 1=PWR).
    **kwargs : dict
        Extra keyword args forwarded to plt.plot (e.g., linestyle).
    """
    if not isinstance(bxy, np.ndarray) or bxy.dtype != object:
        raise TypeError("bxy must be a numpy.ndarray[dtype=object], as produced by OutlineGenerator")

    if len(bxy) == 1:
        polygon = bxy[0]  # collapsed outline: same polygon for all layers
    else:
        if layer < 0 or layer >= len(bxy):
            raise IndexError(f"Layer index {layer} out of range for bxy with {len(bxy)} layers")
        polygon = bxy[layer]

    if not isinstance(polygon, np.ndarray) or polygon.shape[1] != 2:
        raise ValueError("Each bxy entry must be a (N,2) ndarray representing xy coordinates")

    if ax is None:
        fig, ax = plt.subplots()

    outline_color = "red" if stackup_mask[layer] == 1 else "black"
    x, y = polygon[:, 0] * 1e3, polygon[:, 1] * 1e3  # convert to mm
    ax.plot(x, y, color=outline_color, linewidth=1.5,
            **{k: v for k, v in kwargs.items() if k != "color"})

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"Board Outline - Layer {layer} ({'PWR' if stackup_mask[layer] == 1 else 'GND'})"
    )

    return ax


def plot_vias_on_layer(
    bxy: np.ndarray,
    vias,
    *,
    layer: int,
    stackup_mask: np.ndarray,
    ax: plt.Axes = None,
    **kwargs
):
    """
    Plot vias on top of a board outline for a given layer.

    Parameters
    ----------
    bxy : np.ndarray[dtype=object]
        Board outlines, shape (N_layers,), each entry (M,2) in meters.
    vias : ViaCollection
        Collection of vias from PDNBoard.
    layer : int
        Which layer index to plot.
    stackup_mask : np.ndarray
        NetType mask per conductor layer (0=GND, 1=PWR).
    ax : matplotlib.axes.Axes, optional
        Axis from plot_board_outline. If None, one will be created.
    **kwargs : dict
        Extra style overrides for markers.
    """
    # First plot the outline (in mm) with mask coloring
    ax = plot_board_outline(bxy, layer=layer, ax=ax, stackup_mask=stackup_mask, **kwargs)

    # Marker shapes
    marker_start = "^"
    marker_end = "v"
    marker_thru = "o"

    # Colors for via types
    type_colors = {"PWR": "red", "GND": "black"}

    for v in vias.vias:
        x_mm, y_mm = v.xy[0] * 1e3, v.xy[1] * 1e3
        color = type_colors[v.via_type.name]

        if v.start_layer == layer and v.stop_layer == layer:
            ax.scatter(x_mm, y_mm, marker=marker_start, c=color, s=60,
                       label=f"{v.via_type.name} via (start=end)")
        elif v.start_layer == layer:
            ax.scatter(x_mm, y_mm, marker=marker_start, c=color, s=60,
                       label=f"{v.via_type.name} via start")
        elif v.stop_layer == layer:
            ax.scatter(x_mm, y_mm, marker=marker_end, c=color, s=60,
                       label=f"{v.via_type.name} via end")
        elif v.start_layer < layer < v.stop_layer:
            ax.scatter(x_mm, y_mm, marker=marker_thru, edgecolors=color,
                       facecolors="none", s=40,
                       label=f"{v.via_type.name} via through")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    return ax


def plot_layers(
    bxy: np.ndarray,
    vias,
    *,
    stackup_mask: np.ndarray,
    layers: List[int] = None,
    figsize=(12, 8),
    **kwargs
):
    """
    Plot vias + board outlines for multiple layers in subplots.

    Parameters
    ----------
    bxy : np.ndarray[dtype=object]
        Board outlines, shape (N_layers,), each entry (M,2) in meters.
    vias : ViaCollection
        Collection of vias from PDNBoard.
    stackup_mask : np.ndarray
        NetType mask per conductor layer (0=GND, 1=PWR).
    layers : list[int], optional
        List of layer indices to plot. If None, plot all layers.
    figsize : tuple, default (12, 8)
        Figure size in inches.
    **kwargs : dict
        Extra style arguments for via markers or outline.
    """
    num_layers = len(stackup_mask)
    if layers is None:
        layers = list(range(num_layers))

    n = len(layers)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, layer in enumerate(layers):
        ax = axes[i]
        plot_vias_on_layer(bxy, vias, layer=layer, stackup_mask=stackup_mask, ax=ax, **kwargs)
        ax.set_title(f"Layer {layer}")

        ax.annotate(
            f"L{layer}",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7),
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_stackup(
    stackup: Stackup,
    *,
    stackup_mask: np.ndarray,
    ax: plt.Axes = None,
    **kwargs
):
    """
    Plot a cross-section of the PCB stackup.
    Each conductor layer is colored by stackup_mask (GND=black, PWR=red).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 6))

    die_t = stackup.die_t[::-1]
    er_list = stackup.er_list[::-1]
    d_r = stackup.d_r[::-1]
    mask_rev = stackup_mask[::-1]

    current_y = 0.0
    for i, sig_thick in enumerate(d_r):
        outline_color = "red" if mask_rev[i] == 1 else "black"
        ax.fill_between(
            [-0.5, 0.5],
            current_y * 1e6,
            (current_y + sig_thick) * 1e6,
            color=outline_color,
            alpha=kwargs.get("signal_alpha", 0.7),
        )
        ax.text(
            0, (current_y + sig_thick / 2) * 1e6,
            f"Layer {len(d_r)-i-1}\n{sig_thick*1e6:.0f} µm",
            ha="center", va="center",
            fontsize=kwargs.get("fontsize", 8),
            color="white",
        )
        current_y += sig_thick

        if i < len(die_t):
            diel_thick = die_t[i]
            er = er_list[i]
            ax.fill_between(
                [-0.5, 0.5],
                current_y * 1e6,
                (current_y + diel_thick) * 1e6,
                color=kwargs.get("dielectric_color", "lightblue"),
                alpha=kwargs.get("dielectric_alpha", 0.5),
            )
            ax.text(
                0, (current_y + diel_thick / 2) * 1e6,
                f"εr={er:.2f}\n{diel_thick*1e6:.0f} µm",
                ha="center", va="center",
                fontsize=kwargs.get("fontsize", 8),
                color="black",
            )
            current_y += diel_thick

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, current_y * 1e6)
    ax.set_xticks([])
    ax.set_ylabel("Thickness [µm]")
    ax.set_title(kwargs.get("title", "PCB Stackup Cross-Section"))
    return ax


def plot_vias_cross_section(
    stackup: Stackup,
    vias: Union[ViaCollection, List[Via]],
    *,
    stackup_mask: np.ndarray,
    ax: plt.Axes = None,
    **kwargs
):
    """
    Plot a PCB stackup cross-section with vias as vertical connections.
    PWR vias = red, GND vias = black.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 6))

    ax = plot_stackup(stackup, stackup_mask=stackup_mask, ax=ax, **kwargs)

    d_r_rev = stackup.d_r[::-1]
    die_t_rev = stackup.die_t[::-1]
    n_layers = len(stackup.d_r)
    layer_bounds_um = [(0.0, 0.0)] * n_layers

    current_y_m = 0.0
    for i_top in range(n_layers):
        sig_thick_m = d_r_rev[i_top]
        y0_um = current_y_m * 1e6
        y1_um = (current_y_m + sig_thick_m) * 1e6
        orig_idx = n_layers - 1 - i_top
        layer_bounds_um[orig_idx] = (y0_um, y1_um)
        current_y_m += sig_thick_m
        if i_top < len(die_t_rev):
            current_y_m += die_t_rev[i_top]

    if isinstance(vias, ViaCollection):
        vias = vias.vias

    via_lw = kwargs.get("via_linewidth", 2)
    base_x = {"PWR": -0.15, "GND": +0.15}
    jitter_step = 0.015
    counts = {"PWR": 0, "GND": 0}

    for v in vias:
        if v.start_layer >= n_layers or v.stop_layer >= n_layers:
            continue
        y_start_um = layer_bounds_um[v.start_layer][0]
        y_end_um = layer_bounds_um[v.stop_layer][1]
        vtype = v.via_type.name
        color = "red" if vtype == "PWR" else "black"
        counts[vtype] += 1
        x = base_x[vtype] + (counts[vtype] - 1) * jitter_step
        ax.plot([x, x], [y_start_um, y_end_um],
                color=color, linewidth=via_lw, zorder=10, label=f"{vtype} via")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    return ax




def plot_stackup(
    stackup: Stackup,
    *,
    stackup_mask: np.ndarray,
    ax: plt.Axes = None,
    **kwargs,
):
    """
    Plot a cross-section of the PCB stackup.
    Each conductor layer is colored by stackup_mask (GND=black, PWR=red).

    Parameters
    ----------
    stackup : Stackup
        The stackup object (with die_t, er_list, d_r).
    stackup_mask : np.ndarray
        NetType mask per conductor layer (0=GND, 1=PWR).
    ax : matplotlib.axes.Axes, optional
        Existing axes. If None, a new one is created.
    **kwargs : dict
        Style options forwarded to ax.fill_between and ax.text
        (e.g., alpha, fontsize, etc.).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 6))

    # Flip arrays so plotting is from top to bottom
    die_t   = stackup.die_t[::-1]
    er_list = stackup.er_list[::-1]
    d_r     = stackup.d_r[::-1]
    mask_rev = stackup_mask[::-1]

    current_y = 0.0
    for i, sig_thick in enumerate(d_r):
        # --- Conductor (signal plane) ---
        outline_color = "red" if mask_rev[i] == 1 else "black"
        ax.fill_between(
            [-0.5, 0.5],
            current_y * 1e6,
            (current_y + sig_thick) * 1e6,
            color=outline_color,
            alpha=kwargs.get("signal_alpha", 0.7),
        )
        ax.text(
            0, (current_y + sig_thick / 2) * 1e6,
            f"Layer {len(d_r)-i-1}\n{sig_thick*1e6:.0f} µm",
            ha="center", va="center",
            fontsize=kwargs.get("fontsize", 8),
            color="white",
        )
        current_y += sig_thick

        # --- Dielectric ---
        if i < len(die_t):
            diel_thick = die_t[i]
            er = er_list[i]
            ax.fill_between(
                [-0.5, 0.5],
                current_y * 1e6,
                (current_y + diel_thick) * 1e6,
                color=kwargs.get("dielectric_color", "lightblue"),
                alpha=kwargs.get("dielectric_alpha", 0.5),
            )
            ax.text(
                0, (current_y + diel_thick / 2) * 1e6,
                f"εr={er:.2f}\n{diel_thick*1e6:.0f} µm",
                ha="center", va="center",
                fontsize=kwargs.get("fontsize", 8),
                color="black",
            )
            current_y += diel_thick

    # Style
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, current_y * 1e6)
    ax.set_xticks([])
    ax.set_ylabel("Thickness [µm]")
    ax.set_title(kwargs.get("title", "PCB Stackup Cross-Section"))

    return ax


def plot_vias_cross_section(
    stackup: Stackup,
    vias: Union[ViaCollection, List[Via]],
    *,
    stackup_mask: np.ndarray,
    ax: plt.Axes = None,
    **kwargs,
):
    """
    Plot a PCB stackup cross-section with vias as vertical connections (inside the stack).

    Y-axis is in micrometers (µm), matching plot_stackup.
    PWR vias = red, GND vias = black.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 6))

    # 1) Draw the stack background first (this sets limits, labels, etc.)
    ax = plot_stackup(stackup, stackup_mask=stackup_mask, ax=ax, **kwargs)

    # 2) Build per-layer bounds (bottom/top in µm)
    d_r_rev   = stackup.d_r[::-1]
    die_t_rev = stackup.die_t[::-1]
    n_layers = len(stackup.d_r)
    layer_bounds_um = [(0.0, 0.0)] * n_layers

    current_y_m = 0.0
    for i_top in range(n_layers):
        sig_thick_m = d_r_rev[i_top]
        y0_um = current_y_m * 1e6
        y1_um = (current_y_m + sig_thick_m) * 1e6
        orig_idx = n_layers - 1 - i_top
        layer_bounds_um[orig_idx] = (y0_um, y1_um)
        current_y_m += sig_thick_m
        if i_top < len(die_t_rev):
            current_y_m += die_t_rev[i_top]

    # 3) Normalize vias input
    if isinstance(vias, ViaCollection):
        vias = vias.vias

    # 4) Draw vias
    via_lw = kwargs.get("via_linewidth", 2)
    base_x = {"PWR": -0.15, "GND": +0.15}
    jitter_step = 0.015
    counts = {"PWR": 0, "GND": 0}

    for v in vias:
        if v.start_layer >= n_layers or v.stop_layer >= n_layers:
            continue

        y_start_um = layer_bounds_um[v.start_layer][0]
        y_end_um   = layer_bounds_um[v.stop_layer][1]

        vtype_name = v.via_type.name
        color = "red" if vtype_name == "PWR" else "black"

        counts[vtype_name] += 1
        x = base_x[vtype_name] + (counts[vtype_name] - 1) * jitter_step

        ax.plot(
            [x, x], [y_start_um, y_end_um],
            color=color, linewidth=via_lw, zorder=10,
            label=f"{vtype_name} via",
        )

    # 5) De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    return ax
