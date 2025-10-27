import matplotlib.pyplot as plt
import numpy as np


def plot_z_matrix(freq, z_matrix, indices, ax=None, **kwargs):
    """
    Plot selected Z-matrix elements with flexible kwargs.

    Parameters
    ----------
    freq : ndarray
        Frequency vector (Hz).
    z_matrix : ndarray (nf, n_ports, n_ports)
        Frequency-dependent Z-matrix.
    indices : tuple or list of tuples
        Which (i,j) entries of Z to plot.
    ax : matplotlib.axes.Axes or None
        If provided, plot on this axis; otherwise create new.
    kwargs : dict
        Flexible options:
          - scale=("log","log")
          - xlabel="Frequency (Hz)"
          - ylabel="|Z| (Ω)"
          - title="Some title"
          - legend=["curve1", "curve2"]
          - dB=True/False
          - grid=True/False
          - tight=True/False
          - plus all matplotlib.plot() kwargs like color, linestyle, marker, linewidth
    """

    # Defaults
    scale = kwargs.pop("scale", ("log", "log"))
    xlabel = kwargs.pop("xlabel", "Frequency (Hz)")
    ylabel = kwargs.pop("ylabel", "|Z| (Ω)")
    title = kwargs.pop("title", None)
    legend = kwargs.pop("legend", None)
    dB = kwargs.pop("dB", False)
    grid = kwargs.pop("grid", True)
    tight = kwargs.pop("tight", True)

    if isinstance(indices, tuple):
        indices = [indices]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for k, (i, j) in enumerate(indices):
        data = np.abs(z_matrix[:, i, j])
        if dB:
            data = 20 * np.log10(data + 1e-20)
        label = legend[k] if legend and k < len(legend) else f"Z{i+1}{j+1}"
        ax.plot(freq, data, label=label, **kwargs)

    ax.set_xscale(scale[0])
    ax.set_yscale(scale[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel + (" (dB)" if dB else ""))
    if title:
        ax.set_title(title)

    if legend:
        ax.legend()
    if grid:
        ax.grid(True, which="both", ls="--", alpha=0.7)
    if tight and ax is None:
        plt.tight_layout()

    return fig, ax
