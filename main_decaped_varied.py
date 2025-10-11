# main_decaped_varied.py
# 2-port PDN from PowerSI (port 0 = IC, port 1 = decap node).
# Sweep five vendor capacitors (series S2P) connected from decap port to GND.
# Plot the IC |Z11| for each case on one figure (10 kHz – 200 MHz), using plot_z_matrix.

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
import skrf as rf

from BEM_AC_NVM_PDN import short_1port, connect_1decap
from input_AH import touchstone_path, stackup_path
from pdn_analysis.plotting import plot_z_matrix

# ---------------- config ----------------
FSTART, FSTOP, NPOINTS = 10e3, 200e6, 201  # 10 kHz .. 200 MHz, log
# the five capacitors to compare (your filenames)
DECAP_FILES = [
    Path(r"capacitors\47u - GCM32ED70E107ME36_DC0V_25degC_series.s2p"),
    Path(r"capacitors\10u - GRT21BC71C106KE13_DC0V_25degC_series.s2p"),
    Path(r"capacitors\1u - GRM155C71C105ME11_DC0V_25degC_series.s2p"),
    Path(r"capacitors\220n - GCM21BR71H224KA37_DC0V_25degC_series.s2p"),
    Path(r"capacitors\100n - GCM155R71H104KE02_DC0V_25degC_series.s2p"),
]
TITLE = "IC |Z11| with Varied Decaps on Decap Port (PowerSI 2-port) — 10 kHz–200 MHz"


# ------------- helpers -------------
def _resolve(p: Path | str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (Path(__file__).parent / p).resolve()

def _cap_to_shunt_z11(series_s2p_path: Path, freq: rf.Frequency) -> np.ndarray:
    """
    Convert vendor 'series' S2P into a 1-port shunt model on 'freq'.
    Returns Z11(f) with shape (nf,1,1).
    """
    cap2p = rf.Network(str(series_s2p_path)).interpolate(freq)   # align grids
    cap1p, _ = short_1port(cap2p, shorted_port=1)                # short 'far' port -> shunt 1-port
    return cap1p.z                                               # (nf,1,1)


# --------------- main ---------------
if __name__ == "__main__":

    # frequency grid (no DC point)
    freq = np.logspace(np.log10(FSTART), np.log10(FSTOP), NPOINTS)
    Freq = rf.Frequency(start=FSTART/1e6, stop=FSTOP/1e6, npoints=NPOINTS,
                        unit="mhz", sweep_type="log")

    # load the 2-port PDN from PowerSI and put it on this grid
    ts_path = _resolve(touchstone_path)
    ntw = rf.Network(str(ts_path)).interpolate(Freq)
    Z2 = ntw.z
    nf, n, _ = Z2.shape
    assert n == 2, f"Expected a 2-port Touchstone (IC, decap). Got n={n}."

    # for each capacitor: build shunt Z, connect at port 1 (decap), reduce to IC (port 0)
    curves = []   # list of (label, Zred)
    for cap_path in DECAP_FILES:
        cap_path = _resolve(cap_path)
        if not cap_path.exists():
            raise FileNotFoundError(f"Capacitor file not found:\n  {cap_path}")

        Zcap = _cap_to_shunt_z11(cap_path, ntw.frequency)    # (nf,1,1)

        Zcur = deepcopy(Z2)
        port_map = [0, 1]                                    # current index -> original

        # connect at the decap port (original port 1)
        idx_dec = port_map.index(1)
        Zred, port_map = connect_1decap(Zcur, port_map, idx_dec, Zcap)  # -> (nf,1,1), keep IC
        assert len(port_map) == 1 and port_map[0] == 0

        curves.append((cap_path.name, Zred))

    # ---------- plot all five on one figure ----------
    # start the figure with the first curve via your helper
    first_label, first_Z = curves[0]
    fig, ax = plot_z_matrix(
        freq=freq,
        z_matrix=first_Z,       # (nf,1,1)
        indices=(0, 0),
        legend=[first_label],
        scale=("log", "log"),
        xlabel="Frequency (Hz)",
        ylabel="|Z| (Ω)",
        title=TITLE,
        color="tab:blue",
        linestyle="-",
        linewidth=2,
    )

    # add the rest
    COLORS = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    STYLES = ["-", "--", "-.", ":"]
    for (label, Zred), c, ls in zip(curves[1:], COLORS, STYLES):
        _ = plot_z_matrix(
            freq=freq,
            z_matrix=Zred,
            indices=(0, 0),
            legend=[label],
            color=c,
            linestyle=ls,
            linewidth=2,
            ax=ax,
        )

    # bottom captions (centered)
    board_text = f"Board: {Path(stackup_path).parent.name}"
    fig.text(0.5, 0.03, board_text, ha="center", va="bottom", fontsize=9)

    plt.show()
