# main_decap.py
# Compare IC Z11 with REAL decaps for both:
# (A) PowerSI 3-port Touchstone and (B) Your Python BEM/CIM/NVM Z.
# Frequency: 10 kHz – 200 MHz (log), just show the plot (no saving).

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import skrf as rf

from BEM_NVM import PDN
from CIM_NVM import main_res
from input_AH import input_path, stackup_path, touchstone_path

from pdn_io.spd_parser import parse_spd
from pdn_io.stackup_parser import read_stackup
from utils.plotting import plot_z_matrix
from utils.ztool import short_1port, connect_1decap

# ------------------ Config ------------------
FSTART  = 10e3          # 10 kHz
FSTOP   = 200e6         # 200 MHz
NPOINTS = 201           # log points

# Choose capacitor S2P files (vendor "series" models)
DECAP_FILES = {
    1: Path(r"capacitors\1u - GRM155C71C105ME11_DC0V_25degC_series.s2p"),     # decap @ port 1
    2: Path(r"capacitors\100n - GCM155R71H104KE02_DC0V_25degC_series.s2p"),   # decap @ port 2
}

TITLE = "IC Z11 with Real Decaps (PowerSI vs Python) — 10 kHz–200 MHz"


# ----------------- Helpers ------------------
def gen_brd_data(brd, spd_path: str, stackup_path: str, d: float = 1e-3):
    """Build board state and compute Z using your existing parsers and solvers."""
    parse_spd(
        brd, spd_path,
        ground_net="gnd", power_net="pwr",
        ic_port_tag="ic_port", decap_port_tag="decap_port",
        verbose=True
    )

    brd.sxy = np.concatenate([brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy], axis=0)
    brd.sxy_index_ranges, offset = [], 0
    for b in brd.bxy:
        n_seg = brd.seg_bd_node(b, d).shape[0]
        brd.sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg
    brd.sxy_list = [brd.seg_bd_node(b, d) for b in brd.bxy]

    die_t, er_list, d_r = read_stackup(stackup_path)
    brd.er_list, brd.die_t, brd.d_r = er_list, die_t, d_r

    res_matrix = main_res(brd=brd)
    z = brd.calc_z_fast(res_matrix=res_matrix, verbose=True)  # shape: (nf, 3, 3)
    return z


def _cap_to_shunt_z11(path: Path, freq: rf.Frequency) -> np.ndarray:
    """
    Convert vendor series S2P -> 1-port shunt-to-ground on 'freq'.
    Returns Z11(f) with shape (nf, 1, 1).
    """
    cap2p = rf.Network(str(path)).interpolate(freq)  # align to our 10 kHz..200 MHz grid
    cap1p, _ = short_1port(cap2p, shorted_port=1)    # short far port -> shunt 1-port
    return cap1p.z


# ------------------- Main -------------------
if __name__ == '__main__':

    # Unified frequency grid: 10 kHz .. 200 MHz (log)
    freq = np.logspace(np.log10(FSTART), np.log10(FSTOP), NPOINTS)
    Freq = rf.Frequency(start=FSTART/1e6, stop=FSTOP/1e6, npoints=NPOINTS, unit='mhz', sweep_type='log')

    # --- Load PowerSI 3-port on this grid
    pdn_ts = rf.Network(touchstone_path).interpolate(Freq)   # 3-port: [IC(0), decap1(1), decap2(2)]
    Z_ts = pdn_ts.z                                          # (nf, 3, 3)
    nf, n, _ = Z_ts.shape
    assert n == 3, "Expected a 3-port PDN (IC=0, decap1=1, decap2=2) in touchstone."

    # --- Build Python Z (your solver) on the same grid
    brd = PDN()
    # force PDN() to use the same grid for its internal calc
    brd.fstart = FSTART; brd.fstop = FSTOP; brd.nf = NPOINTS
    brd.freq = rf.Frequency(start=FSTART/1e6, stop=FSTOP/1e6, npoints=NPOINTS, unit='mhz', sweep_type='log')

    Z_py = gen_brd_data(brd, input_path, stackup_path)       # expect (nf, 3, 3)
    assert Z_py.shape[:2] == (NPOINTS, 3), "Python Z must be (nf,3,3) on the same grid."

    # --- Load two capacitor models and convert to shunt 1-port Z11 on this grid
    for p in (1, 2):
        if not DECAP_FILES[p].exists():
            raise FileNotFoundError(f"Cap file for port {p} not found: {DECAP_FILES[p]}")
    Zcap = {
        1: _cap_to_shunt_z11(DECAP_FILES[1], pdn_ts.frequency),  # (nf,1,1)
        2: _cap_to_shunt_z11(DECAP_FILES[2], pdn_ts.frequency),
    }

    # --- Connect decaps to PowerSI Z and reduce to IC-only
    Z_ts_cur = deepcopy(Z_ts)
    port_map_ts = [0, 1, 2]

    idx_p1 = port_map_ts.index(1)
    Z_ts_cur, port_map_ts = connect_1decap(Z_ts_cur, port_map_ts, idx_p1, Zcap[1])

    idx_p2 = port_map_ts.index(2)
    Z_ts_cur, port_map_ts = connect_1decap(Z_ts_cur, port_map_ts, idx_p2, Zcap[2])

    assert len(port_map_ts) == 1 and port_map_ts[0] == 0, f"Unexpected touchstone port map: {port_map_ts}"
    # Z_ts_cur is (nf, 1, 1)

    # --- Connect decaps to Python Z and reduce to IC-only
    Z_py_cur = deepcopy(Z_py)
    port_map_py = [0, 1, 2]

    idx_p1_py = port_map_py.index(1)
    Z_py_cur, port_map_py = connect_1decap(Z_py_cur, port_map_py, idx_p1_py, Zcap[1])

    idx_p2_py = port_map_py.index(2)
    Z_py_cur, port_map_py = connect_1decap(Z_py_cur, port_map_py, idx_p2_py, Zcap[2])

    assert len(port_map_py) == 1 and port_map_py[0] == 0, f"Unexpected Python port map: {port_map_py}"
    # Z_py_cur is (nf, 1, 1)

    # --- Plot: 4 curves — open-others (PowerSI & Python) and with decaps (PowerSI & Python)
    fig, ax = plot_z_matrix(
        freq=freq,
        z_matrix=Z_ts,
        indices=(0, 0),
        legend=["PowerSI Z11 (others open)"],
        scale=("log", "log"),
        xlabel="Frequency (Hz)",
        ylabel="|Z| (Ω)",
        title=TITLE,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
    )

    _ = plot_z_matrix(
        freq=freq,
        z_matrix=Z_py,
        indices=(0, 0),
        legend=["Python Z11 (others open)"],
        color="tab:green",
        linestyle="-.",
        linewidth=2,
        ax=ax,
    )

    _ = plot_z_matrix(
        freq=freq,
        z_matrix=Z_ts_cur,       # (nf,1,1)
        indices=(0, 0),
        legend=["PowerSI Z11 (with decaps)"],
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        ax=ax,
    )

    _ = plot_z_matrix(
        freq=freq,
        z_matrix=Z_py_cur,       # (nf,1,1)
        indices=(0, 0),
        legend=["Python Z11 (with decaps)"],
        color="tab:red",
        linestyle="-",
        linewidth=2,
        ax=ax,
    )

    # small caption for which caps were used
    cap_note = f"Decaps: {DECAP_FILES[1].name} (port 1), {DECAP_FILES[2].name} (port 2)"
    fig.text(0.99, 0.02, cap_note, ha="right", va="bottom", fontsize=9)

    plt.show()
