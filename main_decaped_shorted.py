# main_compare_all_in_one.py
# One figure with 6 curves:
#   PowerSI OPEN, Python OPEN, PowerSI SHORTED, Python SHORTED, PowerSI DECAP, Python DECAP
# Band: 10 kHz – 200 MHz (log). Plots only.

import re
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import skrf as rf
from copy import deepcopy

from BEM_AC_NVM_PDN import PDN, short_1port, connect_1decap
from input_AH import input_path, stackup_path, touchstone_path
from pdn_io.spd_parser import parse_spd
from pdn_io.stackup_parser import read_stackup
from pdn_analysis.ztool import short_ports_reduce_z
from pdn_analysis.plotting import plot_z_matrix

# ----------------- Config -----------------
FSTART, FSTOP, NPOINTS = 10e3, 200e6, 201
DECAP_FILES = {
    1: pathlib.Path(r"capacitors\1u - GRM155C71C105ME11_DC0V_25degC_series.s2p"),
    2: pathlib.Path(r"capacitors\100n - GCM155R71H104KE02_DC0V_25degC_series.s2p"),
}

# ---------------- Utilities ----------------
def gen_brd_data(brd, spd_path: str, stackup_path: str, d: float = 1e-3):
    parse_spd(brd, spd_path,
              ground_net="gnd", power_net="pwr",
              ic_port_tag="ic_port", decap_port_tag="decap_port",
              verbose=True)
    brd.sxy = np.concatenate([brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy], axis=0)
    brd.sxy_index_ranges = []
    offset = 0
    for b in brd.bxy:
        n_seg = brd.seg_bd_node(b, d).shape[0]
        brd.sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg
    brd.sxy_list = [brd.seg_bd_node(b, d) for b in brd.bxy]
    die_t, er_list, d_r = read_stackup(stackup_path)
    brd.er_list, brd.die_t, brd.d_r = er_list, die_t, d_r
    z = brd.calc_z_fast(res_matrix=None, verbose=True)   # (nf, 3, 3)
    return z

def detect_ic_port_index(snp_path: str) -> int:
    idx = None
    for ln in pathlib.Path(snp_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"!\s*Port(\d+).*ic_port", ln, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            break
    return idx if idx is not None else 0

def cap_to_shunt_z11(cap_file: pathlib.Path, freq: rf.Frequency) -> np.ndarray:
    cap2p = rf.Network(str(cap_file)).interpolate(freq)
    cap1p, _ = short_1port(cap2p, shorted_port=1)   # series 2-port -> shunt 1-port
    return cap1p.z                                  # (nf,1,1)

def connect_two_decaps(Z3p: np.ndarray, Zcap1: np.ndarray, Zcap2: np.ndarray) -> np.ndarray:
    Zcur = deepcopy(Z3p)
    port_map = [0, 1, 2]
    i1 = port_map.index(1)
    Zcur, port_map = connect_1decap(Zcur, port_map, i1, Zcap1)
    i2 = port_map.index(2)
    Zcur, port_map = connect_1decap(Zcur, port_map, i2, Zcap2)
    assert len(port_map) == 1 and port_map[0] == 0
    return Zcur  # (nf,1,1)

# ---------------- Main ----------------
if __name__ == "__main__":

    # Frequency grid
    freq = np.logspace(np.log10(FSTART), np.log10(FSTOP), NPOINTS)
    Freq = rf.Frequency(start=FSTART/1e6, stop=FSTOP/1e6, npoints=NPOINTS, unit='mhz', sweep_type='log')

    # Touchstone (PowerSI) on this grid
    ntw = rf.Network(touchstone_path).interpolate(Freq)
    Z_ts = ntw.z  # (nf,3,3)

    # Python 3-port Z on same grid
    brd = PDN()
    brd.fstart, brd.fstop, brd.nf = FSTART, FSTOP, NPOINTS
    brd.freq = Freq
    Z_py = gen_brd_data(brd=brd, spd_path=input_path, stackup_path=stackup_path)  # (nf,3,3)

    # IC index and ports to short
    ic_idx = detect_ic_port_index(touchstone_path)
    n_ports = Z_py.shape[1]
    ports_to_short = [p for p in range(n_ports) if p != ic_idx]

    # SHORTED reductions
    Z_ts_short, _ = short_ports_reduce_z(Z_ts, ports_to_short)  # (nf,1,1)
    Z_py_short, _ = short_ports_reduce_z(Z_py, ports_to_short)  # (nf,1,1)

    # DECAP reductions (two real caps)
    for p in (1, 2):
        if not DECAP_FILES[p].exists():
            raise FileNotFoundError(f"Capacitor file for port {p} not found: {DECAP_FILES[p]}")
    Zcap1 = cap_to_shunt_z11(DECAP_FILES[1], ntw.frequency)
    Zcap2 = cap_to_shunt_z11(DECAP_FILES[2], ntw.frequency)
    Z_ts_decap = connect_two_decaps(Z_ts, Zcap1, Zcap2)        # (nf,1,1)
    Z_py_decap = connect_two_decaps(Z_py, Zcap1, Zcap2)        # (nf,1,1)

    # --------- Single Figure: 6 curves ---------
    fig, ax = plot_z_matrix(
        freq=freq, z_matrix=Z_ts, indices=(ic_idx, ic_idx),
        legend=["PowerSI Z11 (OPEN)"],
        scale=("log", "log"),
        xlabel="Frequency (Hz)", ylabel="|Z| (Ω)",
        title="IC |Z11| — OPEN vs SHORTED vs DECAP (PowerSI & Python) — 10 kHz–200 MHz",
        color="tab:orange", linestyle="-", linewidth=2,
    )
    _ = plot_z_matrix(freq=freq, z_matrix=Z_py, indices=(ic_idx, ic_idx),
                      legend=["Python Z11 (OPEN)"],
                      color="tab:green", linestyle="-.", linewidth=2, ax=ax)
    _ = plot_z_matrix(freq=freq, z_matrix=Z_ts_short, indices=(0, 0),
                      legend=["PowerSI Z11 (SHORTED)"],
                      color="tab:cyan", linestyle="-", linewidth=2, ax=ax)
    _ = plot_z_matrix(freq=freq, z_matrix=Z_py_short, indices=(0, 0),
                      legend=["Python Z11 (SHORTED)"],
                      color="purple", linestyle="--", linewidth=2, ax=ax)
    _ = plot_z_matrix(freq=freq, z_matrix=Z_ts_decap, indices=(0, 0),
                      legend=[f"PowerSI Z11 (DECAP: {DECAP_FILES[1].name.split(' - ')[0]}, {DECAP_FILES[2].name.split(' - ')[0]})"],
                      color="tab:blue", linestyle="-", linewidth=2, ax=ax)
    _ = plot_z_matrix(freq=freq, z_matrix=Z_py_decap, indices=(0, 0),
                      legend=[f"Python Z11 (DECAP: {DECAP_FILES[1].name.split(' - ')[0]}, {DECAP_FILES[2].name.split(' - ')[0]})"],
                      color="tab:red", linestyle="--", linewidth=2, ax=ax)

    # small caption
    cap_note = f"Decaps used: {DECAP_FILES[1].name} (port 1), {DECAP_FILES[2].name} (port 2)"
    fig.text(0.5, 0.03, cap_note, ha="center", va="bottom", fontsize=11)
    cap_note = f"Board: {pathlib.Path(stackup_path).parent.name}"
    fig.text(0.5, 0.01, cap_note, ha="center", va="bottom", fontsize=9)
    plt.show()
