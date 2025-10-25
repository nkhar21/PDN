import re
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import skrf as rf

# from BEM_AC_NVM_PDN import PDN
from BEM_NVM import PDN
from input_AH import input_path, stackup_path, touchstone_path

from pdn_io.spd_parser import parse_spd
from pdn_io.stackup_parser import read_stackup
from pdn_analysis.ztool import short_ports_reduce_z
from pdn_analysis.plotting import plot_z_matrix


def gen_brd_data(brd, spd_path: str, stackup_path: str, d: float = 1e-3):
    """
    Build board state and compute Z using external parsers for SPD and stackup/layer-type.
    """
    # --- 1) Parse SPD (board shapes, vias, layers, etc.) ---
    result = parse_spd(brd, spd_path, verbose=True)

    # --- 2) Board boundary segments ---
    brd.sxy = np.concatenate([brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy], axis=0)
    brd.sxy_index_ranges = []
    offset = 0
    for b in brd.bxy:
        n_seg = brd.seg_bd_node(b, d).shape[0]
        brd.sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg
    brd.sxy_list = [brd.seg_bd_node(b, d) for b in brd.bxy]

    # --- 3) Stackup & layer-type ---
    die_t, er_list, d_r = read_stackup(stackup_path)
    brd.er_list, brd.die_t, brd.d_r = er_list, die_t, d_r

    # --- 4) Compute Z-matrix ---
    z = brd.calc_z_fast(res_matrix=None, verbose=True)

    return (
        z,
        brd.bxy,
        brd.ic_via_xy,
        brd.ic_via_type,
        brd.decap_via_xy,
        brd.decap_via_type,
        brd.decap_via_loc,
        brd.stackup,
        brd.die_t,
        brd.sxy_list,
        brd.buried_via_xy,
        brd.buried_via_type,
    )


def detect_ic_port_index(touchstone_path: str) -> int:
    idx = None
    for ln in pathlib.Path(touchstone_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"!\s*Port(\d+).*ic_port", ln, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1  # zero-based
            break
    return idx if idx is not None else 2  # fallback to 2 (port 3) for b4_1.S3P


def compare_ic_z11_with_other_ports_shorted(touchstone_path, python_z, freq, ic_port_idx):
    """
    Compare IC input impedance (Z11) with all other ports shorted,
    between Python solver and Touchstone data.
    """
    try:
        # --- Load Touchstone and align frequencies ---
        Freq = rf.Frequency.from_f(freq, unit="hz")
        ntw = rf.Network(touchstone_path).interpolate(Freq, kind="linear")
        z_touchstone = ntw.z

        n_ports = python_z.shape[1]
        ports_to_short = [p for p in range(n_ports) if p != ic_port_idx]
        print(f"Shorting ports: {ports_to_short} (keeping IC port {ic_port_idx})")

        # --- Apply multi-port short reduction ---
        python_z_shorted, keep_py = short_ports_reduce_z(python_z, ports_to_short)
        touchstone_z_shorted, keep_ts = short_ports_reduce_z(z_touchstone, ports_to_short)

        # --- Should reduce to 1×1 network (Z11 for IC) ---
        if python_z_shorted.shape[1] != 1:
            raise RuntimeError("Reduction did not yield single-port Z matrix.")

        Zin_python = python_z_shorted[:, 0, 0]
        Zin_touch = touchstone_z_shorted[:, 0, 0]

        # --- Plot in ohms ---
        plt.figure()
        plt.semilogx(freq / 1e6, np.abs(Zin_python), label="Python Z11 (IC, others shorted)")
        plt.semilogx(freq / 1e6, np.abs(Zin_touch), "--", label="Touchstone Z11 (IC, others shorted)")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("|Z11| (Ω)")
        plt.title("IC Input Impedance with All Other Ports Shorted")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return Zin_python, Zin_touch

    except Exception as e:
        print(f"[ERROR] Failed to compare IC Z11 with shorts: {e}")
        return None, None


if __name__ == "__main__":

    BASE_PATH = "output/"
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    brd = PDN()

    # --- Generate Board Data ---
    result = gen_brd_data(brd=brd, spd_path=input_path, stackup_path=stackup_path)

    z, bxy, ic_via_xy, ic_via_type, decap_via_xy, decap_via_type, \
        decap_via_loc, stackup, die_t, sxy_list, buried_via_xy, buried_via_type = result

    if np.isnan(np.sum(z)):
        print("[Error] Z contains NaN values — skipping")

    # --- Detect IC port ---
    ic_port_index = detect_ic_port_index(touchstone_path)

    # --- Frequency setup ---
    fstart, fstop, nf = 10e3, 200e6, 201
    freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)

    # --- Compare IC Z11 (others shorted) ---
    compare_ic_z11_with_other_ports_shorted(
        touchstone_path=touchstone_path,
        python_z=z,
        freq=freq,
        ic_port_idx=ic_port_index,
    )
