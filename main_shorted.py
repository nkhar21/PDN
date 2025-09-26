import re
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from copy import deepcopy
from BEM_AC_NVM_PDN import PDN
import os
import skrf as rf
from input_AH import input_path, stackup_path, touchstone_path

from pdn_io.spd_parser import parse_spd
from pdn_io.stackup_parser import read_stackup
from pdn_analysis.ztool import short_1port_toolz, short_1port_z, short_ports_reduce_z
from pdn_analysis.plotting import plot_z_matrix

 
def gen_brd_data(
    brd,
    spd_path: str,
    stackup_path: str,
    d: float = 1e-3,
):
    """
    Build board state and compute Z using external parsers for SPD and stackup/layer-type.
    """
    # --- 1) Parse SPD (board shapes, vias, layers, etc.) ---
    result = parse_spd(brd, spd_path, verbose=True)

    # Accept new 9/11 tuple (stackup included)
    if len(result) == 9:
        (bxy, ic_via_xy, ic_via_type,
         start_layers, stop_layers, via_type,
         decap_via_xy, decap_via_type, stackup) = result
        brd.buried_via_xy, brd.buried_via_type = None, None
    elif len(result) == 11:
        (bxy, ic_via_xy, ic_via_type,
         start_layers, stop_layers, via_type,
         decap_via_xy, decap_via_type, stackup,
         buried_via_xy, buried_via_type) = result
        brd.buried_via_xy = buried_via_xy
        brd.buried_via_type = buried_via_type
    else:
        raise ValueError(f"Unexpected return count from parse_spd: {len(result)}")

    # --- 2) Board boundary segments (unchanged logic) ---
    bxy = np.array([np.round(np.array(item), 6) for item in bxy], dtype=object)
    brd.bxy = bxy

    brd.sxy = np.concatenate([brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy], axis=0)
    brd.sxy_index_ranges = []
    offset = 0
    for b in brd.bxy:
        n_seg = brd.seg_bd_node(b, d).shape[0]
        brd.sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg
    brd.sxy_list = [brd.seg_bd_node(b, d) for b in brd.bxy]


    # --- 3) Stackup & layer-type via external parsers ---
    die_t, er_list, d_r = read_stackup(stackup_path)
    brd.er_list = er_list
    brd.die_t = die_t
    brd.d_r = d_r

    # --- 4) Z computation (unchanged) ---
    z = brd.calc_z_fast(res_matrix=None, verbose=True)

    brd.buried_via_xy   = brd.buried_via_xy   if hasattr(brd, "buried_via_xy")   else None
    brd.buried_via_type = brd.buried_via_type if hasattr(brd, "buried_via_type") else None

    result_out = [
        z,
        brd.bxy,
        brd.ic_via_xy,
        brd.ic_via_type,
        brd.decap_via_xy,
        brd.decap_via_type,
        brd.decap_via_loc,
        brd.stackup,
        die_t,
        brd.sxy_list,
    ]

    if brd.buried_via_xy is not None and brd.buried_via_type is not None:
        result_out.append(brd.buried_via_xy)
        result_out.append(brd.buried_via_type)

    if len(result_out) == 10:
        return tuple(result_out) + (None, None)
    elif len(result_out) == 12:
        return tuple(result_out)
    else:
        raise ValueError(f"Unexpected number of return values from gen_brd_data: {len(result)}")


def compare_touchstone_with_python_z(touchstone_path, python_z, freq):
    try:
        ntw = rf.Network(touchstone_path)
        z_touchstone = ntw.z

        if not np.allclose(ntw.f, freq):

            ntw = ntw.interpolate(freq, kind='linear')
            z_touchstone = ntw.z

        n_ports = z_touchstone.shape[1]
        for i in range(n_ports):
            for j in range(n_ports):
                plt.figure()
                plt.plot(freq / 1e6, 20 * np.log10(np.abs(python_z[:, i, j]) + 1e-20), label='Python Z')
                plt.plot(freq / 1e6, 20 * np.log10(np.abs(z_touchstone[:, i, j]) + 1e-20), '--', label='Touchstone Z')

                plt.xlabel('Frequency (MHz)')
                plt.ylabel(f'|Z{i+1}{j+1}| (dBΩ)')
                plt.title(f'Comparison of Z{i+1}{j+1}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    except Exception as e:
        print(f"[ERROR] Failed to compare Touchstone and Python Z: {e}")

def compare_shorted_z(touchstone_path, python_z, freq, shorted_port_idx):
    """
    Compare Z-parameters after shorting a port between Python solver and Touchstone.
    
    freq : ndarray (Hz) — frequency array used in python_z
    """
    try:
        # Build scikit-rf Frequency object from given numpy freq array
        Freq = rf.Frequency.from_f(freq, unit='hz')

        # --- Load and interpolate Touchstone onto the same freq grid ---
        ntw = rf.Network(touchstone_path).interpolate(Freq, kind='linear')
        z_touchstone = ntw.z  # shape (201, n, n)

        # --- Short the port for each frequency ---
        python_z_shorted = np.array([short_1port_toolz(z, shorted_port_idx) for z in python_z])
        touchstone_z_shorted = np.array([short_1port_toolz(z, shorted_port_idx) for z in z_touchstone])

        # --- Plot comparison ---
        n_ports = python_z_shorted.shape[1]
        for i in range(n_ports):
            for j in range(n_ports):
                plt.figure()
                plt.semilogx(freq / 1e6,
                             20 * np.log10(np.abs(python_z_shorted[:, i, j]) + 1e-20),
                             label='Python Z (shorted)')
                plt.semilogx(freq / 1e6,
                             20 * np.log10(np.abs(touchstone_z_shorted[:, i, j]) + 1e-20),
                             '--',
                             label='Touchstone Z (shorted)')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel(f'|Z{i+1}{j+1}| (dBΩ)')
                plt.title(f'Comparison of Shorted Z{i+1}{j+1}')
                plt.legend()
                plt.grid(True, which='both', ls='--')
                plt.tight_layout()
                plt.show()

    except Exception as e:
        print(f"[ERROR] Failed to compare shorted Z: {e}")



def save2s(self, z, filename, path, z0=50):
    brd = rf.Network()
    brd.z0 = z0
    brd.frequency = self.freq
    brd.s = rf.network.z2s(z)
    brd.write_touchstone(path + filename + ".s" + str(z.shape[1]) + "p")
    freq = np.logspace(np.log10(10e6), np.log10(200e6), 201)

def detect_ic_port_index(touchstone_path: str) -> int:
    idx = None
    for ln in pathlib.Path(touchstone_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"!\s*Port(\d+).*ic_port", ln, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1  # zero-based
            break
    return idx if idx is not None else 2  # fallback to 2 (port 3) for b4_1.S3P


if __name__ == '__main__':

    N = 1 
    BASE_PATH = 'output/'
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    brd = PDN()


    result = gen_brd_data(
        brd=brd,
        spd_path=input_path,                  # e.g., "b4_1.spd"
        stackup_path=stackup_path,  # e.g., "b4_1_stackup.csv"
    )


    z, bxy, ic_via_xy, ic_via_type, decap_via_xy, decap_via_type, \
    decap_via_loc, stackup, die_t, sxy_list, buried_via_xy, buried_via_type = result

    if np.isnan(np.sum(z)):
        print("[Error] Z contains NaN values — skipping")

    ic_port_index = detect_ic_port_index(touchstone_path)
    
    fstart = 10e3
    fstop = 200e6
    nf = 201
    freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    Freq = rf.Frequency(start=fstart/1e6, stop=fstop/1e6, npoints=nf, unit='mhz', sweep_type='log')
    
    compare_shorted_z(
        touchstone_path=touchstone_path,
        python_z=z,
        freq=freq,
        shorted_port_idx=ic_port_index
    )
    


