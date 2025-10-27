import numpy as np
import matplotlib.pyplot as plt

from BEM_NVM import PDN
from CIM_NVM import main_res
import time
import os
import skrf as rf
from input_AH import input_path, stackup_path, touchstone_path

from pdn_io.spd_parser import parse_spd
from pdn_io.stackup_parser import read_stackup
from utils.plotting import plot_z_matrix
 
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
    result = parse_spd(brd, 
                       spd_path, ground_net="gnd", power_net="pwr", 
                       ic_port_tag="ic_port", decap_port_tag="decap_port",
                       verbose=True)
    
    # --- 2) Board boundary segmentation (unchanged logic) ---
    brd.sxy = np.concatenate([brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy], axis=0)
    brd.sxy_index_ranges = []
    offset = 0
    for b in brd.bxy:
        n_seg = brd.seg_bd_node(b, d).shape[0]
        brd.sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg
    brd.sxy_list = [brd.seg_bd_node(b, d) for b in brd.bxy]

    # --- 3) Stackup ---
    die_t, er_list, d_r = read_stackup(stackup_path)
    brd.er_list = er_list
    brd.die_t = die_t
    brd.d_r = d_r

    # --- 4) R computation ---
    res_matrix = main_res(brd=brd)
    
    # --- 5) Z computation ---
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

    board_num = "4_1"
    board_name = f"board{board_num}"
    
    t0 = time.time()

    fstart = 10e3
    fstop = 200e6
    nf = 201
    freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    Freq = rf.Frequency(start=fstart/1e6, stop=fstop/1e6, npoints=nf, unit='mhz', sweep_type='log')
    
    powersi_snp = rf.Network(touchstone_path).interpolate(Freq)
    input_net = powersi_snp.z

    try:
        # First plot Python Z11
        fig, ax = plot_z_matrix(
            freq=freq,
            z_matrix=z,
            indices=(0, 0), # assuming IC port is the first port
            legend=["Python Z11"],
            scale=("log","log"),
            xlabel="Frequency (Hz)",
            ylabel="Impedance (Ω)",
            title="Comparison",
            color="blue",
            linestyle="-",
            linewidth=2,
        )

        # Overlay another curve
        _, ax = plot_z_matrix(
            freq=freq,
            z_matrix=input_net,
            indices=(0, 0), # assuming IC port is the first port
            legend=["PowerSI Z11"],
            color="red",
            linestyle="--",
            linewidth=2,
            ax=ax,
        )

        plt.show()

    except Exception as e:
        print(f"[ERROR] Touchstone failed: {e}")


