import re
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from copy import deepcopy
from code_pdn_AH import PDN
from RES1_AH import main_res
import time
import os
import skrf as rf
import pandas as pd
from collections import deque
from input_AH import input_path, stack_up_csv_path, layer_type_path, touchstone_path

from pdn_io.spd_parser import parse_spd
from pdn_io.stackup_parser import (
    read_stackup_from_csv,
    read_layer_type_csv,
    build_stackup_mask,
)

class Board:
    def __init__(self):
        self.bxy = np.array([])
        self.ic_via_xy = np.array([])
        self.ic_via_type = np.array([])
        self.decap_via_xy = np.array([])
        self.decap_via_type = np.array([])
        self.ic_via_loc = np.array([])        
        self.decap_via_loc = np.array([])

def set_ic_via_loc_per_via(brd, start_layers, stop_layers):
    # Set via start and stop layers individually for each IC via.

    # Ensure inputs are 1D numpy arrays of correct shape
    start_layers = np.asarray(start_layers).flatten()
    stop_layers = np.asarray(stop_layers).flatten()
   
    n_vias = brd.ic_via_xy.shape[0]

    if start_layers.shape[0] != n_vias or stop_layers.shape[0] != n_vias:
        raise ValueError(f"start_layers and stop_layers must each have {n_vias} elements, "
                         f"but got {start_layers.shape[0]} and {stop_layers.shape[0]}.")

    # Adjust for 0-based indexing
    start_layers -= 1
    stop_layers  -= 1

   
def handle_gen_brd_data_output(result):
    if len(result) == 10:
        return result + (None, None)
    elif len(result) == 12:
        return result
    else:
        raise ValueError(f"Unexpected number of return values from gen_brd_data: {len(result)}")

def gen_brd_data(
    brd,
    spd_path: str,
    stack_up_csv_path: str,
    layer_type_csv_path: str,
    d: float = 1e-3,
):
    """
    Build board state and compute Z using external parsers for SPD and stackup/layer-type.
    """
    # --- 1) Parse SPD (board shapes, vias, layers, etc.) ---
    result = parse_spd(brd, spd_path)

    # Safely unpack depending on the presence of buried vias
    if len(result) == 8:
        (bxy, ic_via_xy, ic_via_type,
         start_layers, stop_layers, via_type,
         decap_via_xy, decap_via_type) = result
        brd.buried_via_xy, brd.buried_via_type = None, None
    elif len(result) == 10:
        (bxy, ic_via_xy, ic_via_type,
         start_layers, stop_layers, via_type,
         decap_via_xy, decap_via_type,
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

    # --- 3) Assign via arrays to brd (round, types, layers) ---
    SNAP_DECIMALS = 7
    brd.ic_via_xy   = np.round(np.array(ic_via_xy), SNAP_DECIMALS)
    brd.ic_via_type = np.array(ic_via_type, dtype=int)

    start_layers = np.array(start_layers).tolist()
    stop_layers  = np.array(stop_layers).tolist()
    brd.start_layers = start_layers
    brd.stop_layers  = stop_layers

    # Optional: keep per-IC via cavity assignment using start/stop if you rely on it downstream
    n_ic = brd.ic_via_xy.shape[0]
    set_ic_via_loc_per_via(brd, start_layers[:n_ic], stop_layers[:n_ic])

    brd.via_type       = np.array(via_type, dtype=int)
    brd.decap_via_xy   = np.round(np.array(decap_via_xy), SNAP_DECIMALS)
    brd.decap_via_type = np.array(decap_via_type, dtype=int)

    # --- 4) Guards against overlapping decap/IC vias (your existing fixes) ---
    for i in range(len(brd.decap_via_xy)):
        while any(np.allclose(brd.decap_via_xy[i], ic_xy, atol=1e-9) for ic_xy in brd.ic_via_xy):
            brd.decap_via_xy[i][0] += 1e-7
            brd.decap_via_xy[i][1] += 1e-7

    # b4_1 fix: ensure decaps are unique per cavity & globally
    seen = {0: set(), 1: set()}
    for i in range(len(brd.decap_via_xy)):
        l = int(brd.decap_via_loc[i])  # 1=top, 0=bottom
        x, y = brd.decap_via_xy[i]
        key = (round(x, 9), round(y, 9))
        while key in seen[l] or any(np.allclose([x, y], ic_xy, atol=1e-9) for ic_xy in brd.ic_via_xy):
            x += 1e-7; y += 1e-7
            key = (round(x, 9), round(y, 9))
        seen[l].add(key)
        brd.decap_via_xy[i] = [x, y]

    def _dedupe_global_vias(brd, eps=1e-7, rdec=9):
        seen = set()
        def proc(arr):
            if arr is None or np.size(arr) == 0:
                return
            for i in range(len(arr)):
                x, y = float(arr[i][0]), float(arr[i][1])
                key = (round(x, rdec), round(y, rdec))
                while key in seen:
                    x += eps; y += eps
                    key = (round(x, rdec), round(y, rdec))
                seen.add(key)
                arr[i] = [x, y]
        if hasattr(brd, 'ic_via_xy') and brd.ic_via_xy.size:
            proc(brd.ic_via_xy)
        if hasattr(brd, 'decap_via_xy') and brd.decap_via_xy.size:
            proc(brd.decap_via_xy)
        if hasattr(brd, 'buried_via_xy') and brd.buried_via_xy is not None and brd.buried_via_xy.size:
            proc(brd.buried_via_xy)
    _dedupe_global_vias(brd)

    # --- 5) Stackup & layer-type via external parsers ---
    die_t, er_list, d_r = read_stackup_from_csv(stack_up_csv_path)
    layer_type_df = read_layer_type_csv(layer_type_csv_path)
    stackup = build_stackup_mask(layer_type_df)

    brd.er_list = er_list
    brd.stackup = stackup
    brd.die_t = die_t
    brd.d_r = d_r

    # --- 6) Resistance & Z computation (unchanged) ---
    res_matrix = main_res(
        brd=brd,
        die_t=die_t,
        d=d_r,
        stackup=brd.stackup,
        start_layer=brd.start_layers,
        stop_layer=brd.stop_layers,
        decap_via_type=brd.decap_via_type,
        decap_via_xy=brd.decap_via_xy,
        decap_via_loc=brd.decap_via_loc,
        ic_via_xy=brd.ic_via_xy,
        ic_via_loc=brd.ic_via_loc,
        ic_via_type=brd.ic_via_type,
    )
    z = brd.calc_z_fast(res_matrix=res_matrix)

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
        stackup,
        die_t,
        brd.sxy_list,
    ]
    if brd.buried_via_xy is not None and brd.buried_via_type is not None:
        result_out.append(brd.buried_via_xy)
        result_out.append(brd.buried_via_type)

    return tuple(result_out)






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


def connect_z_short(z1, shorted_port):
    a_ports = list(range(z1.shape[1]))

    del a_ports[shorted_port]
    p_ports = [shorted_port]
    Zaa = z1[np.ix_(a_ports, a_ports)]
    Zpp = z1[np.ix_(p_ports, p_ports)]
    Zap = z1[np.ix_(a_ports, p_ports)]
    Zpa = z1[np.ix_(p_ports, a_ports)]
    z_connect = Zaa - np.matmul(np.matmul(Zap,np.linalg.inv(Zpp)),Zpa)
    return z_connect

def short_1port_z( z, map2orig_input, shorted_port):
        output_net = deepcopy(z)
        output_net = np.linalg.inv(np.delete(np.delete(np.linalg.inv(output_net), shorted_port, axis=1),                      
                                             shorted_port, axis=2))
        map2orig_output = deepcopy(map2orig_input)
        del map2orig_output[shorted_port]


        return output_net, map2orig_output


def short_1port_toolz(z_full, shorted_port):
    z_inv = np.linalg.inv(z_full)                                                              
    z_shorted = np.delete(np.delete(z_inv, shorted_port, axis=0), shorted_port, axis=1)
    return np.linalg.inv(z_shorted)

network = rf.Network(touchstone_path)

z_tool_full = network.z
freqs = network.f  


shorted_port_index = 0        
z_tool_shorted = np.array([short_1port_toolz(z, shorted_port_index) for z in z_tool_full])  

z22_tool = np.abs(z_tool_shorted[:, 0, 0])  # Z22 after IC port is shorted

def compare_shorted_z(touchstone_path, python_z, freq, ic_port_idx):
    try:
        # Load Touchstone and interpolate
        ntw = rf.Network(touchstone_path)
        if not np.allclose(ntw.f, freq):

            ntw = ntw.interpolate(freq, kind='linear')
        z_touchstone = ntw.z
        # Short IC port in both matrices
        python_z_shorted = short_1port_z(python_z, ic_port_idx)
        touchstone_z_shorted = short_1port_toolz(z_touchstone, ic_port_idx)

        n_ports = python_z_shorted.shape[1]
        for i in range(n_ports):
            for j in range(n_ports):
                plt.figure()
                plt.plot(freq / 1e6, 20 * np.log10(np.abs(python_z_shorted[:, i, j]) + 1e-20), label='Python Z (shorted)')
                plt.plot(freq / 1e6, 20 * np.log10(np.abs(touchstone_z_shorted[:, i, j]) + 1e-20), '--', label='Touchstone Z (shorted)')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel(f'|Z{i+1}{j+1}| (dBΩ)')
                plt.title(f'Comparison of Shorted Z{i+1}{j+1}')
                plt.legend()
                plt.grid(True)
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

    for i in range(N):

        result = gen_brd_data(
            brd=brd,
            spd_path=input_path,                  # e.g., "b4_1.spd"
            stack_up_csv_path=stack_up_csv_path,  # e.g., "b4_1_stackup.csv"
            layer_type_csv_path=layer_type_path,  # e.g., "b4_1_layer_type.csv"
        )


        z, bxy, ic_via_xy, ic_via_type, decap_via_xy, decap_via_type, \
        decap_via_loc, stackup, die_t, sxy_list, buried_via_xy, buried_via_type = handle_gen_brd_data_output(result)

        if np.isnan(np.sum(z)):
            print("[Error] Z contains NaN values — skipping")
            num_error += 1
            continue

        ic_port_index = detect_ic_port_index(touchstone_path)

        board_name = f"board{i}"
        save2s(brd, z, board_name, BASE_PATH)
        t0 = time.time()

        fstart = 10e3
        fstop = 200e6
        nf = 201
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
        Freq = rf.Frequency(start=fstart/1e6, stop=fstop/1e6, npoints=nf, unit='mhz', sweep_type='log')
        
        try:
            net = rf.Network(touchstone_path).interpolate(Freq)
            input_net = net.z

            plt.figure()
            plt.loglog(freq, np.abs(z[:, ic_port_index, ic_port_index]), label='Node Voltage Method Z')
            plt.loglog(freq, np.abs(input_net[:, ic_port_index, ic_port_index]), '--', label='Power SI Z')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('|Z11| (Ohm)')
            plt.legend()
            plt.grid(True)
            plt.title('Z_IC')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[ERROR] Touchstone failed: {e}")


