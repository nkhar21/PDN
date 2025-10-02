# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import pathlib

# from copy import deepcopy
# from code_pdn_AH import PDN
# import time
# import os
# import skrf as rf
# from input_AH import input_path, stackup_path, touchstone_path

# from pdn_io.spd_parser import parse_spd
# from pdn_io.stackup_parser import read_stackup
# from pdn_io.spd_parser_multi import parse_spd_multi


# if __name__ == '__main__':
#     # Get the directory where the script is located
#     script_dir = os.path.dirname(__file__)

#     # Construct the absolute path to the SPD file
#     spd_path = os.path.join(script_dir, "input files", "board 5", "b5.spd")

#     boards = parse_spd_multi(PDN, spd_path, ["pwr", "pwr1", "pwr2", "pwr3", "pwr4", "pwr5"], verbose=True)  
#     if boards:
#         for i, board_data in enumerate(boards):
#             print(f"\n--- Printing Board {i+1} ({['pwr', 'pwr1', 'pwr2', 'pwr3', 'pwr4', 'pwr5'][i]}) ---")
            
#             # The 'board_data' is a tuple. You can access its elements by index.
#             # Printing each element separately makes the output more readable.
#             print(f"Board Polygons (bxy): {board_data[0]}")
#             print(f"IC Via XY: {board_data[1]}")
#             print(f"IC Via Type: {board_data[2]}")
#             print(f"Start Layers: {board_data[3]}")
#             print(f"Stop Layers: {board_data[4]}")
#             print(f"Via Type: {board_data[5]}")
#             print(f"Decap Via XY: {board_data[6]}")
#             print(f"Decap Via Type: {board_data[7]}")
#             print(f"Stackup: {board_data[8]}")
            
#             # Check for buried vias, as they are optional in the tuple
#             if len(board_data) > 9:
#                 print(f"Buried Via XY: {board_data[9]}")
#                 print(f"Buried Via Type: {board_data[10]}")

#             print("------------------------------------------")
#     else:
#         print("\nNo boards were successfully parsed.")



import os
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

from BEM_AC_NVM_PDN import PDN
from pdn_io.spd_parser_multi import parse_spd_multi
from pdn_io.stackup_parser import read_stackup


if __name__ == "__main__":

    # --- Paths ---
    script_dir = os.path.dirname(__file__)
    spd_path = os.path.join(script_dir, "input files", "board 5", "b5.spd")
    stackup_path = os.path.join(script_dir, "input files", "board 5", "b5_stackup.csv")
    touchstone_path = os.path.join(script_dir, "input files", "board 5", "b5.S6P")

    # --- Build all 6 boards ---

    boards = parse_spd_multi(PDN, spd_path,
                             ["pwr", "pwr1", "pwr2", "pwr3", "pwr4", "pwr5"],
                             verbose=True)

    # --- Pick board 1 (net 'pwr') ---

    brd = boards[0]  # index 0 → 'pwr'
    # Attach stackup info
    die_t, er_list, d_r = read_stackup(stackup_path)
    brd.er_list = er_list
    brd.die_t = die_t
    brd.d_r = d_r

    # --- Build boundary segments for calc_z_fast ---
    d = 1e-3  # your spacing step
    brd.bxy = np.array([np.round(np.array(item), 6) for item in brd.bxy], dtype=object)
    brd.sxy = np.concatenate([brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy], axis=0)

    brd.sxy_index_ranges = []
    offset = 0
    for b in brd.bxy:
        n_seg = brd.seg_bd_node(b, d).shape[0]
        brd.sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg

    brd.sxy_list = [brd.seg_bd_node(b, d) for b in brd.bxy]

    # Now you can print using the named attributes as requested
    print(f"Board Polygons (bxy): {brd.bxy}")
    print(f"IC Via XY: {brd.ic_via_xy}")
    print(f"IC Via Type: {brd.ic_via_type}")
    print(f"Start Layers: {brd.start_layers}")
    print(f"Stop Layers: {brd.stop_layers}")
    print(f"Via Type: {brd.via_type}")
    print(f"Decap Via XY: {brd.decap_via_xy}")
    print(f"Decap Via Type: {brd.decap_via_type}")
    print(f"Stackup: {brd.stackup}")
    
    # Check for buried vias, as they are optional
    if brd.buried_via_xy is not None:
        print(f"Buried Via XY: {brd.buried_via_xy}")
        print(f"Buried Via Type: {brd.buried_via_type}")

    print("------------------------------------------")


    # --- Compute Z from Python solver ---
    z_python = brd.calc_z_fast(res_matrix=None, verbose=False)  # shape (nfreq, nports, nports)

    # Frequency setup (should match your sweep)
    fstart = 1e5   # 100 kHz
    fstop = 200e6  # 200 MHz
    nf = 201
    freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)

    # --- Load Touchstone ---
    ntw = rf.Network(touchstone_path)
    # Interpolate to our frequency grid
    ntw = ntw.interpolate(freq, kind="linear")
    z_touch = ntw.z

    # --- Compare |Z11| ---
    plt.figure()
    plt.loglog(freq, np.abs(z_python[:, 0, 0]), label="Python calc_z_fast Z11")
    plt.loglog(freq, np.abs(z_touch[:, 0, 0]), "--", label="Touchstone b5.S6P Z11")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|Z11| (Ohm)")
    plt.title("Comparison: Board 1 (pwr) Z11")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
