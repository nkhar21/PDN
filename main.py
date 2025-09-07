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
from input_AH import input_path, input_type, stack_up_csv_path, layer_type, touchstone_path

class Board:
    def __init__(self):
        self.bxy = np.array([])
        self.ic_via_xy = np.array([])
        self.ic_via_type = np.array([])
        self.decap_via_xy = np.array([])
        self.decap_via_type = np.array([])
        self.ic_via_loc = np.array([])        
        self.decap_via_loc = np.array([])

################################################# nk 
# def canon_node(name: str) -> str:
#     m = re.match(r'(Node)(\d+)$', name)
#     if m:
#         # Strip leading zeros: 'Node013' -> 'Node13'
#         return m.group(1) + str(int(m.group(2)))
#     return name

def canon_node(name: str) -> str:
    # Normalize Node labels like Node01 / Node001 / Node1 -> Node1
    m = re.match(r"Node0*([1-9]\d*)$", name, flags=re.IGNORECASE)
    return f"Node{m.group(1)}" if m else name

#################################################

#Reorders PWR and GND vias in alternating sequence and computes location flags based on via type and layer info.
def reorder_pwr_gnd_alternating_csv(xy_array, type_array,
                                     port_num=None, start_layers=None, stop_layers=None,
                                     loc_flag=0,
                                     global_max_stop_layer=None):
    xy_array = np.array(xy_array)
    type_array = np.array(type_array)

    pwr_indices = np.where(type_array == 1)[0]
    gnd_indices = np.where(type_array == 0)[0]
    min_len = min(len(pwr_indices), len(gnd_indices))
    ordered_indices = []
    for i in range(min_len):
        ordered_indices.append(pwr_indices[i])
        ordered_indices.append(gnd_indices[i])
    remaining_indices = np.setdiff1d(np.arange(len(type_array)), ordered_indices, assume_unique=True)
    ordered_indices.extend(remaining_indices.tolist())

    reordered_xy = xy_array[ordered_indices]
    reordered_type = type_array[ordered_indices]

    if loc_flag == "both":
        if port_num is None or start_layers is None or stop_layers is None:
            raise ValueError("When loc_flag='both', port_num, start_layers, and stop_layers are required.")
        if global_max_stop_layer is None:
            raise ValueError("When loc_flag='both', the global_max_stop_layer argument is also required.")

        reordered_port_num = np.array(port_num)[ordered_indices]
        reordered_start_layers = np.array(start_layers)[ordered_indices]
        reordered_stop_layers = np.array(stop_layers)[ordered_indices]

        reordered_loc = []
        for i in range(len(ordered_indices)):
            if reordered_port_num[i] == 1:  # IC
                reordered_loc.append(1 if reordered_start_layers[i] == 1 else 0)
            else:  # decap
                reordered_loc.append(0 if reordered_stop_layers[i] == global_max_stop_layer else 1)

        reordered_loc = np.array(reordered_loc)
    else:
        reordered_loc = np.full(len(reordered_type), loc_flag)

    return reordered_xy, reordered_type, reordered_loc

#Extracts start/stop layers and via types by matching IC and DECAP via node pairs from structured block data
def parse_input(brd, via_spd_path=None, via_csv_path=None):

    def extract_start_stop_layers_strict_order(via_lines, node_info, ic_blocks, decap_blocks):
       
        def find_via_by_node(node):
            # return the first VIA that touches this node (upper or lower)
            for upper, lower in via_lines:
                if upper == node or lower == node:
                    if (upper in node_info) and (lower in node_info):
                        return node_info[upper]['layer'], node_info[lower]['layer'], upper, lower
            return None


        def process_blocks(blocks, is_ic=True):
            results = []
            for block in blocks:
                lines = block.strip().splitlines()
                #plus_nodes = [re.search(r"\$Package\.(Node\d+)", l).group(1) for l in lines if l.strip().startswith("1")]
                #minus_nodes = [re.search(r"\$Package\.(Node\d+)", l).group(1) for l in lines if l.strip().startswith("2")]
                
                ######################################### nk
                plus_nodes = [
                    canon_node(re.search(r"\$Package\.(Node\d+)", l).group(1))
                    for l in lines if l.strip().startswith("1")
                ]
                minus_nodes = [
                    canon_node(re.search(r"\$Package\.(Node\d+)", l).group(1))
                    for l in lines if l.strip().startswith("2")
                ]
                #########################################

                for p_node, m_node in zip(plus_nodes, minus_nodes):
                    # + via
                    p_result = find_via_by_node(p_node)
                    if p_result:
                        start, stop, u, l = p_result
                        typ = 1 if (node_info[u]['type'] == 1 or node_info[l]['type'] == 1) else 0
                        results.append((start, stop, typ))
                    # - via
                    m_result = find_via_by_node(m_node)
                    if m_result:
                        start, stop, u, l = m_result
                        typ = 1 if (node_info[u]['type'] == 1 or node_info[l]['type'] == 1) else 0
                        results.append((start, stop, typ))
            return results

        ic_entries = process_blocks(ic_blocks, is_ic=True)
        decap_entries = process_blocks(decap_blocks, is_ic=False)

        all_entries = ic_entries + decap_entries
        start_layers = [e[0] for e in all_entries]
        stop_layers = [e[1] for e in all_entries]
        via_type = [e[2] for e in all_entries]

        return np.array(start_layers), np.array(stop_layers), np.array(via_type)

#Shifts decap via coordinates slightly to avoid overlap with IC vias
    def adjust_decap_xy_to_avoid_ic_overlap(ic_via_xy, decap_via_xy, shift=1e-6):
        ic_set = set(map(tuple, np.round(ic_via_xy, decimals=9)))
        adjusted = []
        for x, y in decap_via_xy:
            original = (round(x, 9), round(y, 9))
            while original in ic_set:
                x += shift
                y += shift
                original = (round(x, 9), round(y, 9))
            adjusted.append([x, y])
        return np.array(adjusted)


#--spd path
    if via_spd_path:
        with open(via_spd_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import numpy as np
        import re

        def is_duplicate_shape(new_shape, existing_shapes, tol=1e-9):
            for shape in existing_shapes:
                if shape.shape == new_shape.shape and np.allclose(shape, new_shape, atol=tol):
                    return True
            return False

        shape_section = re.findall(r"\.Shape\s+(ShapeSignal\d+)\n((?:(?:Box|Polygon).*?\n(?:\+.*?\n)?)?)", content)

        patch_mapping = re.findall(r"PatchSignal\d+\s+Shape\s*=\s*(ShapeSignal\d+)\s+Layer\s*=\s*(Signal\d+)", content)

        sorted_patch_mapping = sorted(patch_mapping, key=lambda x: int(x[1].replace("Signal", "")))

        shape_dict = {name: body for name, body in shape_section}

        polygon_shapes = []
        box_shapes = []

        for shape_name, _ in sorted_patch_mapping:
            shape = shape_dict.get(shape_name)
            if shape is None:
                continue

            if "Polygon" in shape:
                poly_coords = re.findall(r"(-?[\d\.eE\+\-]+)mm", shape)
                if len(poly_coords) >= 6 and len(poly_coords) % 2 == 0:
                    coords = [(float(poly_coords[i]), float(poly_coords[i + 1])) for i in range(0, len(poly_coords), 2)]
                    coords.append(coords[0])
                    shape_arr = np.array(coords) * 1e-3
                    polygon_shapes.append(shape_arr)
            elif "Box" in shape:
                box_match = re.search(
                    r"Box\d*[:\w+\-]*\s+(-?[\d\.eE\+\-]+)mm\s+(-?[\d\.eE\+\-]+)mm\s+([\d\.eE\+\-]+)mm\s+([\d\.eE\+\-]+)mm",
                    shape
                )
                if box_match:
                    x0, y0, w, h = map(float, box_match.groups())
                    x1, y1 = x0 + w, y0 + h
                    coords = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
                    shape_arr = np.array(coords) * 1e-3
                    box_shapes.append(shape_arr)

        # Final assignment of brd.bxy
        if polygon_shapes:
            first_shape = polygon_shapes[0]
            all_same = all(
                shape.shape == first_shape.shape and np.allclose(first_shape, shape, atol=1e-9)
                for shape in polygon_shapes
            )
            brd.bxy = [first_shape] if all_same else polygon_shapes

        elif box_shapes:
            first_shape = box_shapes[0]
            all_same = all(
                shape.shape == first_shape.shape and np.allclose(first_shape, shape, atol=1e-9)
                for shape in box_shapes
            )
            brd.bxy = [first_shape] if all_same else box_shapes
        else:
            brd.bxy = []


        node_lines = re.findall(
            r"(Node\d+)(?:::)?(PWR|GND)?\s+X\s*=\s*([-\d\.eE\+]+)mm\s+Y\s*=\s*([-\d\.eE\+]+)mm\s+Layer\s*=\s*Signal(\d+)",
            content, re.IGNORECASE
        )

        # node_info = {
        #     n[0]: {
        #         'type': 1 if n[1] and n[1].lower() == 'pwr' else 0,
        #         'x': float(n[2]),
        #         'y': float(n[3]),
        #         'layer': int(n[4])
        #     } for n in node_lines
            
        # }

        ######################### nk insert canonized keys too ###############################
        node_info = {}
        for n in node_lines:
            raw = n[0]                         # e.g. "Node013"
            name = canon_node(raw)             # "Node13"
            info = {
                'type': 1 if n[1] and n[1].lower() == 'pwr' else 0,
                'x': float(n[2]),
                'y': float(n[3]),
                'layer': int(n[4])
            }
            node_info[name] = info
            node_info[raw]  = info             # accept both forms
        ##########################################################

        # Canonicalize node keys (e.g., 'Node013' → 'Node13') and store both - nk
        for k, v in list(node_info.items()):
            node_info[canon_node(k)] = v

        component_section = re.search(r"\* Component description lines(.*?)\*", content, re.DOTALL)
        component_block = component_section.group(1) if component_section else ""

        ic_blocks = re.findall(r"\.Connect\s+ic_port.*?\n(.*?)\.EndC", component_block, re.DOTALL | re.IGNORECASE)
        decap_blocks = re.findall(r"\.Connect\s+(?:decap|cap)_port\d*\s+.*?\n(.*?)\.EndC", component_block, re.DOTALL | re.IGNORECASE)

        ic_node_ids = re.findall(r"\$Package\.Node(\d+)", '\n'.join(ic_blocks))
        decap_node_ids = re.findall(r"\$Package\.Node(\d+)", '\n'.join(decap_blocks))
    
        ic_node_ids = [f"Node{id}" for id in ic_node_ids]
        decap_node_ids = [f"Node{id}" for id in decap_node_ids]


            
        # New version: preserve order and include duplicates - NK
        # ic_xy_list, ic_type_list = [], []
        # for node in ic_node_ids:
        #     if node in node_info:
        #         info = node_info[node]
        #         ic_xy_list.append([info['x'] * 1e-3, info['y'] * 1e-3])
        #         ic_type_list.append(info['type'])
        # brd.ic_via_xy   = np.array(ic_xy_list)
        # brd.ic_via_type = np.array(ic_type_list)

        SNAP_DEC = 7
        # --- IC vias: preserve order & names from the IC block(s)
        ic_names_ordered, ic_xy_list, ic_type_list = [], [], []
        for blk in ic_blocks:
            lines = [ln.strip() for ln in blk.strip().splitlines()]
            plus_nodes  = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("1")]
            minus_nodes = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("2")]
            for n in plus_nodes + minus_nodes:
                if n in node_info:
                    info = node_info[n]
                    ic_names_ordered.append(n)
                    ic_xy_list.append([info['x']*1e-3, info['y']*1e-3])
                    ic_type_list.append(info['type'])

        brd.ic_node_names = ic_names_ordered
        brd.ic_via_xy     = np.round(np.array(ic_xy_list), SNAP_DEC)
        brd.ic_via_type   = np.array(ic_type_list, dtype=int)
        ######################################################## 

        via_lines = re.findall(r"Via\d+::\w+\s+UpperNode\s*=\s*(Node\d+)(?:::)?\w*\s+LowerNode\s*=\s*(Node\d+)(?:::)?\w*", content, flags=re.IGNORECASE)
        
        ################################################# nk
        via_lines = [(canon_node(u), canon_node(l)) for (u, l) in via_lines]
        missing = [n for pair in via_lines for n in pair if n not in node_info]
        if missing:
            print("[WARN] VIA references missing Node(s):", sorted(set(missing)))
        
        brd.start_layers, brd.stop_layers, brd.via_type = extract_start_stop_layers_strict_order(via_lines, node_info, ic_blocks, decap_blocks)
        brd.start_layers -= 1
        brd.stop_layers -= 1

        ############################## nk ##############################
        # decap_xy_list = []
        # decap_type_list = []

        # for node in decap_node_ids:
        #     if node in node_info:
        #         info = node_info[node]
        #         x, y = info['x'], info['y']
        #         t = info['type']  # 1 = pwr, 0 = gnd
        #         decap_xy_list.append([x * 1e-3, y * 1e-3])
        #         decap_type_list.append(t)

        # brd.decap_via_xy = np.array(decap_xy_list)
        # brd.decap_via_type = np.array(decap_type_list)

        # --- Decap vias: preserve Connect order (& names)
        decap_names_ordered, decap_xy_list, decap_type_list = [], [], []
        for blk in decap_blocks:
            lines = [ln.strip() for ln in blk.strip().splitlines()]
            plus_nodes  = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("1")]
            minus_nodes = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("2")]
            for n in plus_nodes + minus_nodes:
                if n in node_info:
                    info = node_info[n]
                    decap_names_ordered.append(n)
                    decap_xy_list.append([info['x']*1e-3, info['y']*1e-3])
                    decap_type_list.append(info['type'])

        brd.decap_node_names = decap_names_ordered
        brd.decap_via_xy     = np.round(np.array(decap_xy_list), SNAP_DEC)
        brd.decap_via_type   = np.array(decap_type_list, dtype=int)

        ############################## nk ##############################

        buried_dict = {}
        existing_xy_keys = set([tuple(xy) for xy in np.round(np.concatenate([brd.ic_via_xy, brd.decap_via_xy], axis=0) * 1e3, 6)])
        max_layer = max(n['layer'] for n in node_info.values())
        min_layer = min(n['layer'] for n in node_info.values())

        # Modify to include all vias as candidates for merging
        filtered_via_lines = []
        for upper, lower in via_lines:
            if upper not in node_info or lower not in node_info:
                continue
            filtered_via_lines.append((upper, lower))  


            upper_node = node_info[upper]
            lower_node = node_info[lower]

            # Condition 1: Neither is on the top or bottom layer
            if upper_node['layer'] in [min_layer, max_layer] or lower_node['layer'] in [min_layer, max_layer]:
                continue

            x = round((upper_node['x'] + lower_node['x']) / 2, 6)
            y = round((upper_node['y'] + lower_node['y']) / 2, 6)
            key = (x, y)


            ############################ nk ############################
            # if key in existing_xy_keys:
            #     # Check if it overlaps with existing nodes at the same location
            #     overlap_nodes = [n for n in node_info.values()
            #                      if round(n['x'], 6) == x and round(n['y'], 6) == y]
            #     if any(n['layer'] in [min_layer, max_layer] for n in overlap_nodes):
            #         continue  # Existing via connects to top/bottom layer → not eligible as a buried via

            if key not in buried_dict:
                buried_dict[key] = {
                    'type': 1 if upper_node['type'] == 1 or lower_node['type'] == 1 else 0,
                    'start': min(upper_node['layer'], lower_node['layer']) - 1,
                    'stop': max(upper_node['layer'], lower_node['layer']) - 1
                }

        if buried_dict:
            buried_via_xy = np.array([[x * 1e-3, y * 1e-3] for x, y in buried_dict])
            buried_via_type = np.array([v['type'] for v in buried_dict.values()])
            buried_start = np.array([v['start'] for v in buried_dict.values()])
            buried_stop = np.array([v['stop'] for v in buried_dict.values()])
            buried_sorted_idx = np.lexsort((-buried_via_xy[:, 1], buried_via_xy[:, 0]))
            brd.buried_via_xy = buried_via_xy[buried_sorted_idx]
            brd.buried_via_type = buried_via_type[buried_sorted_idx]
            buried_start = buried_start[buried_sorted_idx]
            buried_stop = buried_stop[buried_sorted_idx]
        else:
            brd.buried_via_xy = np.array([])
            brd.buried_via_type = np.array([])
            buried_start = np.array([])
            buried_stop = np.array([])

        if buried_start.size > 0 and buried_stop.size > 0:
                    brd.start_layers = np.concatenate([brd.start_layers, buried_start])
                    brd.stop_layers = np.concatenate([brd.stop_layers, buried_stop])
                    brd.via_type = np.concatenate([brd.via_type, brd.buried_via_type])

        from collections import defaultdict

        # Group vias based on coordinates
        coord_node_map = defaultdict(list)
        for idx, (upper, lower) in enumerate(via_lines):
            if upper not in node_info or lower not in node_info:
                continue
            up_info = node_info[upper]
            low_info = node_info[lower]
            x = round((up_info['x'] + low_info['x']) / 2, 6)
            y = round((up_info['y'] + low_info['y']) / 2, 6)
            key = (x, y)

            via_data = {
                'start': min(up_info['layer'], low_info['layer']) - 1,
                'stop': max(up_info['layer'], low_info['layer']) - 1,
                'nodes': {upper, lower},
                'type': 1 if up_info['type'] == 1 or low_info['type'] == 1 else 0,
                'port': (
                    'IC' if upper in ic_node_ids or lower in ic_node_ids else
                    'DECAP' if upper in decap_node_ids or lower in decap_node_ids else
                    'NONE'
                ),
                'index': idx  
            }
            coord_node_map[key].append(via_data)

        # Select merge targets and track indices (merge all via groups)
        merged_results = []
        merged_indices = set()

        for key, group in coord_node_map.items():
            visited = [False] * len(group)
            for i in range(len(group)):
                if visited[i]:
                    continue

                # Traverse connected groups using BFS or DFS
                queue = deque([i])
                connected = []
                while queue:
                    cur = queue.popleft()
                    if visited[cur]:
                        continue
                    visited[cur] = True
                    connected.append(group[cur])

                    for j in range(len(group)):
                        if visited[j]:
                            continue
                        vi, vj = group[cur], group[j]
                        # Vias are merged only if their ports are compatible and their nodes overlap
                        if vi['port'] != vj['port'] and 'NONE' not in (vi['port'], vj['port']):
                            continue
                        if len(vi['nodes'] & vj['nodes']) > 0:
                            queue.append(j)

                if len(connected) > 1:
                    start = min(v['start'] for v in connected)
                    stop = max(v['stop'] for v in connected)
                    typ = max(v['type'] for v in connected)
                    idx = min(v['index'] for v in connected)
                    merged_results.append({
                        'start': start,
                        'stop': stop,
                        'type': typ,
                        'index': idx
                    })
                    for v in connected:
                        merged_indices.add(v['index'])

        # Filter out only the original vias that were not merged
        original_data = []
        for idx, (upper, lower) in enumerate(via_lines):
            if idx in merged_indices:
                continue
            if upper not in node_info or lower not in node_info:
                continue
            up_info = node_info[upper]
            low_info = node_info[lower]
            original_data.append({
                'start': min(up_info['layer'], low_info['layer']) - 1,
                'stop': max(up_info['layer'], low_info['layer']) - 1,
                'type': 1 if up_info['type'] == 1 or low_info['type'] == 1 else 0,
                'index': idx
            })

        ##################################################################

        all_layers = [v['layer'] for v in node_info.values()]
        max_layer = max(all_layers)
        min_layer = min(all_layers)
        
        ################################# nk ##############################
        # # List of node IDs for IC and DECAP vias
        # ic_node_ids = [f"Node{id}" for id in re.findall(r"\$Package\.Node(\d+)", '\n'.join(ic_blocks))]
        # decap_node_ids = [f"Node{id}" for id in re.findall(r"\$Package\.Node(\d+)", '\n'.join(decap_blocks))]

        # brd.ic_via_loc = []
        # for xy in brd.ic_via_xy:
        #     x, y = round(xy[0] * 1e3, 6), round(xy[1] * 1e3, 6)
        #     found = False
        #     for node, info in node_info.items():
        #         if node in ic_node_ids and round(info['x'], 6) == x and round(info['y'], 6) == y:
        #             brd.ic_via_loc.append(1 if info['layer'] == min_layer else 0)
        #             found = True
        #             break
        #     if not found:
        #         brd.ic_via_loc.append(-1)  

        # brd.decap_via_loc = []
        # for xy in brd.decap_via_xy:
        #     x, y = round(xy[0] * 1e3, 6), round(xy[1] * 1e3, 6)
        #     found = False
        #     for node, info in node_info.items():
        #         if node in decap_node_ids and round(info['x'], 6) == x and round(info['y'], 6) == y:
        #             brd.decap_via_loc.append(1 if info['layer'] == min_layer else 0)
        #             found = True
        #             break
        #     if not found:
        #         brd.decap_via_loc.append(0)  

        # IC locations: top if on top layer, else bottom. Use node layer from names:
        top_layer = min(n['layer'] for n in node_info.values())
        bot_layer = max(n['layer'] for n in node_info.values())

        ic_locs = []
        for n in brd.ic_node_names:
            lyr = node_info[n]['layer']
            ic_locs.append(1 if lyr == top_layer else (0 if lyr == bot_layer else (1 if lyr == top_layer else 0)))  # ICs almost always on top; fallback conservative
        brd.ic_via_loc = np.array(ic_locs, dtype=int)

        # Decap locations: use the node layer directly
        dec_locs = []
        for n in brd.decap_node_names:
            lyr = node_info[n]['layer']
            dec_locs.append(1 if lyr == top_layer else 0 if lyr == bot_layer else (1 if lyr == top_layer else 0))
        brd.decap_via_loc = np.array(dec_locs, dtype=int)

        #################################### nk ##############################



        # --- Return ---
        return_values = (
            brd.bxy,
            brd.ic_via_xy,
            brd.ic_via_type,
            brd.start_layers,
            brd.stop_layers,
            brd.via_type,
            brd.decap_via_xy,
            brd.decap_via_type,
        )

        if brd.buried_via_xy.size > 0:
            return_values += (brd.buried_via_xy, brd.buried_via_type)

        return return_values

    #---csv file
    if via_csv_path:
        df = pd.read_csv(via_csv_path)
        df.columns = df.columns.str.strip()

        df["via_type"] = df["via_type"].map({"PWR": 1, "GND": 0})
        df["Port_number"] = df["Port_number"].fillna(-1).astype(int)
        import re
        brd.bxy = []

        def sort_by_index(col):
            match = re.search(r"b_[xy](\d+)", col)
            return int(match.group(1)) if match else 0

        x_cols = sorted([col for col in df.columns if col.startswith("b_x")], key=sort_by_index)
        y_cols = sorted([col for col in df.columns if col.startswith("b_y")], key=sort_by_index)

        # Check the number of shapes
        num_shapes = len(x_cols)

        for i in range(num_shapes):
            x_col = x_cols[i]
            y_col = y_cols[i]

            # Extract valid coordinates
            x_vals = df[x_col].dropna().astype(float).values
            y_vals = df[y_col].dropna().astype(float).values

            if len(x_vals) != len(y_vals):
                continue  

            coords = list(zip(x_vals, y_vals))
            if len(coords) >= 3:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])  # Close the polygon

                brd.bxy.append(np.array(coords) * 1e-3) 


        ic_df = df[df["Port_number"] == 1]
        decap_df = df[df["Port_number"] > 1]
        global_max_stop_layer = decap_df["stop_layer"].max()


        ic_xy = ic_df[["via_x_(mm)", "via_y_(mm)"]].drop_duplicates().values * 1e-3
        ic_type = ic_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["via_type"].values
        port_num = ic_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["Port_number"].values
        start_layers = ic_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["start_layer"].values
        stop_layers = ic_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["stop_layer"].values


        brd.ic_via_xy, brd.ic_via_type, brd.ic_via_loc = reorder_pwr_gnd_alternating_csv(ic_xy, ic_type, port_num=port_num, 
                                                                                        start_layers=start_layers, stop_layers=stop_layers, 
                                                                                        global_max_stop_layer=global_max_stop_layer, loc_flag="both")

        decap_xy_all, decap_type_all, decap_loc_all = [], [], []

        for port in sorted(decap_df["Port_number"].unique()):
            port_df = decap_df[decap_df["Port_number"] == port]
            xy = port_df[["via_x_(mm)", "via_y_(mm)"]].drop_duplicates().values * 1e-3
            typ = port_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["via_type"].values
            port_num = port_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["Port_number"].values
            start_layers = port_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["start_layer"].values
            stop_layers = port_df.drop_duplicates(subset=["via_x_(mm)", "via_y_(mm)"])["stop_layer"].values

            xy_r, typ_r, loc_r = reorder_pwr_gnd_alternating_csv(xy, typ, port_num=port_num, 
                                                                 start_layers=start_layers, stop_layers=stop_layers, 
                                                                 global_max_stop_layer=global_max_stop_layer, loc_flag="both")

            decap_xy_all.extend(xy_r)
            decap_type_all.extend(typ_r)
            decap_loc_all.extend(loc_r)

        brd.decap_via_xy = np.array(decap_xy_all)
        brd.decap_via_type = np.array(decap_type_all)
        brd.decap_via_loc = np.array(decap_loc_all)

        port_groups = {}
        for _, row in df.iterrows():
            if pd.isna(row["start_layer"]) or pd.isna(row["stop_layer"]):
                continue
            try:
                start = int(row["start_layer"])
                stop = int(row["stop_layer"])
            except ValueError:
                continue

            port = int(row["Port_number"]) if pd.notna(row["Port_number"]) else -1
            key = (round(float(row["via_x_(mm)"]), 6), round(float(row["via_y_(mm)"]), 6))
            vtype = 1 if row["via_type"] == 1 else 0
            if port not in port_groups:
                port_groups[port] = []
            port_groups[port].append((key, start, stop, vtype))

        full_sorted_list = []
        buried_via_list = []

        for port in sorted(p for p in port_groups if p >= 1):
            via_list = port_groups[port]
            pwr = [v for v in via_list if v[3] == 1]
            gnd = [v for v in via_list if v[3] == 0]
            for i in range(max(len(pwr), len(gnd))):
                if i < len(pwr): full_sorted_list.append(pwr[i])
                if i < len(gnd): full_sorted_list.append(gnd[i])

        all_layers = df[["start_layer", "stop_layer"]].stack().dropna().astype(int)
        if not all_layers.empty:
            min_layer = all_layers.min()
            max_layer = all_layers.max()
            for _, row in df.iterrows():
                if pd.isna(row["start_layer"]) or pd.isna(row["stop_layer"]):
                    continue
                try:
                    start = int(row["start_layer"])
                    stop = int(row["stop_layer"])
                except ValueError:
                    continue

                if start != min_layer and stop != max_layer:
                    key = (float(row["via_x_(mm)"]), float(row["via_y_(mm)"]))
                    vtype = 1 if row["via_type"] == 1 else 0
                    buried_via_list.append((key, start, stop, vtype))

        brd.start_layers = np.array([start - 1 for _, start, _, _ in full_sorted_list + buried_via_list])
        brd.stop_layers = np.array([stop - 1 for _, _, stop, _ in full_sorted_list + buried_via_list])
        brd.via_type = np.array([vtype for _, _, _, vtype in full_sorted_list + buried_via_list])
        brd.buried_via_xy = np.array([[x * 1e-3, y * 1e-3] for (x, y), _, _, _ in buried_via_list])
        brd.buried_via_type = np.array([vtype for _, _, _, vtype in buried_via_list])

        result = [
            brd.bxy,
            brd.ic_via_xy,
            brd.ic_via_type,
            brd.start_layers,
            brd.stop_layers,
            brd.via_type,
            brd.decap_via_xy,
            brd.decap_via_type,
        ]

        ################################# nk ###############################
        # Total via order in calc_z_fast: [IC] + [DECAP] + ([BURIED] if any)
        N_ic    = brd.ic_via_xy.shape[0]
        N_dec   = brd.decap_via_xy.shape[0]
        N_bury  = (0 if not hasattr(brd, 'buried_via_xy') or brd.buried_via_xy is None else brd.buried_via_xy.shape[0])
        N_total = N_ic + N_dec + N_bury

        top_port_num = [[-1] for _ in range(N_total)]
        bot_port_num = [[-1] for _ in range(N_total)]

        # Port 0 is IC. Every IC via belongs to port 0
        for i in range(N_ic):
            top_port_num[i] = [0]  # org_merge_pdn will split +/− by via_type

        # Decaps: 1..M in the order they appear in decap_blocks
        port_id = 1
        for blk in decap_blocks:
            lines = [ln.strip() for ln in blk.strip().splitlines()]
            plus_nodes  = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("1")]
            minus_nodes = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("2")]

            # one PWR node and one GND node per decap port
            for n in plus_nodes + minus_nodes:
                # find the decap-via index for this node name
                try:
                    j = brd.decap_node_names.index(n)               # 0..N_dec-1
                except ValueError:
                    continue  # node name not in decap set (shouldn't happen if SPD is consistent)

                global_idx = N_ic + j                                # shift into total via index
                if brd.decap_via_loc[j] == 1:
                    top_port_num[global_idx] = [port_id]
                else:
                    bot_port_num[global_idx] = [port_id]
            port_id += 1

        # Stash on the PDN object so calc_z_fast can use them
        brd.top_port_num = np.array(top_port_num, dtype=object)
        brd.bot_port_num = np.array(bot_port_num, dtype=object)
        ################################# nk end ###############################

        ################################# nk start ###############################
        # fix for b4_1: is gives PDN an exact per-via → (port, cavity) map for SPD inputs, the same way you already do for CSV. 
        # Without it, the heuristic can co-locate top/bottom vias in the same cavity and trigger log(0)

        # ---- Port/cavity maps for SPD (mirror CSV logic) ----
        N_ic   = brd.ic_via_xy.shape[0]
        N_dec  = brd.decap_via_xy.shape[0]
        N_bury = 0 if not hasattr(brd, 'buried_via_xy') or brd.buried_via_xy is None else brd.buried_via_xy.shape[0]
        N_total = N_ic + N_dec + N_bury

        top_port_num = [[-1] for _ in range(N_total)]
        bot_port_num = [[-1] for _ in range(N_total)]

        # IC = port 0
        for i in range(N_ic):
            # Assign IC vias to the cavity they physically sit on; PDN will split +/- by via_type
            if brd.ic_via_loc[i] == 1:
                top_port_num[i] = [0]
            else:
                bot_port_num[i] = [0]

        # Decaps: 1..M in .Connect order
        port_id = 1
        for blk in ic_blocks + decap_blocks:   # keep decap port ordering consistent with Connect blocks
            if blk in ic_blocks:
                continue
            lines = [ln.strip() for ln in blk.strip().splitlines()]
            plus  = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("1")]
            minus = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1)) for ln in lines if ln.startswith("2")]
            for n in plus + minus:
                if n in brd.decap_node_names:
                    j = brd.decap_node_names.index(n)      # 0..N_dec-1
                    global_idx = N_ic + j
                    if brd.decap_via_loc[j] == 1:
                        top_port_num[global_idx] = [port_id]
                    else:
                        bot_port_num[global_idx] = [port_id]
            port_id += 1

        brd.top_port_num = np.array(top_port_num, dtype=object)
        brd.bot_port_num = np.array(bot_port_num, dtype=object)

        ################################# nk end ###############################


        if hasattr(brd, "buried_via_xy") and hasattr(brd, "buried_via_type") and brd.buried_via_xy.size > 0:
            result.append(brd.buried_via_xy)
            result.append(brd.buried_via_type)

        return tuple(result)

    else:
        raise ValueError("Either 'via_spd_path' or 'via_csv_path' must be specified..")



# return thicknesses, ers
def read_stackup_from_csv(stack_up_csv_path):
    df = pd.read_csv(stack_up_csv_path)
    df.columns = df.columns.str.strip()

    df_medium = df[df['Layer Name'].str.startswith('Medium')]
    df_Signal = df[df['Layer Name'].str.startswith('Signal')]

    # mm -> m
    thicknesses = df_medium['Thickness(mm)'].astype(float).values * 1e-3
    d_r = df_Signal['Thickness(mm)'].astype(float).values * 1e-3
    ers = df_medium['Er'].astype(float).values
    
    return thicknesses, ers, d_r

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

def gen_brd_data(brd, input_path, input_type, stack_up_csv_path=None, d=1e-3):
    via_spd_path = input_path if input_type == "spd" else None
    via_csv_path = input_path if input_type == "csv" else None

    result = parse_input(brd, via_spd_path=via_spd_path, via_csv_path=via_csv_path)

    # Safely unpack depending on the presence of buried vias
    if len(result) == 8:
        bxy, ic_via_xy, ic_via_type, start_layers, stop_layers, via_type, decap_via_xy, decap_via_type = result
        brd.buried_via_xy, brd.buried_via_type = None, None
    elif len(result) == 10:
        bxy, ic_via_xy, ic_via_type, start_layers, stop_layers, via_type, decap_via_xy, decap_via_type, buried_via_xy, buried_via_type = result
        brd.buried_via_xy = buried_via_xy
        brd.buried_via_type = buried_via_type
    else:
        raise ValueError(f"Unexpected return count from parse_input: {len(result)}")
    
    bxy = np.array([np.round(np.array(item), 6) for item in bxy], dtype=object)
    brd.bxy = bxy

    brd.sxy = np.concatenate( [brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy], axis=0)
    brd.sxy = [brd.seg_bd_node(single_bxy, d) for single_bxy in brd.bxy]
    brd.sxy = np.concatenate([brd.seg_bd_node(b, d) for b in brd.bxy], axis=0)

    brd.sxy_index_ranges = []
    offset = 0
    for b in brd.bxy:
        n_seg = brd.seg_bd_node(b, d).shape[0]
        brd.sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg

    brd.sxy_list = [brd.seg_bd_node(b, d) for b in brd.bxy]

    ############################################# nk rounding
    SNAP_DECIMALS = 7
    
    brd.ic_via_xy = np.round(np.array(ic_via_xy), SNAP_DECIMALS)
    brd.ic_via_type = np.array(ic_via_type, dtype=int)

    start_layers = np.array(start_layers).tolist()
    stop_layers = np.array(stop_layers).tolist()
    brd.start_layers = start_layers
    brd.stop_layers = stop_layers

    n_ic = brd.ic_via_xy.shape[0]
    set_ic_via_loc_per_via(brd, start_layers[:n_ic], stop_layers[:n_ic])

    brd.via_type = np.array(via_type, dtype=int)
    brd.decap_via_xy = np.round(np.array(decap_via_xy), SNAP_DECIMALS)
    brd.decap_via_type = np.array(decap_via_type, dtype=int)

    ########################################### nk debug vias
    # total_vias = 0
    # if brd.ic_via_xy.size:     total_vias += brd.ic_via_xy.shape[0]
    # if brd.decap_via_xy.size:  total_vias += brd.decap_via_xy.shape[0]
    # if getattr(brd, 'buried_via_xy', np.array([])).size:
    #     total_vias += brd.buried_via_xy.shape[0]

    # print("[INFO] counts:",
    #     "start", len(brd.start_layers),
    #     "stop", len(brd.stop_layers),
    #     "type", len(brd.via_type),
    #     "vias_xy", total_vias)

    # if not (len(brd.start_layers) == len(brd.stop_layers) == len(brd.via_type) == total_vias):
    #     raise RuntimeError("Via array length mismatch: start/stop/type must equal total_vias.")
    ########################################### nk debug vias end


    for i in range(len(brd.decap_via_xy)):
        while any(np.allclose(brd.decap_via_xy[i], ic_xy, atol=1e-9) for ic_xy in brd.ic_via_xy):
            brd.decap_via_xy[i][0] += 1e-7
            brd.decap_via_xy[i][1] += 1e-7

    ############################# nk ###############################
    # b4_1 fix
    # After brd.decap_via_xy / brd.decap_via_loc are set
    seen = {0:set(), 1:set()}
    for i in range(len(brd.decap_via_xy)):
        l = int(brd.decap_via_loc[i])  # 1=top, 0=bottom
        x, y = brd.decap_via_xy[i]
        key = (round(x,9), round(y,9))
        # avoid IC overlap (you already do elsewhere), plus intra-cavity decap overlap:
        while key in seen[l] or any(np.allclose([x,y], ic_xy, atol=1e-9) for ic_xy in brd.ic_via_xy):
            x += 1e-7; y += 1e-7
            key = (round(x,9), round(y,9))
        seen[l].add(key)
        brd.decap_via_xy[i] = [x, y]

    # FINAL guard: make all via XY globally unique (IC + DECAP + BURIED)
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

        # Order matters: PDN concatenates IC -> DECAP -> BURIED
        if hasattr(brd, 'ic_via_xy') and brd.ic_via_xy.size:
            proc(brd.ic_via_xy)
        if hasattr(brd, 'decap_via_xy') and brd.decap_via_xy.size:
            proc(brd.decap_via_xy)
        if hasattr(brd, 'buried_via_xy') and brd.buried_via_xy is not None and brd.buried_via_xy.size:
            proc(brd.buried_via_xy)

    # call it here, after your current per-cavity fix
    _dedupe_global_vias(brd)    



    def _assert_no_exact_duplicates(brd, rdec=9):
        pts = []
        lbls = []
        if brd.ic_via_xy.size:
            pts += [tuple(np.round(p, rdec)) for p in brd.ic_via_xy]
            lbls += [('IC', i) for i in range(len(brd.ic_via_xy))]
        if brd.decap_via_xy.size:
            pts += [tuple(np.round(p, rdec)) for p in brd.decap_via_xy]
            lbls += [('DECAP', i) for i in range(len(brd.decap_via_xy))]
        if getattr(brd, 'buried_via_xy', np.array([])).size:
            pts += [tuple(np.round(p, rdec)) for p in brd.buried_via_xy]
            lbls += [('BURIED', i) for i in range(len(brd.buried_via_xy))]

        from collections import Counter
        dup_keys = [k for k, c in Counter(pts).items() if c > 1]
        if dup_keys:
            print('[WARN] Global duplicate via XY exist (will cause log(0) in BEM):', dup_keys)

    #_assert_no_exact_duplicates(brd) 
    ############################# nk ###############################

    if stack_up_csv_path:
        die_t, er_list, d_r = read_stackup_from_csv(stack_up_csv_path)
        stackup = np.zeros(len(er_list) + 1)


        stackup = np.zeros(len(layer_type), dtype=int)
        for idx, row in layer_type.iterrows():
            if row['type'] == 1:
                signal_number = int(row['layer'].replace('Signal', ''))
                stackup[signal_number - 1] = 1


        brd.er_list = er_list

    brd.stackup = stackup
    brd.die_t = die_t
    brd.d_r=d_r
    
    # resistance part
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
        ic_via_xy = brd.ic_via_xy,
        ic_via_loc = brd.ic_via_loc,
        ic_via_type = brd.ic_via_type
    )

    #R/L
    z = brd.calc_z_fast(res_matrix=res_matrix)
   
    brd.buried_via_xy = brd.buried_via_xy if hasattr(brd, "buried_via_xy") else None
    brd.buried_via_type = brd.buried_via_type if hasattr(brd, "buried_via_type") else None

    result = [
        z,
        brd.bxy,
        brd.ic_via_xy,
        brd.ic_via_type,
        brd.decap_via_xy,
        brd.decap_via_type,
        brd.decap_via_loc,
        stackup,
        die_t,
        brd.sxy_list
    ]

    if brd.buried_via_xy is not None and brd.buried_via_type is not None:
        result.append(brd.buried_via_xy)
        result.append(brd.buried_via_type)

    return tuple(result)

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
            input_path=input_path,
            input_type=input_type,
            stack_up_csv_path=stack_up_csv_path
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


