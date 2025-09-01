import numpy as np
from math import sqrt, pi, sin, cos, log, atan
from copy import deepcopy
import time


def seg_bd_node(bxy_b, dl):
    # For outer boundary, it must rotate counter-clockwise !!!
    # note that input matrix bxy must go back to the origin point!!!
    # bxy is the boundary coordinate information,N*2 matrix
    # dl is the length of the segments
    # sxy is a S*4 matrix, x1, y1, x2, y2

    # if dl is smaller than the interval of the bxy points, do linear
    # interpolation

    '''Another way is to use numpy.append here'''

    bxy_old = deepcopy(bxy_b)

    if bxy_old.ndim == 3 and bxy_old.shape[0] == 1:
        bxy_old = bxy_old[0]

    if bxy_old[-1, 0] != bxy_old[0, 0] or bxy_old[-1, 1] != bxy_old[0, 1]:
        bxy_b = np.zeros((bxy_old.shape[0] + 1, bxy_old.shape[1]))
        bxy_b[0:-1, :] = bxy_old
        bxy_b[-1, :] = bxy_old[0, :]
    # calculate number of segments needed first
    nseg = 0
    for i in range(0, bxy_b.shape[0] - 1):
        len_ith = sqrt((bxy_b[i + 1, 0] - bxy_b[i, 0]) ** 2 + (bxy_b[i + 1, 1] - bxy_b[i, 1]) ** 2)
        if dl <= len_ith:
            ne = np.floor(len_ith / dl)
            if (len_ith - ne * dl) > dl * 0.01:
                nseg += ne + 1
            else:
                nseg += ne
        else:
            nseg += 1
    if type(nseg) == int:
        nseg = nseg
    else:
        nseg = nseg.astype(int)
    sxy = np.ndarray((nseg, 4))
    s = 0
    for i in range(0, bxy_b.shape[0] - 1):
        len_ith = sqrt((bxy_b[i + 1, 0] - bxy_b[i, 0]) ** 2 + (bxy_b[i + 1, 1] - bxy_b[i, 1]) ** 2)
        if dl <= len_ith:
            ne = np.floor(len_ith / dl).astype(int)
            for j in range(0, ne):
                sxy[s, 0] = bxy_b[i, 0] + j * dl / len_ith * (bxy_b[i + 1, 0] - bxy_b[i, 0])
                sxy[s, 1] = bxy_b[i, 1] + j * dl / len_ith * (bxy_b[i + 1, 1] - bxy_b[i, 1])
                sxy[s, 2] = bxy_b[i, 0] + (j + 1) * dl / len_ith * (bxy_b[i + 1, 0] - bxy_b[i, 0])
                sxy[s, 3] = bxy_b[i, 1] + (j + 1) * dl / len_ith * (bxy_b[i + 1, 1] - bxy_b[i, 1])
                s += 1
            if (len_ith - ne * dl) > dl * 0.01:
                sxy[s, 0] = bxy_b[i, 0] + (j + 1) * dl / len_ith * (bxy_b[i + 1, 0] - bxy_b[i, 0])
                sxy[s, 1] = bxy_b[i, 1] + (j + 1) * dl / len_ith * (bxy_b[i + 1, 1] - bxy_b[i, 1])
                sxy[s, 2] = bxy_b[i + 1, 0]
                sxy[s, 3] = bxy_b[i + 1, 1]
                s += 1
        else:
            sxy[s, 0] = bxy_b[i, 0]
            sxy[s, 1] = bxy_b[i, 1]
            sxy[s, 2] = bxy_b[i + 1, 0]
            sxy[s, 3] = bxy_b[i + 1, 1]
            s += 1
    return sxy

def segment_port(x0,y0,r,n):
# Define a function for port segmentation
# for port boundary, it must rotate clockwise
# x0, y0 is the port center location, r is radius
# n is the number of the segments
# sxy is a S*4 matrix, x1, y1, x2, y2
    dtheta = 2*pi/n
    n = int(n)
    sxy = np.ndarray((n,4))
    for i in range(0,n):
        sxy[i,0] = x0 + r*cos(-(i)*dtheta)
        sxy[i,1] = y0 + r*sin(-(i)*dtheta)
        sxy[i,2] = x0 + r*cos(-(i+1)*dtheta)
        sxy[i,3] = y0 + r*sin(-(i+1)*dtheta)
    return sxy


def planesresistance(bxy_b, via_xy_b, via_r, d):
    sigma = 5.8e7
    dl = 1e-3
    num_via_seg = 6
    via_sxy = np.zeros((via_xy_b.shape[0] * num_via_seg, 4))

    for i in range(0, via_xy_b.shape[0]):
        via_sxy[i * num_via_seg:(i + 1) * num_via_seg, :] = segment_port(via_xy_b[i, 0], via_xy_b[i, 1], via_r, num_via_seg)


    s_b = seg_bd_node(bxy_b, dl)
    b = np.concatenate((via_sxy, s_b))
    Ntot = b.shape[0]

    U = np.ndarray((Ntot, Ntot))
    H = np.ndarray((Ntot, Ntot))

    for m in range(0, b.shape[0]):
        for n in range(0, b.shape[0]):
            if m == n:
                U[m, n] = 1
                wj = sqrt((b[n, 0] - b[n, 2]) ** 2 + (b[n, 1] - b[n, 3]) ** 2)
                H[m, n] = -1 / (sigma * pi * d) * (log(wj / 2) - 1)
            else:
                xi1 = b[m, 0]
                yi1 = b[m, 1]
                xi2 = b[m, 2]
                yi2 = b[m, 3]
                xi0 = (xi1 + xi2) / 2
                yi0 = (yi1 + yi2) / 2

                xj1 = b[n, 0]
                yj1 = b[n, 1]
                xj2 = b[n, 2]
                yj2 = b[n, 3]
                xj0 = (xj1 + xj2) / 2
                yj0 = (yj1 + yj2) / 2

                wj = sqrt((xj1 - xj2) ** 2 + (yj1 - yj2) ** 2)

                R = sqrt((xi0 - xj0) ** 2 + (yi0 - yj0) ** 2)

                cos_theta = ((xj0 - xi0) * (yj2 - yj1) - (yj0 - yi0) * (xj2 - xj1)) / (R * wj)

                U[m, n] = -1 / pi * wj * cos_theta / R
                H[m, n] = -1 / (sigma * pi * d) * log(R)

    U_H = np.matmul(np.linalg.inv(U), H)
    U_H_inv = np.linalg.inv(U_H)

    # create a list to merge the segments on the same via
    reduce_list = list(range(0, (via_xy_b.shape[0] - 1) * num_via_seg + 1, num_via_seg)) + list(
        range(via_xy_b.shape[0] * num_via_seg, b.shape[0]))

    U_H_inv_R = np.add.reduceat(np.add.reduceat(U_H_inv, reduce_list, axis=0), reduce_list,
                                axis=1)  # merge the segments on the same via

    R_mat_tot = np.linalg.inv(U_H_inv_R)  # the total resistance matrix

    R_mat_nodes = R_mat_tot[0:via_xy_b.shape[0], 0:via_xy_b.shape[0]]  # nodal resistance matrix
    ref_via_num = -1


    via_list = list(range(0, via_xy_b.shape[0]))
    del via_list[ref_via_num]

    # subtract the row and column corresponding to the reference via number
    for i in via_list:
        R_mat_nodes[i, :] = R_mat_nodes[i, :] - R_mat_nodes[ref_via_num, :]
        R_mat_nodes[:, i] = R_mat_nodes[:, i] - R_mat_nodes[:, ref_via_num]

    R = R_mat_nodes[np.ix_(via_list, via_list)]#+1e-12



    G = np.linalg.inv(R)

    rb = np.zeros((R.shape))

    for i in range(0, rb.shape[0]):
        for j in range(0, rb.shape[1]):
            if i != j:
                rb[i, j] = -1 / G[i, j]
            else:
                rb[i, j] = 1 / (np.sum(G[i, :]))
    return rb

import numpy as np
from itertools import combinations


def org_resistance(stackup, via_type, start_layer, stop_layer, via_xy, decap_via_type, decap_via_xy, decap_via_loc, ic_via_xy, ic_via_type, ic_via_loc):

# Generate branches based on resistors (consider resistance within plane → all positions are independent nodes)
# Create vertical branches + connect nodes on the same layer using layer_type-based rules

    branch = np.zeros((0, 9))
    branch_n = 0
    node_n = -1
    vertical_nodes = []
    xy_node_map = {}

    top_layer = np.min(start_layer)

    # Identify IC vias
    is_ic_via = np.zeros(len(via_xy), dtype=bool)
    is_ic_via[:len(ic_via_xy)] = True


    is_decap_via = np.zeros(len(via_type), dtype=bool)
    for idx in range(len(decap_via_xy)):
        is_decap_via[idx + len(ic_via_xy)] = True


    # Store shared nodes separately for PWR and GND
    shared_top_node_map = {}

    for via_n in range(len(via_type)):
        current_type = via_type[via_n]  # 1 (PWR) or 0 (GND)
        x = round(via_xy[via_n][0], 6)
        y = round(via_xy[via_n][1], 6)

        start_cavity = start_layer[via_n]
        end_cavity = stop_layer[via_n]
        key1 = (x, y, start_cavity)

        # Condition 1: If top layer → create shared node using (x,y,type)
        # if start_cavity == top_layer:
        if start_cavity == top_layer and is_ic_via[via_n] and not is_decap_via[via_n]:

            shared_key = (current_type)
            if shared_key in shared_top_node_map:
                node1 = shared_top_node_map[shared_key]
            else:
                node_n += 1
                node1 = node_n
                shared_top_node_map[shared_key] = node1
            xy_node_map[key1] = node1
            vertical_nodes.append((start_cavity, node1, current_type, via_n))


        # Condition 2: If start_layer == 0 and IC via → reuse shared node
        elif is_ic_via[via_n] and start_cavity == 0 and not is_decap_via[via_n]:
            shared_key = (current_type)
            if shared_key in shared_top_node_map:
                node1 = shared_top_node_map[shared_key]
            else:
                node_n += 1
                node1 = node_n
                shared_top_node_map[shared_key] = node1
            xy_node_map[key1] = node1
            vertical_nodes.append((start_cavity, node1, current_type, via_n))


        # Else: assign independent node for each via
        else:
            if key1 in xy_node_map:
                node1 = xy_node_map[key1]
            else:
                node_n += 1
                node1 = node_n
                xy_node_map[key1] = node1
                vertical_nodes.append((start_cavity, node1, current_type, via_n))

        # Vertical traversal across cavities
        curr_cavity = start_cavity
        while curr_cavity < end_cavity:
            found_match = False
            for next_cavity in range(curr_cavity + 1, end_cavity + 1):
                if next_cavity == end_cavity or stackup[next_cavity] == current_type:
                    key2 = (x, y, next_cavity)
                    if key2 in xy_node_map:
                        node2 = xy_node_map[key2]
                    else:
                        node_n += 1
                        node2 = node_n
                        xy_node_map[key2] = node2
                        vertical_nodes.append((next_cavity, node2, current_type, via_n))

                    new_branch = np.array([[branch_n, node1, node2, via_n, via_n, current_type, curr_cavity, curr_cavity, next_cavity]])
                    branch = np.vstack([branch, new_branch])
                    branch_n += 1

                    node1 = node2
                    curr_cavity = next_cavity
                    found_match = True
                    break
            if not found_match:
                break

# === Add horizontal branches ===
    
    # Create mapping from node to via_idx based on vertical_nodes
    node_to_via_idx = {node: via_idx for (lay, node, vtype, via_idx) in vertical_nodes}

    unique_layers = sorted(set([layer for (layer, _, _, _) in vertical_nodes]))
    seen_pairs = set()

    for layer in unique_layers:
        layer_type = stackup[layer] 
        if layer_type not in [0, 1]:
            continue

        # Only consider vias that match the layer_type of the current layer
        nodes_on_layer = [
            (node, vtype) for (lay, node, vtype, _) in vertical_nodes
            if lay == layer and vtype == layer_type
        ]
        if len(nodes_on_layer) < 2:
            continue

        # Extract only nodes that can be connected within the current layer
        filtered_nodes = [node for (node, _) in nodes_on_layer]

        for (ni, nj) in combinations(filtered_nodes, 2):
            pair = tuple(sorted((ni, nj)))
            if ni == nj or pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            vi = node_to_via_idx.get(ni, -1)
            vj = node_to_via_idx.get(nj, -1)

            new_branch = np.array([[branch_n, ni, nj, vi, vj, layer_type, layer, 0, 0]])
            branch = np.vstack([branch, new_branch])
            branch_n += 1


    # === Decap pwr-gnd branch ===
    top_layer = 0  # fisrt_layer
    bottom_layer = stackup.shape[0] - 1  # last_layer
    used_nodes = set()

    for pi in range(len(decap_via_type)):
        if decap_via_type[pi] != 1:
            continue
        if pi in used_nodes:
            continue  #Skip if the PWR via is already used

        for gi in range(len(decap_via_type)):
            if decap_via_type[gi] != 0:
                continue
            if gi in used_nodes:
                continue  #Skip if the GND via is already used

            layer_pi = top_layer if decap_via_loc[pi] == 1 else bottom_layer
            layer_gi = top_layer if decap_via_loc[gi] == 1 else bottom_layer

            key1 = (round(decap_via_xy[pi][0], 6), round(decap_via_xy[pi][1], 6), layer_pi)
            key2 = (round(decap_via_xy[gi][0], 6), round(decap_via_xy[gi][1], 6), layer_gi)

            if key1 not in xy_node_map or key2 not in xy_node_map:
                continue

            ni = xy_node_map[key1]
            nj = xy_node_map[key2]

            vi = node_to_via_idx.get(ni, -1)
            vj = node_to_via_idx.get(nj, -1)

            # Add branch
            new_branch = np.array([[branch_n, ni, nj, vi, vj, -1, layer_pi, 0, 0]])  # layer_pi or layer_gi?
            branch = np.vstack([branch, new_branch])
            branch_n += 1

            # Mark nodes as used
            used_nodes.update([pi, gi])
            break  #Break after connecting this PWR via

    #Connect top-layer decap PWR vias
    top_layer = 0  
    xy_offset_m = 0.0001  # 0.1mm = 0.0001m

    # 1. Select a coordinate of a PWR via at the top layer to define the new node location
    for idx, (xy, typ) in enumerate(zip(np.vstack([ic_via_xy, decap_via_xy]),
                                        np.concatenate([ic_via_type, decap_via_type]))):
        node_key = (round(xy[0], 6), round(xy[1], 6), top_layer)
        if typ == 1 and node_key in xy_node_map:
            base_xy = xy
            vj_temp = idx  # via index for temp_node
            break
    else:
        raise ValueError("There is no PWR via located on the top layer (layer 0).")

    # 2. Create and assign coordinates for the temp_node
    new_xy = (round(base_xy[0] + xy_offset_m, 9), round(base_xy[1], 9), top_layer)
    temp_node = max(xy_node_map.values()) + 1
    xy_node_map[new_xy] = temp_node

    connected_nodes = set()

    # 3. Connect PWR nodes from decap branches at layer 0 to the temp_node
    for b in branch:
        _, ni, nj, vi, vj, port, lay, _, _ = b
        if lay != top_layer:
            continue
        if port != -1:
            continue  # Skip branches connected to a port
        pwr_node = int(ni)  # Assume node 'ni' in decap branch is always the PWR node
        if pwr_node == 0 or pwr_node in connected_nodes:
            continue
        temp_branch = np.array([[branch_n, pwr_node, temp_node, vi, vj_temp, -1, top_layer, 0, 0]])
        branch = np.vstack([branch, temp_branch])
        branch_n += 1
        connected_nodes.add(pwr_node)


    # 4. Connect GND vias at layer 0 to the temp_node
    for idx, (xy, typ) in enumerate(zip(np.vstack([ic_via_xy, decap_via_xy]),
                                        np.concatenate([ic_via_type, decap_via_type]))):
        if typ != 0:
            continue 
        node_key = (round(xy[0], 6), round(xy[1], 6), top_layer)
        if node_key not in xy_node_map:
            continue
        gnd_node = xy_node_map[node_key]
        if gnd_node == 0 or gnd_node in connected_nodes:
            continue
        vi = idx
        temp_branch = np.array([[branch_n, gnd_node, temp_node, vi, vj_temp, -1, top_layer, 0, 0]])
        branch = np.vstack([branch, temp_branch])
        branch_n += 1
        connected_nodes.add(gnd_node)


    return branch




def main_res(brd, stackup, die_t, d, start_layer, stop_layer, decap_via_type, decap_via_xy, decap_via_loc, ic_via_xy, ic_via_type, ic_via_loc):
    import numpy as np
    from copy import deepcopy

    if hasattr(brd, 'buried_via_xy') and brd.buried_via_xy is not None and len(brd.buried_via_xy) > 0:
        via_xy = np.concatenate((brd.ic_via_xy, brd.decap_via_xy, brd.buried_via_xy), axis=0)
        via_type = np.concatenate((brd.ic_via_type, brd.decap_via_type, brd.buried_via_type), axis=0)
    else:
        via_xy = np.concatenate((brd.ic_via_xy, brd.decap_via_xy), axis=0)
        via_type = np.concatenate((brd.ic_via_type, brd.decap_via_type), axis=0)

    branch = org_resistance(stackup, via_type, start_layer, stop_layer, via_xy, decap_via_type, decap_via_xy, decap_via_loc, ic_via_xy, ic_via_type, ic_via_loc)


    branch_num = branch.shape[0]
    node_num = int(np.max(branch[:, [1, 2]]))
    branch_verti = np.where((branch[:, 3] == branch[:, 4]) & (branch[:, 3] != -1))[0] 



    # A matrix
    A = np.zeros((node_num+1, branch_num))
    for i in range(branch.shape[0]):
        A[int(branch[i, 1]), i] = 1
        A[int(branch[i, 2]), i] = -1

    # Initialize zb
    via_radius = brd.via_r
    sigma = 5.814e7
    via_r = 1 / (np.pi * sigma * via_radius ** 2)  

    zb = np.eye(branch_num) * 1e-5  
    for i in range(branch.shape[0]):
        vj = branch[i, 4]

        if vj == -1:
            zb[i, i] = 1e-5

        elif i in branch_verti:
            layer_b = int(branch[i, 6])
            zb[i, i] = die_t[layer_b]+ d[layer_b]* via_r 

    for i in branch_verti:
        start_lay = int(branch[i, 7])
        stop_lay = int(branch[i, 8])
        if start_lay > stop_lay:
            start_lay, stop_lay = stop_lay, start_lay  
        total = 0.0
        for lay in range(start_lay, stop_lay + 1):  # inclusive range from start to stop layer
            if 0 <= lay < len(d):
                if lay == stop_lay:
                    # Do NOT include dielectric thickness on the last layer
                    total += d[lay]

                elif lay < len(die_t):
                    # Normal case: include both conductor and dielectric thickness
                    total += d[lay] + die_t[lay]

                else:
                    # In case die_t is shorter than d, add only conductor thickness
                    total += d[lay]

            else:
                print(f"[Warning] layer index {lay} out of bounds for d/die_t → skipped")

        zb[i, i] = total * via_r

    bxy = brd.bxy  # polygon shape

    #If bxy is 3D with shape (1, N, 2): single shape only
    if isinstance(bxy, np.ndarray) and bxy.ndim == 3 and bxy.shape[0] == 1 and bxy.shape[2] == 2:
        bxy = bxy[0]  

    # Check again and duplicate for each layer
    if isinstance(bxy, np.ndarray) and bxy.ndim == 2 and bxy.shape[1] == 2:
        num_layers = len(stackup)
        bxy = [bxy.copy() for _ in range(num_layers)]

    horizontal_branches = branch[
    (branch[:, 0] > branch_verti.shape[0] - 1)
]

    target_layers = np.unique(horizontal_branches[:, 6].astype(int))

    for i in target_layers:
        corres_branches = np.where(branch[:, 6] == i)
        branches = branch[corres_branches]
        # branches = branches[
        #     (branches[:, 0] > branch_verti.shape[0] - 1) & (branches[:, 5] != -1)
        # ]
        branches = branches[
            (branches[:, 0] > branch_verti.shape[0] - 1) 
        ]



        # skip if no horizontal branch found
        if branches.shape[0] == 0:
            continue

        values_col3 = branches[:, 3]
        values_col4 = branches[:, 4]
        combined_values = np.concatenate((values_col3, values_col4))
        unique_values = np.unique(combined_values)
        via_xy_b = via_xy[unique_values.astype(int)]

        for i in range(len(bxy)):
            bxy_b = np.array(bxy[i]) 

        via_xy_b = np.vstack([via_xy_b, bxy_b[1] + 1e-5])
        rb = planesresistance(bxy_b, via_xy_b, via_radius, d[i])


        for j in range(branches.shape[0]):
            r1 = np.where(unique_values == branches[j, 3])
            r2 = np.where(unique_values == branches[j, 4])
            idx1 = r1[0][0]
            idx2 = r2[0][0]
            branch_idx = int(branches[j, 0])

            zb[branch_idx, branch_idx] = abs(rb[idx1, idx2])


    max_node = int(np.max(branch[:, [1, 2]]))  


    # for i in range(0, 22):
    #     zb[i,i]=1

    # After computation, generate impedance matrix
    at = np.transpose(A)
    yb = np.linalg.inv(zb)

    yn = np.matmul(A, yb)
    yn1 = np.matmul(yn, at)
    yn2 = np.delete(yn1, max_node, 0)
    yn3 = np.delete(yn2, max_node, 1)


    zn2 = np.linalg.inv(yn3)
    t2 = time.time()

    return zn2[0][0]





