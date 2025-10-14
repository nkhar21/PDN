from copy import deepcopy
import numpy as np
from math import sqrt, pi, sin, cos, log, atan
import skrf as rf
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon
import numpy as np
import time

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def max_value(inputlist): 
    return max([max(sublist) for sublist in inputlist])


def seg_port(x0, y0, r, n=6):
    # Define a function for port segmentation
    # for port boundary, it must rotate clockwise
    # x0, y0 is the port center location, r is radius
    # n is the number of the segments
    # sxy is a S*4 matrix, x1, y1, x2, y2
    dtheta = 2 * pi / n
    n = int(n)
    sxy = np.ndarray((n, 4))
    for i in range(0, n):
        sxy[i, 0] = x0 + r * cos(-(i) * dtheta)
        sxy[i, 1] = y0 + r * sin(-(i) * dtheta)
        sxy[i, 2] = x0 + r * cos(-(i + 1) * dtheta)
        sxy[i, 3] = y0 + r * sin(-(i + 1) * dtheta)
    return sxy


# short 1 port of S-parameter
def short_1port(input_net, map2orig_input=[0, 1], shorted_port=1):
    # default shorted port for decap is port 1. if input_net is a network, need to point out shorted port #
    short_net = deepcopy(input_net.s11)
    short_net.s = -1 * np.ones(short_net.f.shape[0])
    output_net = rf.network.connect(input_net, shorted_port, short_net, 0)
    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[shorted_port]
    return output_net, map2orig_output


def connect_1decap(input_net_z, map2orig_input, connect_port, decap_z11):
    Zaa = deepcopy(input_net_z)
    Zaa = np.delete(Zaa, connect_port, 1)
    Zaa = np.delete(Zaa, connect_port, 2)
    Zpp = input_net_z[:, connect_port, connect_port]
    Zpp = Zpp.reshape((Zpp.shape[0], 1, 1))
    Zqq = decap_z11
    Zap = input_net_z[:, :, connect_port]
    Zap = Zap.reshape((Zap.shape[0], Zap.shape[1], 1))
    Zap = np.delete(Zap, connect_port, 1)
    Zpa = input_net_z[:, connect_port, :]
    Zpa = Zpa.reshape((Zpa.shape[0], 1, Zpa.shape[1]))
    Zpa = np.delete(Zpa, connect_port, 2)
    inv = np.linalg.inv(Zpp + Zqq)
    second = np.einsum('rmn,rkk->rmn', Zap, inv)
    second = np.einsum('rmn,rnd->rmd', second, Zpa)
    output_net_z = Zaa - second
    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[connect_port]
    return output_net_z, map2orig_output


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


# find the indices of multiple elements in a list
def find_index(input_list, element_list):
    index_list = []
    for e in element_list:
        index_list.append(input_list.index(int(e)))
    return index_list


# merge a square matrix by list of indices
def merge_square_mat(Min, merge_port_list, map2orig_input=[]):
    # by default, the first index in merge_port_list will be kept after merging
    Mout = deepcopy(Min)
    if map2orig_input == []:
        map2orig_output = list(range(0, Min.shape[0]))
    else:
        map2orig_output = deepcopy(map2orig_input)

    for i in merge_port_list[1:]:
        Mout[merge_port_list[0], :] += Mout[i, :]
        Mout[:, merge_port_list[0]] += Mout[:, i]

    Mout = np.delete(np.delete(Mout, merge_port_list[1:], axis=0), merge_port_list[1:], axis=1)

    delete_multiple_element(map2orig_output, merge_port_list[1:])
    return Mout, map2orig_output


# branch contains: branch #, node1, node2, cavity #, global via #
# branch_new contains: new branch #, new node1, new node2, old branch #, old node1, old node2
def merge_L_big(L_big, branch_merge_list, branch):
    old_branch_nodes = branch[:, 0:3]  # three columns: branch #, node1, node2
    map2orig = list(range(0, L_big.shape[0]))
    
    for i in range(0, len(branch_merge_list)):
        for c in range(0, branch_merge_list[i].shape[1]):
            if branch_merge_list[i].shape[0] > 1:
                old_branch_nodes[find_index(old_branch_nodes[:, 0].tolist(), [branch_merge_list[i][0, c]]), 2] = \
                    old_branch_nodes[find_index(old_branch_nodes[:, 0].tolist(), [branch_merge_list[i][-1, c]]), 2]
                old_branch_nodes = np.delete(old_branch_nodes,
                                             find_index(map2orig, branch_merge_list[i][1:, c].tolist()), axis=0)
                L_big, map2orig = merge_square_mat(L_big, find_index(map2orig, branch_merge_list[i][:, c].tolist()),
                                                   map2orig)


    L_new_inv = np.linalg.inv(L_big)

    for i in range(0, len(branch_merge_list)):
        if branch_merge_list[i].shape[1] > 1:
            old_branch_nodes = np.delete(old_branch_nodes, find_index(map2orig, branch_merge_list[i][0, 1:].tolist()),
                                         axis=0)
            L_new_inv, map2orig = merge_square_mat(L_new_inv, find_index(map2orig, branch_merge_list[i][0, :].tolist()),
                                                   map2orig)

    old_branch_left = np.unique(old_branch_nodes[:, 0]).tolist()
    old_node_left = np.unique(old_branch_nodes[:, 1:3]).tolist()

    new_branch_nodes = deepcopy(old_branch_nodes)
    for i in range(0, new_branch_nodes.shape[0]):
        new_branch_nodes[i, 0] = old_branch_left.index(old_branch_nodes[i, 0])
        new_branch_nodes[i, 1] = old_node_left.index(old_branch_nodes[i, 1])
        new_branch_nodes[i, 2] = old_node_left.index(old_branch_nodes[i, 2])

    new_old_node_map = np.zeros((len(old_node_left), 2))
    new_old_node_map[:, 0] = np.array(list(range(0, len(old_node_left))))
    new_old_node_map[:, 1] = np.unique(old_branch_nodes[:, 1:3])
    return L_new_inv, old_branch_nodes, new_branch_nodes, new_old_node_map


def calc_lpul_bem(via_xy, via_r, sxy, option='v1', n=6):
    '''
    Option: v1 means using small port approximation
    v2 means dividing each port into several segments

    '''
    # er = 4.4
    # C_pul = np.array([er*e*area/1])
    # directly calculate z parameters given decap via locations
    
    e = 8.85e-12 # ε0
    u = 4 * pi * 1e-7 # μ0
    
    d = 1   # unit thickness (scaled later)
    R = 0.3 # “far-field” distance cap

    area = np.array([PolyArea(sxy[:, 0], sxy[:, 1])]) # board cross-sectional area

    if option == 'v1':

        Ntot = via_xy.shape[0] + sxy.shape[0]
        D = np.zeros((Ntot, Ntot))
        Gh = np.zeros((Ntot, Ntot))
        L_pul = np.zeros((via_xy.shape[0], via_xy.shape[0]))
        for k in range(0, Ntot):
            for m in range(0, Ntot):
                if m >= sxy.shape[0] and k != m:
                    D[k, m] = 0
                elif m >= sxy.shape[0] and k == m:
                    D[k, m] = -1
                elif m < sxy.shape[0] and k >= sxy.shape[0]:
                    xk = via_xy[k - sxy.shape[0], 0]
                    yk = via_xy[k - sxy.shape[0], 1]
                    xa = sxy[m, 0]
                    ya = sxy[m, 1]
                    xe = sxy[m, 2]
                    ye = sxy[m, 3]
                    lm = sqrt((xa - xe) ** 2 + (ya - ye) ** 2)
                    v1 = 1 / lm * ((xk - xa) * (xe - xa) + (yk - ya) * (ye - ya))
                    v2 = -1 / lm * ((xk - xa) * (ye - ya) - (yk - ya) * (xe - xa))
                    ra = sqrt((xa - xk) ** 2 + (ya - yk) ** 2)
                    # if np.abs(v2)<1e-10:
                    if np.abs(v1 ** 2 - ra ** 2) < 1e-10:
                        D[k, m] = 0
                    else:
                        D[k, m] = v2 / pi * (1 / sqrt(ra ** 2 - v1 ** 2) * (atan((-v1 + lm) / sqrt(ra ** 2 - v1 ** 2)) +
                                                                            atan(v1 / sqrt(ra ** 2 - v1 ** 2))) - lm / (
                                                         R ** 2))
                elif m < sxy.shape[0] and k < sxy.shape[0]:
                    xk = (sxy[k, 0] + sxy[k, 2]) / 2
                    yk = (sxy[k, 1] + sxy[k, 3]) / 2
                    xa = sxy[m, 0]
                    ya = sxy[m, 1]
                    xe = sxy[m, 2]
                    ye = sxy[m, 3]

                    # pdb.set_trace()

                    lm = sqrt((xa - xe) ** 2 + (ya - ye) ** 2)

                    v1 = 1 / lm * ((xk - xa) * (xe - xa) + (yk - ya) * (ye - ya))
                    v2 = -1 / lm * ((xk - xa) * (ye - ya) - (yk - ya) * (xe - xa))
                    ra = sqrt((xa - xk) ** 2 + (ya - yk) ** 2)
                    # if np.abs(v2)<1e-10:
                    if np.abs(v1 ** 2 - ra ** 2) < 1e-10:
                        D[k, m] = 0
                    else:
                        D[k, m] = v2 / pi * (1 / sqrt(ra ** 2 - v1 ** 2) * (atan((-v1 + lm) / sqrt(ra ** 2 - v1 ** 2)) +
                                                                            atan(v1 / sqrt(ra ** 2 - v1 ** 2))) - lm / (
                                                         R ** 2))
                    
        for k in range(0, Ntot):
            for m in range(0, Ntot):
                if m >= sxy.shape[0] and k == m:
                    Gh[k, m] = -u * d / (4 * area) * (via_r ** 2) * (4 * log(via_r / R) - 1)

                elif m >= sxy.shape[0] and k != m:
                    Gh[k, m] = 0
                elif m < sxy.shape[0] and k >= sxy.shape[0]:
                    xk = via_xy[k - sxy.shape[0], 0]
                    yk = via_xy[k - sxy.shape[0], 1]
                    xa = sxy[m, 0]
                    ya = sxy[m, 1]
                    xe = sxy[m, 2]
                    ye = sxy[m, 3]
                    lm = sqrt((xa - xe) ** 2 + (ya - ye) ** 2)
                    v1 = 1 / lm * ((xk - xa) * (xe - xa) + (yk - ya) * (ye - ya))
                    v2 = -1 / lm * ((xk - xa) * (ye - ya) - (yk - ya) * (xe - xa))
                    ra = sqrt((xa - xk) ** 2 + (ya - yk) ** 2)
                    # if np.abs(v2)<1e-10:
                    if np.abs(v1 ** 2 - ra ** 2) < 1e-10:
                        Gh[k, m] = 0
                    else:
                        Gh[k, m] = u * d * v2 / (4 * pi * area) * (
                                    lm * log((lm ** 2 + ra ** 2 - 2 * v1 * lm) / R ** 2) -
                                    v1 * log((lm ** 2 + ra ** 2 - 2 * v1 * lm) / ra ** 2)
                                    - (1 / 3 * lm ** 3 + ra ** 2 * lm - lm ** 2 * v1) / (2 * R ** 2) - 3 * lm
                                    + 2 * sqrt(ra ** 2 - v1 ** 2) *
                                    (atan((-v1 + lm) / sqrt(ra ** 2 - v1 ** 2)) + atan(v1 / sqrt(ra ** 2 - v1 ** 2))))
                elif m < sxy.shape[0] and k < sxy.shape[0]:
                    xk = (sxy[k, 0] + sxy[k, 2]) / 2
                    yk = (sxy[k, 1] + sxy[k, 3]) / 2
                    xa = sxy[m, 0]
                    ya = sxy[m, 1]
                    xe = sxy[m, 2]
                    ye = sxy[m, 3]
                    lm = sqrt((xa - xe) ** 2 + (ya - ye) ** 2)
                    v1 = 1 / lm * ((xk - xa) * (xe - xa) + (yk - ya) * (ye - ya))
                    v2 = -1 / lm * ((xk - xa) * (ye - ya) - (yk - ya) * (xe - xa))
                    ra = sqrt((xa - xk) ** 2 + (ya - yk) ** 2)
                    # if np.abs(v2)<1e-10:
                    if np.abs(v1 ** 2 - ra ** 2) < 1e-10:
                        Gh[k, m] = 0
                    else:
                        Gh[k, m] = u * d * v2 / (4 * pi * area) * (
                                    lm * log((lm ** 2 + ra ** 2 - 2 * v1 * lm) / R ** 2) -
                                    v1 * log((lm ** 2 + ra ** 2 - 2 * v1 * lm) / ra ** 2)
                                    - (1 / 3 * lm ** 3 + ra ** 2 * lm - lm ** 2 * v1) / (2 * R ** 2) - 3 * lm
                                    + 2 * sqrt(ra ** 2 - v1 ** 2) *
                                    (atan((-v1 + lm) / sqrt(ra ** 2 - v1 ** 2)) + atan(v1 / sqrt(ra ** 2 - v1 ** 2))))
        E_D_inv = np.linalg.inv(np.identity(Ntot) - D)
        Gh_k = np.sum(Gh, axis=1)
        G_k = np.zeros((Ntot))
        # Excite each via and obtain inductance
        for j in range(0, via_xy.shape[0]):
            m = j + sxy.shape[0]
            for k in range(0, Ntot):
                if k == m:
                    G_k[k] = -u * d / pi * log(via_r / R)
                elif k != m and k >= sxy.shape[0]:
                    r_km = sqrt((via_xy[m - sxy.shape[0], 0] - via_xy[k - sxy.shape[0], 0]) ** 2
                                + (via_xy[m - sxy.shape[0], 1] - via_xy[k - sxy.shape[0], 1]) ** 2)
                    r_km = max(r_km, 1e-10) # guard against zero distance (when xy of two vias are identical)
                    G_k[k] = -u * d / pi * (log(r_km / R) - r_km ** 2 / (2 * R ** 2))
                else:
                    xm = via_xy[m - sxy.shape[0], 0]
                    ym = via_xy[m - sxy.shape[0], 1]
                    xk = (sxy[k, 0] + sxy[k, 2]) / 2
                    yk = (sxy[k, 1] + sxy[k, 3]) / 2
                    r_km = sqrt((xm - xk) ** 2 + (ym - yk) ** 2)
                    r_km = max(r_km, 1e-10) # guard against zero distance (when xy of two vias are identical)
                    G_k[k] = -u * d / pi * (log(r_km / R) - r_km ** 2 / (2 * R ** 2))
            G = Gh_k + G_k
            L = np.dot(E_D_inv, G)
            L_pul[:, j] = L[sxy.shape[0]:Ntot]
        # add internal inductance
        # for i in range(0,L_pul.shape[0]):
        #     L_pul[i,i] += u*d/(8*pi)

    elif option == 'v2':
        Nseg = sxy.shape[0]
        Nvia = via_xy.shape[0]
        # L_pul = np.zeros((Nvia,Nvia))
        l_pul_nseg = np.zeros((Nvia * n, Nvia * n))

        for i in range(0, Nvia):
            s = seg_port(via_xy[i, 0], via_xy[i, 1], via_r, n)
            if i == 0:
                svia = s
            else:
                svia = np.concatenate((svia, s))

        sxy = np.concatenate((svia, sxy))
        Ntot = Nvia * n + Nseg

        # calculate D matrix
        D = np.zeros((Ntot, Ntot))
        for k in range(0, Ntot):
            for m in range(0, Ntot):
                if k == m:
                    D[k, m] = 0
                else:
                    xk = (sxy[k, 0] + sxy[k, 2]) / 2
                    yk = (sxy[k, 1] + sxy[k, 3]) / 2
                    xm1 = sxy[m, 0]
                    ym1 = sxy[m, 1]
                    xm2 = sxy[m, 2]
                    ym2 = sxy[m, 3]
                    xm = (xm1 + xm2) / 2
                    ym = (ym1 + ym2) / 2
                    lm = sqrt((xm1 - xm2) ** 2 + (ym1 - ym2) ** 2)
                    rkm = sqrt((xk - xm) ** 2 + (yk - ym) ** 2)
                    cos_phi = ((xm - xk) * (ym2 - ym1) - (ym - yk) * (xm2 - xm1)) / (rkm * lm)
                    D[k, m] = 1 / pi * lm * (1 / rkm - rkm / R ** 2) * cos_phi
        # D = D/2
        E_D_inv = np.linalg.inv(np.identity(Ntot) - D)

        # excite each via and obtain the mutual inductance
        for j in range(0, Nvia * n):
            G = np.zeros((Ntot))
            for k in range(0, Ntot):
                G_km = np.zeros((Ntot))
                Gh_km = np.zeros((Ntot))
                for m in range(0, Ntot):
                    if k == m and m == j:
                        xm1 = sxy[m, 0]
                        ym1 = sxy[m, 1]
                        xm2 = sxy[m, 2]
                        ym2 = sxy[m, 3]
                        lm = sqrt((xm1 - xm2) ** 2 + (ym1 - ym2) ** 2)
                        G_km[m] = -u * d / pi * (log(lm / 2 / R) - 1)
                        Gh_km[m] = 0
                    elif k == m and m != j:
                        G_km[m] = 0
                        Gh_km[m] = 0
                    elif k != m and m == j:
                        xk = (sxy[k, 0] + sxy[k, 2]) / 2
                        yk = (sxy[k, 1] + sxy[k, 3]) / 2
                        xm1 = sxy[m, 0]
                        ym1 = sxy[m, 1]
                        xm2 = sxy[m, 2]
                        ym2 = sxy[m, 3]
                        xm = (xm1 + xm2) / 2
                        ym = (ym1 + ym2) / 2
                        lm = sqrt((xm1 - xm2) ** 2 + (ym1 - ym2) ** 2)
                        rkm = sqrt((xk - xm) ** 2 + (yk - ym) ** 2)
                        cos_phi = ((xm - xk) * (ym2 - ym1) - (ym - yk) * (xm2 - xm1)) / (rkm * lm)
                        G_km[m] = -u * d / pi * (log(rkm / R) - rkm ** 2 / (2 * R ** 2))
                        Gh_km[m] = u * d / (4 * pi * area) * lm * (
                                    rkm * (2 * log(rkm / R) - 1) - rkm ** 3 / (2 * R ** 2)) * cos_phi
                    else:
                        xk = (sxy[k, 0] + sxy[k, 2]) / 2
                        yk = (sxy[k, 1] + sxy[k, 3]) / 2
                        xm1 = sxy[m, 0]
                        ym1 = sxy[m, 1]
                        xm2 = sxy[m, 2]
                        ym2 = sxy[m, 3]
                        xm = (xm1 + xm2) / 2
                        ym = (ym1 + ym2) / 2
                        lm = sqrt((xm1 - xm2) ** 2 + (ym1 - ym2) ** 2)
                        rkm = sqrt((xk - xm) ** 2 + (yk - ym) ** 2)
                        cos_phi = ((xm - xk) * (ym2 - ym1) - (ym - yk) * (xm2 - xm1)) / (rkm * lm)
                        G_km[m] = 0
                        Gh_km[m] = u * d / (4 * pi * area) * lm * (
                                    rkm * (2 * log(rkm / R) - 1) - rkm ** 3 / (2 * R ** 2)) * cos_phi

                G[k] = np.sum(G_km) + np.sum(Gh_km)

            L = np.dot(E_D_inv, G)
            l_pul_nseg[:, j] = L[0:Nvia * n]

        # create a list to merge the segments on the same via
        reduce_list = list(range(0, (Nvia - 1) * n + 1, n))

        L_pul = np.add.reduceat(np.add.reduceat(l_pul_nseg, reduce_list, axis=0), reduce_list, axis=1) / (n * n)

    return L_pul


# A function to organize and merge the parallel branches for multi-layer PDN with arbitrary shape
def org_merge_pdn(stackup, via_type, start_layer, stop_layer,
                  top_port_num, bot_port_num, top_port_grp, bot_port_grp):



    branch = np.zeros((1, 4))  # four columns: node 1, node 2, cavity num, global via num
    layer_com_node = -1 * np.ones((stackup.shape[0]))  # the common node number for each layer
    # positive node and negative node for different ports
    port_node = -1 * np.ones((max(max_value(top_port_num), max_value(bot_port_num)) + 1, 2))  
    # node num, node type (1 or 0), layer number of the port groups
    port_grp_node_num = -1 * np.ones((int(np.max([np.max(top_port_grp),
                                        np.max(bot_port_grp)])) + 1,3))  

    branch_n = 0 # branch counter
    node_n = -1 # node counter

    for via_n in range(0, via_type.shape[0]):  # iterate over all vias in the design
        for cavity_n in range(start_layer[via_n], stop_layer[via_n]):  # iterate over each cavity that via passes through
            
            # ============================================================
            # CASE 1 — Via starts at top layer (layer 0) through an antipad,
            #          and it is NOT part of a port group.
            #          (e.g. standalone decap or IC via with its own port)
            # ============================================================
            if cavity_n == start_layer[via_n] and start_layer[via_n] == 0 \
                    and stackup[start_layer[via_n]] != via_type[via_n] \
                    and top_port_grp[via_n] == -1:

                # create node1: via entry point on the top layer
                node_n += 1
                node1 = deepcopy(node_n)

                # check next layer below
                if stackup[cavity_n + 1] != via_type[via_n]:
                    # next layer also antipad ⇒ via isolated through cavity
                    # create new independent node2 for via exit
                    node_n += 1
                    node2 = deepcopy(node_n)
                elif layer_com_node[cavity_n + 1] == -1:
                    # next layer matches via type but not assigned a common node yet
                    # create new node2 and mark that layer’s shared node
                    node_n += 1
                    node2 = deepcopy(node_n)
                    layer_com_node[cavity_n + 1] = node2
                else:
                    # next layer already has a shared node ⇒ reuse it
                    node2 = layer_com_node[cavity_n + 1]

                # -----------------------------------------
                # Assign nodes to ports if this via belongs
                # to an explicitly numbered top port.
                # top_port_num[via_n][0] = port index, or -1 if none.
                # -----------------------------------------
                if top_port_num[via_n][0] != -1:
                    port_idx = top_port_num[via_n][0]
                    if via_type[via_n] == 1:  # power via
                        # assign this node as port's positive node (V+)
                        if port_node[port_idx, 0] == -1:
                            port_node[port_idx, 0] = node1
                    elif via_type[via_n] == 0:  # ground via
                        # assign this node as port's negative node (V−)
                        port_node[port_idx, 1] = node1

            # ============================================================
            # CASE 2 — Via starts at top surface through antipad,
            #          but it BELONGS to a top port group (e.g. IC pad cluster)
            # ============================================================
            elif cavity_n == start_layer[via_n] and start_layer[via_n] == 0 \
                    and stackup[start_layer[via_n]] != via_type[via_n] \
                    and top_port_grp[via_n] != -1:

                # if this is the first via in its port group, create group node
                if port_grp_node_num[top_port_grp[via_n], 0] == -1:
                    node_n += 1
                    node1 = deepcopy(node_n)
                    # record node info in port_grp_node_num:
                    # [node_id, type (1=pwr/0=gnd), layer]
                    port_grp_node_num[top_port_grp[via_n], 0] = node1
                    port_grp_node_num[top_port_grp[via_n], 1] = via_type[via_n]
                    port_grp_node_num[top_port_grp[via_n], 2] = start_layer[via_n]
                else:
                    # reuse existing group node for same port
                    node1 = port_grp_node_num[top_port_grp[via_n], 0]

                # now handle the lower side of the via
                if stackup[cavity_n + 1] != via_type[via_n]:
                    # next layer is antipad ⇒ isolated cavity ⇒ new node
                    node_n += 1
                    node2 = deepcopy(node_n)
                elif layer_com_node[cavity_n + 1] == -1:
                    # next layer matches via type but no common node yet ⇒ assign one
                    node_n += 1
                    node2 = deepcopy(node_n)
                    layer_com_node[cavity_n + 1] = node2
                else:
                    # next layer already shares a common node ⇒ reuse
                    node2 = layer_com_node[cavity_n + 1]

                # assign port connections (same as before)
                if top_port_num[via_n][0] != -1:
                    port_idx = top_port_num[via_n][0]
                    if via_type[via_n] == 1:
                        if port_node[port_idx, 0] == -1:
                            port_node[port_idx, 0] = node1
                    elif via_type[via_n] == 0:
                        port_node[port_idx, 1] = node1

            # ============================================================
            # CASE 3 — Top layer matches via type (no antipad on surface)
            #          Typically an IC/decap pad landing directly on metal.
            #          The via “starts” on a metal region at layer 0.
            # ============================================================
            elif cavity_n == start_layer[via_n] and start_layer[via_n] == 0 \
                    and stackup[start_layer[via_n]] == via_type[via_n]:

                # --- TOP NODE (node1) ON THE ENTRY LAYER (layer 0) ---
                if layer_com_node[cavity_n] == -1:
                    # This top layer (0) has not been assigned a shared node yet.
                    # Create a new node and mark it as the common node for layer 0.
                    node_n += 1
                    node1 = deepcopy(node_n)
                    layer_com_node[cavity_n] = node1
                else:
                    # A shared node for layer 0 already exists, so reuse it.
                    node1 = layer_com_node[cavity_n]

                # --- BOTTOM NODE (node2) ON THE NEXT LAYER (layer 1) ---
                if stackup[cavity_n + 1] != via_type[via_n]:
                    # The next layer is *not* the same net as the via (antipad),
                    # so this via segment is isolated within the cavity.
                    # Create an *independent* node on the lower side.
                    node_n += 1
                    node2 = deepcopy(node_n)
                elif layer_com_node[cavity_n + 1] == -1:
                    # Next layer is same net (continuous metal), but it doesn't
                    # have a shared node yet → create one and record it.
                    node_n += 1
                    node2 = deepcopy(node_n)
                    layer_com_node[cavity_n + 1] = node2
                else:
                    # Next layer is same net and already has a shared node,
                    # so reuse that node for the lower end of this segment.
                    node2 = layer_com_node[cavity_n + 1]

                # --- MAP THIS VIA TO A TOP PORT (IF ANY) ---
                if top_port_num[via_n][0] != -1:
                    port_idx = top_port_num[via_n][0]
                    if via_type[via_n] == 1:  # power via
                        # Only set the V+ node if not already claimed
                        if port_node[port_idx, 0] == -1:
                            port_node[port_idx, 0] = node1
                    elif via_type[via_n] == 0:  # ground via
                        # V− node is always the ground-side node
                        port_node[port_idx, 1] = node1


            # ============================================================
            # CASE 4 — Via starts below top surface, not bottom-most
            #          (internal/blind/buried via that begins at a middle layer,
            #           and this segment is not the final one to the bottom.)
            # ============================================================
            elif cavity_n == start_layer[via_n] and start_layer[via_n] != 0 \
                    and cavity_n + 1 < stackup.shape[0] - 1:

                # --- TOP NODE (node1) ON THIS START LAYER ---
                if stackup[start_layer[via_n]] == via_type[via_n] and layer_com_node[cavity_n] == -1:
                    # Start layer matches via net and has no shared node yet → create it.
                    node_n += 1
                    node1 = deepcopy(node_n)
                    layer_com_node[cavity_n] = node1
                elif stackup[start_layer[via_n]] == via_type[via_n]:
                    # Start layer matches via net and already has a shared node → reuse it.
                    node1 = layer_com_node[cavity_n]
                else:
                    # Start layer is *not* same net (antipad) → create an isolated node.
                    node_n += 1
                    node1 = deepcopy(node_n)

                # --- BOTTOM NODE (node2) ON THE NEXT LAYER ---
                if stackup[cavity_n + 1] != via_type[via_n]:
                    # Next layer is antipad relative to via net ⇒ isolated segment end.
                    node_n += 1
                    node2 = deepcopy(node_n)
                elif layer_com_node[cavity_n + 1] == -1:
                    # Next layer is same net but no shared node yet ⇒ create and record it.
                    node_n += 1
                    node2 = deepcopy(node_n)
                    layer_com_node[cavity_n + 1] = node2
                else:
                    # Next layer is same net and already has a shared node ⇒ reuse it.
                    node2 = layer_com_node[cavity_n + 1]


            # ============================================================
            # CASE 5 — Via reaches bottom layer
            #          (handling vias that terminate on the bottom surface)
            # ============================================================
            elif cavity_n == start_layer[via_n] and start_layer[via_n] != 0 \
                    and cavity_n + 1 == stackup.shape[0] - 1:

                # --- TOP NODE (node1) ---
                # Determine node at the upper interface of this via segment.
                if stackup[start_layer[via_n]] == via_type[via_n] and layer_com_node[cavity_n] == -1:
                    # Layer matches via type and hasn’t been assigned → create shared node
                    node_n += 1
                    node1 = deepcopy(node_n)
                    layer_com_node[cavity_n] = node1
                elif stackup[start_layer[via_n]] == via_type[via_n]:
                    # Layer matches via type and already has shared node → reuse
                    node1 = layer_com_node[cavity_n]
                else:
                    # Antipad region: create isolated node
                    node_n += 1
                    node1 = deepcopy(node_n)

                # --- BOTTOM NODE (node2) ---
                # Handle connections depending on port group or shared layer
                if stackup[-1] != via_type[via_n] and bot_port_grp[via_n] == -1:
                    # Bottom layer is an antipad and via not part of any port group → isolated node
                    node_n += 1
                    node2 = deepcopy(node_n)
                elif stackup[-1] != via_type[via_n] and bot_port_grp[via_n] != -1:
                    # Bottom layer is antipad but belongs to a port group
                    # Check if that port group already has a node assigned
                    if port_grp_node_num[bot_port_grp[via_n], 0] == -1:
                        # No node yet → create and record it as shared port group node
                        node_n += 1
                        node2 = deepcopy(node_n)
                        port_grp_node_num[bot_port_grp[via_n], 0] = node2
                        port_grp_node_num[bot_port_grp[via_n], 1] = via_type[via_n]
                        port_grp_node_num[bot_port_grp[via_n], 2] = cavity_n + 1
                    else:
                        # Already assigned → reuse
                        node2 = port_grp_node_num[bot_port_grp[via_n], 0]
                else:
                    # Bottom layer matches via type (no antipad)
                    # Use or create a common node for bottom metal
                    if layer_com_node[-1] == -1:
                        node_n += 1
                        node2 = deepcopy(node_n)
                        layer_com_node[-1] = node2
                    else:
                        node2 = layer_com_node[-1]

                # --- PORT CONNECTION ---
                # If this via belongs to a bottom port, link it to port terminals
                if bot_port_num[via_n][0] != -1:
                    port_idx = bot_port_num[via_n][0]
                    if via_type[via_n] == 1:  # power
                        if port_node[port_idx, 0] == -1:
                            port_node[port_idx, 0] = node2
                    elif via_type[via_n] == 0:  # ground
                        port_node[port_idx, 1] = node2

            # ============================================================
            # CASE 6 — Middle-layer via (spanning multiple cavities)
            #          (i.e. continuing through stack between internal planes)
            # ============================================================
            elif cavity_n > start_layer[via_n] and cavity_n + 1 < stackup.shape[0] - 1:
                # --- TOP NODE (node1) ---
                # The top node for this segment is the previous segment’s bottom node
                node1 = branch[branch_n - 1, 2]

                # --- BOTTOM NODE (node2) ---
                if stackup[cavity_n + 1] == via_type[via_n] and layer_com_node[cavity_n + 1] == -1:
                    # Next layer same net, not assigned → create new shared node
                    node_n += 1
                    node2 = deepcopy(node_n)
                    layer_com_node[cavity_n + 1] = node2
                elif stackup[cavity_n + 1] == via_type[via_n]:
                    # Next layer same net and shared node exists → reuse
                    node2 = layer_com_node[cavity_n + 1]
                else:
                    # Next layer is antipad → create isolated node
                    node_n += 1
                    node2 = deepcopy(node_n)


            # ============================================================
            # CASE 7 — Final via segment reaching bottom of the stack
            #          (the last piece of a multi-cavity via chain)
            # ============================================================
            elif cavity_n > start_layer[via_n] and cavity_n + 1 == stackup.shape[0] - 1:
                # --- TOP NODE (node1) ---
                # Continue from previous segment’s bottom node
                node1 = branch[branch_n - 1, 2]

                # --- BOTTOM NODE (node2) ---
                if stackup[-1] != via_type[via_n] and bot_port_grp[via_n] == -1:
                    # Bottom is antipad and via not in port group → create isolated bottom node
                    node_n += 1
                    node2 = deepcopy(node_n)
                elif stackup[-1] != via_type[via_n] and bot_port_grp[via_n] != -1:
                    # Bottom is antipad but via is part of bottom port group
                    if port_grp_node_num[bot_port_grp[via_n], 0] == -1:
                        # Create and assign a shared node for that group
                        node_n += 1
                        node2 = deepcopy(node_n)
                        port_grp_node_num[bot_port_grp[via_n], 0] = node2
                        port_grp_node_num[bot_port_grp[via_n], 1] = via_type[via_n]
                        port_grp_node_num[bot_port_grp[via_n], 2] = cavity_n + 1
                    else:
                        # Reuse existing group node
                        node2 = port_grp_node_num[bot_port_grp[via_n], 0]
                else:
                    # Bottom layer is same net (metal connection)
                    # Reuse or create bottom shared node
                    if layer_com_node[-1] == -1:
                        node_n += 1
                        node2 = deepcopy(node_n)
                        layer_com_node[-1] = node2
                    else:
                        node2 = layer_com_node[-1]

                # --- PORT CONNECTION ---
                # If this via corresponds to a bottom port, link to terminals
                if bot_port_num[via_n][0] != -1:
                    port_idx = bot_port_num[via_n][0]
                    if via_type[via_n] == 1:  # power
                        if port_node[port_idx, 0] == -1:
                            port_node[port_idx, 0] = node2
                    elif via_type[via_n] == 0:  # ground
                        port_node[port_idx, 1] = node2


            # ============================================================
            # ADD BRANCH ENTRY: [branch_id, node1, node2, cavity_id, via_id]
            # Every via segment through a cavity creates one branch in the graph.
            # ============================================================
            if branch_n == 0:
                # First branch — initialize the array
                branch = np.array([[branch_n, node1, node2, cavity_n, via_n]])
            else:
                # Append new branch row to the branch table
                branch = np.append(
                    branch,
                    np.array([[branch_n, node1, node2, cavity_n, via_n]]),
                    axis=0
                )

            # increment total branch counter
            branch_n += 1

    # =============================================================================
    '''
    Now we need to identify the port groups that can be merged together.
    A list can be created
    Each group is represented with a 2D matrix

    One more thing can be added: a single via with multi sections can also be merged
    '''
    # =============================================================================

    # layer indexes in stackup for pwr and gnd
    pwr_layers = np.where(stackup == 1)[0]
    gnd_layers = np.where(stackup == 0)[0]

    # Port group node map: [node_id, via_type, layer_idx]
    # find out the nodes that are merged as one group on the top and bottom
    top_pwr_nodes = port_grp_node_num[np.where((port_grp_node_num[:, 1] == 1) # pwr nodes on top layer 
                                               & (port_grp_node_num[:, 2] == 0))[0], 0]
    top_gnd_nodes = port_grp_node_num[np.where((port_grp_node_num[:, 1] == 0) # gnd nodes on top layer
                                                & (port_grp_node_num[:, 2] == 0))[0], 0]
    bot_pwr_nodes = port_grp_node_num[np.where((port_grp_node_num[:, 1] == 1) # pwr nodes on bottom layer
                                               & (port_grp_node_num[:, 2] == stackup.shape[0] - 1))[0], 0]
    bot_gnd_nodes = port_grp_node_num[np.where((port_grp_node_num[:, 1] == 0) # gnd nodes on bottom layer
                                        & (port_grp_node_num[:, 2] == stackup.shape[0] - 1))[0], 0]
    
    # Each 2D matrix describes branches that must be merged electrically
    # Each entry: 2D array (layers × vias) showing which via branches belong to one merged region
    branch_merge_list = []

    for i in range(0, stackup.shape[0] - 1):

        # ============================================================
        # CASE 1 — First cavity (topmost region) where top layer is GND
        # ------------------------------------------------------------
        # According to paper [3] (NVM section), when the top surface
        # is a ground plane, all vias connected through continuous GND
        # layers should be merged into one equivalent “super-network”.
        # ============================================================
        if i == 0 and stackup[i] == 0:

            # ---- Subcase 1A: There is a defined common node on this layer ----
            if layer_com_node[i] != -1:

                # Find vias whose *upper node* is this common GND node
                via2merge_above = branch[np.where(branch[:, 1] == layer_com_node[i])[0], 4]

                # ---- Subcase 1B: Find the next ground layer below (if any) ----
                # This checks for continuity through stacked GND planes.
                if np.where(gnd_layers == i)[0][0] < gnd_layers.shape[0] - 1:
                    next_gnd_idx = gnd_layers[np.where(gnd_layers == i)[0][0] + 1]
                    # Get vias connected to that next GND plane
                    via2merge_below = branch[np.where(branch[:, 1] == layer_com_node[next_gnd_idx])[0], 4]
                else:
                    # No more ground layers below
                    via2merge_below = np.array([])

                # ---- Subcase 1C: Identify vias shared by both layers ----
                # These are the physically continuous ground vias
                via2merge = np.intersect1d(via2merge_above, via2merge_below)

                # ---- Subcase 1D: Merge these vias into one NVM group ----
                if via2merge.shape[0] != 0:
                    # number of cavities between this GND and the next one
                    layer_span = gnd_layers[np.where(gnd_layers == i)[0][0] + 1] - i
                    # create a 2D array for merged branches: (layers × vias)
                    branch2merge = np.ndarray((layer_span, via2merge.shape[0]))

                    # for each cavity between GND layers, find branch indices
                    for c in range(i, gnd_layers[np.where(gnd_layers == i)[0][0] + 1]):
                        branch2merge[c - i, :] = [
                            np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                            for via_num in via2merge.tolist()
                        ]

                    # save the merged group into global list
                    branch_merge_list.append(branch2merge)

                # ============================================================
                # Subcase 1E — Extend merging to the BOTTOM ground plane
                # ------------------------------------------------------------
                # If this is the *last* ground layer in the stack, check whether
                # its vias also connect down to any defined bottom GND port groups.
                # Those continuous paths should also be merged (equipotential).
                # ============================================================
                if np.where(gnd_layers == i)[0][0] == gnd_layers.shape[0] - 1 and bot_gnd_nodes.shape[0] > 0:
                    for n in bot_gnd_nodes.tolist():
                        # Find vias tied to bottom GND node 'n'
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        # Intersect top-side and bottom-side vias to find shared paths
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            # Merge these vias across all intermediate cavities
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))
                            for c in range(i, stackup.shape[0] - 1):
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                    for via_num in via2merge.tolist()
                                ]
                            branch_merge_list.append(branch2merge)

                # ============================================================
                # Subcase 1F — Merge isolated GND vias at the bottom
                # ------------------------------------------------------------
                # Handles single ground vias that reach the bottom layer but
                # don’t belong to a defined bottom port group (bot_port_grp = -1).
                # These still need merging so they are not treated as separate
                # floating networks in the NVM system.
                # ============================================================
                if np.where(gnd_layers == i)[0][0] == gnd_layers.shape[0] - 1 and \
                np.where((branch[:, 3] == stackup.shape[0] - 2) &
                            (via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                            (bot_port_grp[branch[:, 4].astype(int)] == -1))[0].shape[0] > 0:

                    # Get bottom-side nodes belonging to ungrouped GND vias
                    bot_gnd_single_nodes = branch[np.where(
                        (branch[:, 3] == stackup.shape[0] - 2) &
                        (via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                        (bot_port_grp[branch[:, 4].astype(int)] == -1))[0], 2]

                    for n in bot_gnd_single_nodes.tolist():
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))
                            for c in range(i, stackup.shape[0] - 1):
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                    for via_num in via2merge.tolist()
                                ]
                            # Only record nontrivial merges (multiple vias/layers)
                            if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                                branch_merge_list.append(branch2merge)

            # ============================================================
            # Subcase 1G — Merge top PWR port groups
            # ------------------------------------------------------------
            # For the topmost cavity: if any PWR port groups exist on the
            # top layer, check whether their vias continue into the first
            # internal PWR layer below (through decaps or buried vias).
            #
            # These connections represent the start of the power network
            # and must be treated as one equipotential region.
            # ============================================================
            if top_pwr_nodes.shape[0] > 0:
                for n in top_pwr_nodes.tolist():
                    # Vias tied to the top PWR node 'n'
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]
                    # Corresponding vias reaching the next PWR layer below
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[pwr_layers[0]])[0], 4]
                    # Intersection: vias common to both top node and next PWR plane
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        # Stack all via segments between top and first PWR layer
                        branch2merge = np.ndarray((pwr_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, pwr_layers[0]):
                            branch2merge[c - i, :] = [
                                np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                for via_num in via2merge.tolist()
                            ]
                        branch_merge_list.append(branch2merge)

            # ============================================================
            # Subcase 1H — Merge single top PWR vias (not in port groups)
            # ------------------------------------------------------------
            # Some top PWR vias (for example, single decap or blind vias)
            # are not assigned to a specific port group (top_port_grp = -1)
            # but still connect top-side metal to the first PWR layer below.
            #
            # These are merged so that their electrical effect is not
            # duplicated or isolated in the NVM admittance network.
            # ============================================================
            if np.where(
                (branch[:, 3] == 0) &
                (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                (top_port_grp[branch[:, 4].astype(int)] == -1)
            )[0].shape[0] > 0:

                # Find top-layer nodes of ungrouped PWR vias
                top_pwr_single_nodes = branch[np.where(
                    (branch[:, 3] == 0) &
                    (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                    (top_port_grp[branch[:, 4].astype(int)] == -1)
                )[0], 1]

                for n in top_pwr_single_nodes.tolist():
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[pwr_layers[0]])[0], 4]
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        # Build merged via list spanning down to first PWR layer
                        branch2merge = np.ndarray((pwr_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, pwr_layers[0]):
                            branch2merge[c - i, :] = [
                                np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                for via_num in via2merge.tolist()
                            ]
                        # Only keep non-trivial merges
                        if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                            branch_merge_list.append(branch2merge)



        # if the first layer is pwr
        elif i == 0 and stackup[i] == 1:
            # consider the layer common node
            if layer_com_node[i] != -1:
                via2merge_above = branch[np.where(branch[:, 1] == layer_com_node[i])[0], 4]

                # First consider the next pwr layer
                if np.where(pwr_layers == i)[0][0] < pwr_layers.shape[0] - 1:  # if the next pwr layer exists
                    via2merge_below = branch[
                        np.where(branch[:, 1] == layer_com_node[pwr_layers[np.where(pwr_layers == i)[0][0] + 1]])[0], 4]
                else:
                    via2merge_below = np.array([])

                via2merge = np.intersect1d(via2merge_above, via2merge_below)

                if via2merge.shape[0] != 0:
                    branch2merge = np.ndarray((pwr_layers[np.where(pwr_layers == i)[0][0] + 1] - i, via2merge.shape[0]))
                    for c in range(i, pwr_layers[np.where(pwr_layers == i)[0][0] + 1]):
                        branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for
                                                  via_num in via2merge.tolist()]
                    branch_merge_list.append(branch2merge)

                # consider the pwr groups on the bottom plane
                # If the current pwr layer is the last one
                if np.where(pwr_layers == i)[0][0] == pwr_layers.shape[0] - 1 and bot_pwr_nodes.shape[0] > 0:
                    for n in bot_pwr_nodes.tolist():
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))
                            for c in range(i, stackup.shape[0] - 1):
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for via_num in
                                    via2merge.tolist()]
                            branch_merge_list.append(branch2merge)

                # consider single pwr vias on the bottom plane
                # If the current pwr layer is the last one
                # and the pwr vias on the bottom plane is different with the layer type of the last layer
                # and the pwr vias on the bottom plane does not belong to any port grp
                if np.where(pwr_layers == i)[0][0] == pwr_layers.shape[0] - 1 and \
                        np.where((branch[:, 3] == stackup.shape[0] - 2) & (
                                via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                                 (bot_port_grp[branch[:, 4].astype(int)] == -1))[0].shape[0] > 0:
                    bot_pwr_single_nodes = branch[np.where(
                        (branch[:, 3] == stackup.shape[0] - 2) & (via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                        (bot_port_grp[branch[:, 4].astype(int)] == -1))[0], 2]
                    for n in bot_pwr_single_nodes.tolist():
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))

                            if i >= stackup.shape[0] - 1:  # ← 이 줄 수정됨!
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                    for via_num in via2merge.tolist()
                                ]
                                
                                if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                                    branch_merge_list.append(branch2merge)


            # consider the gnd port groups on the top layer
            if top_gnd_nodes.shape[0] > 0:
                for n in top_gnd_nodes.tolist():
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]

                    # consider the next gnd layer
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[gnd_layers[0]])[0], 4]
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        branch2merge = np.ndarray((gnd_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, gnd_layers[0]):
                            branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                                      for via_num in via2merge.tolist()]
                        branch_merge_list.append(branch2merge)

            # consider single gnd vias on the top layer
            if np.where((branch[:, 3] == 0) & (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                        (top_port_grp[branch[:, 4].astype(int)] == -1))[0].shape[0] > 0:
                top_gnd_single_nodes = branch[
                    np.where((branch[:, 3] == 0) & (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                             (top_port_grp[branch[:, 4].astype(int)] == -1))[0], 1]
                for n in top_gnd_single_nodes.tolist():
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]
                    # consider the next gnd layer
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[gnd_layers[0]])[0], 4]
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        branch2merge = np.ndarray((gnd_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, gnd_layers[0]):
                            branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                                      for via_num in via2merge.tolist()]
                        if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                            branch_merge_list.append(branch2merge)


        # if the first layer is either not pwr or gnd. Namely floating layer represented by 2
        elif i == 0 and stackup[i] != 0 and stackup[i] != 1:
            # consider the gnd port groups on the top layer
            if top_gnd_nodes.shape[0] > 0:
                for n in top_gnd_nodes.tolist():
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]

                    # consider the next gnd layer
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[gnd_layers[0]])[0], 4]
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        branch2merge = np.ndarray((gnd_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, gnd_layers[0]):
                            branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                                      for via_num in via2merge.tolist()]
                        branch_merge_list.append(branch2merge)

            # consider the pwr port groups on the top layer
            if top_pwr_nodes.shape[0] > 0:
                for n in top_pwr_nodes.tolist():
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]

                    # consider the next pwr layer
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[pwr_layers[0]])[0], 4]
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        branch2merge = np.ndarray((pwr_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, pwr_layers[0]):
                            branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                                      for via_num in via2merge.tolist()]
                        branch_merge_list.append(branch2merge)

            # consider single pwr vias on the top layer
            if np.where((branch[:, 3] == 0) & (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                        (top_port_grp[branch[:, 4].astype(int)] == -1))[0].shape[0] > 0:
                top_pwr_single_nodes = branch[
                    np.where((branch[:, 3] == 0) & (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                             (top_port_grp[branch[:, 4].astype(int)] == -1))[0], 1]
                for n in top_pwr_single_nodes.tolist():
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]
                    # consider the next pwr layer
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[pwr_layers[0]])[0], 4]
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        branch2merge = np.ndarray((pwr_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, pwr_layers[0]):
                            branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                                      for via_num in via2merge.tolist()]
                        if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                            branch_merge_list.append(branch2merge)

            # consider single gnd vias on the top layer
            if np.where((branch[:, 3] == 0) & (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                        (top_port_grp[branch[:, 4].astype(int)] == -1))[0].shape[0] > 0:
                top_gnd_single_nodes = branch[
                    np.where((branch[:, 3] == 0) & (via_type[branch[:, 4].astype(int)] != stackup[0]) &
                             (top_port_grp[branch[:, 4].astype(int)] == -1))[0], 1]
                for n in top_gnd_single_nodes.tolist():
                    via2merge_above = branch[np.where(branch[:, 1] == n)[0], 4]
                    # consider the next gnd layer
                    via2merge_below = branch[np.where(branch[:, 2] == layer_com_node[gnd_layers[0]])[0], 4]
                    via2merge = np.intersect1d(via2merge_above, via2merge_below)
                    if via2merge.shape[0] != 0:
                        branch2merge = np.ndarray((gnd_layers[0] - i, via2merge.shape[0]))
                        for c in range(i, gnd_layers[0]):
                            branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0]
                                                      for via_num in via2merge.tolist()]
                        if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                            branch_merge_list.append(branch2merge)

        # consider the middle gnd layer
        elif i > 0 and stackup[i] == 0:
            # consider the layer common node
            if layer_com_node[i] != -1:
                via2merge_above = branch[np.where(branch[:, 1] == layer_com_node[i])[0], 4]

                # First consider the next gnd layer
                if np.where(gnd_layers == i)[0][0] < gnd_layers.shape[0] - 1:  # if the next gnd layer exists
                    via2merge_below = branch[
                        np.where(branch[:, 2] == layer_com_node[gnd_layers[np.where(gnd_layers == i)[0][0] + 1]])[0], 4]
                else:
                    via2merge_below = np.array([])

                via2merge = np.intersect1d(via2merge_above, via2merge_below)

                if via2merge.shape[0] != 0:
                    branch2merge = np.ndarray((gnd_layers[np.where(gnd_layers == i)[0][0] + 1] - i, via2merge.shape[0]))
                    for c in range(i, gnd_layers[np.where(gnd_layers == i)[0][0] + 1]):
                        branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for
                                                  via_num in via2merge.tolist()]
                    branch_merge_list.append(branch2merge)

                # consider the gnd groups on the bottom plane
                # If the current gnd layer is the last one
                if np.where(gnd_layers == i)[0][0] == gnd_layers.shape[0] - 1 and bot_gnd_nodes.shape[0] > 0:
                    for n in bot_gnd_nodes.tolist():
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))
                            for c in range(i, stackup.shape[0] - 1):
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for via_num in
                                    via2merge.tolist()]
                            branch_merge_list.append(branch2merge)

                # consider single gnd vias on the bottom plane
                # If the current gnd layer is the last one
                # and the gnd vias on the bottom plane is different with the layer type of the last layer
                # and the gnd vias on the bottom plane does not belong to any port grp
                if np.where(gnd_layers == i)[0][0] == gnd_layers.shape[0] - 1 and \
                        np.where((branch[:, 3] == stackup.shape[0] - 2) & (
                                via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                                 (bot_port_grp[branch[:, 4].astype(int)] == -1))[0].shape[0] > 0:
                    bot_gnd_single_nodes = branch[np.where(
                        (branch[:, 3] == stackup.shape[0] - 2) & (via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                        (bot_port_grp[branch[:, 4].astype(int)] == -1))[0], 2]
                    for n in bot_gnd_single_nodes.tolist():
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))
                            for c in range(i, stackup.shape[0] - 1):
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for via_num in
                                    via2merge.tolist()]
                            if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                                branch_merge_list.append(branch2merge)

        # consider the middle pwr layer
        elif i > 0 and stackup[i] == 1:
            # consider the layer common node
            if layer_com_node[i] != -1:
                via2merge_above = branch[np.where(branch[:, 1] == layer_com_node[i])[0], 4]

                # First consider the next pwr layer
                if np.where(pwr_layers == i)[0][0] < pwr_layers.shape[0] - 1:  # if the next pwr layer exists
                    via2merge_below = branch[
                        np.where(branch[:, 2] == layer_com_node[pwr_layers[np.where(pwr_layers == i)[0][0] + 1]])[0], 4]
                else:
                    via2merge_below = np.array([])

                via2merge = np.intersect1d(via2merge_above, via2merge_below)

                if via2merge.shape[0] != 0:
                    branch2merge = np.ndarray((pwr_layers[np.where(pwr_layers == i)[0][0] + 1] - i, via2merge.shape[0]))
                    for c in range(i, pwr_layers[np.where(pwr_layers == i)[0][0] + 1]):
                        branch2merge[c - i, :] = [np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for
                                                  via_num in via2merge.tolist()]
                    branch_merge_list.append(branch2merge)

                # consider the pwr groups on the bottom plane
                # If the current pwr layer is the last one
                if np.where(pwr_layers == i)[0][0] == pwr_layers.shape[0] - 1 and bot_pwr_nodes.shape[0] > 0:
                    for n in bot_pwr_nodes.tolist():
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))
                            for c in range(i, stackup.shape[0] - 1):
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for via_num in
                                    via2merge.tolist()]
                            branch_merge_list.append(branch2merge)

                # consider single pwr vias on the bottom plane
                # If the current pwr layer is the last one
                # and the pwr vias on the bottom plane is different with the layer type of the last layer
                # and the pwr vias on the bottom plane does not belong to any port grp
                if np.where(pwr_layers == i)[0][0] == pwr_layers.shape[0] - 1 and \
                        np.where((branch[:, 3] == stackup.shape[0] - 2) & (
                                via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                                 (bot_port_grp[branch[:, 4].astype(int)] == -1))[0].shape[0] > 0:
                    bot_pwr_single_nodes = branch[np.where(
                        (branch[:, 3] == stackup.shape[0] - 2) & (via_type[branch[:, 4].astype(int)] != stackup[-1]) &
                        (bot_port_grp[branch[:, 4].astype(int)] == -1))[0], 2]
                    for n in bot_pwr_single_nodes.tolist():
                        via2merge_below = branch[np.where(branch[:, 2] == n)[0], 4]
                        via2merge = np.intersect1d(via2merge_above, via2merge_below)
                        if via2merge.shape[0] != 0:
                            branch2merge = np.ndarray((stackup.shape[0] - 1 - i, via2merge.shape[0]))
                            for c in range(i, stackup.shape[0] - 1):
                                branch2merge[c - i, :] = [
                                    np.where((branch[:, 4] == via_num) & (branch[:, 3] == c))[0][0] for via_num in
                                    via2merge.tolist()]
                            if branch2merge.shape[0] > 1 or branch2merge.shape[1] > 1:
                                branch_merge_list.append(branch2merge)

    return branch, layer_com_node, port_node, port_grp_node_num, branch_merge_list


class PDN():
    def __init__(self):
        self.ic_node_names = np.array([])  # names of IC nodes (nk)
        self.decap_node_names = np.array([])  # names of decap nodes (nk)
        self.top_port_num = np.array([])  # ports on the top layer (nk)
        self.bot_port_num = np.array([])  # on the bottom layer (nk)

        self.stackup = np.array([]) # stackup mask, 1 for pwr, 0 for gnd, 2 for floating layer
        self.die_t = np.array([]) # dielectric layer thickness (m)
        self.er_list = [] # dielectric constant for each layer
        self.d_r = np.array([])  # signal layer thickness (m)
        self.brd = np.array([])

        self.seg_len = np.array([])  # segment length of the boundary

        self.outer_bd_node = np.array([])  # N1*2 matrix, rotate counter-clockwise !!!
        self.inner_bd_node = np.array([])  # N2*2 matrix, rotate clockwise !!!
        self.outer_sxy = np.array([])  # S1*4 matrix, rotate counter-clockwise !!!
        self.inner_sxy = np.array([])  # S2*4 matrix, rotate clockwise !!!
        self.sxy = np.array([])  # S*4 matrix
        self.area = np.array([])  # plane area

        self.bxy = np.array([])

        self.via_r = 0.2e-3

        self.ic_via_xy = np.array([])  # original x,y locations of vias, 2-column matrix
        self.ic_via_type = np.array([])  # original type of vias. 1 for pwr, 0 for gnd
        self.ic_via_loc = np.array([])  # locations for corresponding IC pins, 1 for top layer, 0 for bottom layer

        self.buried_via_xy = np.array([])  # original x,y locations of buried vias, 2-column matrix
        self.buried_via_type = np.array([])  # original type of vias. 1 for pwr, 0 for gnd
        self.blind_via_xy = np.array([])  # original x,y locations of blind vias that are not ic's or decap's, 2-column matrix
        self.blind_via_type = np.array([])  # original type of vias. 1 for pwr, 0 for gnd

        self.decap_via_xy = np.array([])  # x,y locations for decap vias, 2-column matrix
        self.decap_via_type = np.array([])  # type of decap vias, 1 for pwr, 0 for gnd
        self.decap_via_loc = np.array([])  # locations for corresponding decaps, 1 for top layer, 0 for bottom layer

        self.via_xy = np.array([])  # x,y locations of all vias
        self.via_type = np.array([])  # type of all vias. 1 for pwr, 0 for gnd
        self.via_loc = np.array([])  # location of all vias. 1 for top, 0 for bot

        self.start_layers = np.array([])  # start layers of the segments
        self.stop_layers = np.array([])  # stop layers of the segments

        self.decap = []  # matrix of 6 columns: xp, yp, xg, yg, model_num, top_bot ('top' or 'bot')

        self.D = np.array([])  # (S+via_num)*(S+via_num)
        self.Gh = np.array([])  # (S+via_num)*(S+via_num)
        self.L_pul = np.array([])  # via_num*via_num

        self.C_pul = []

        self.R = 0.3  # the worst estimation of distance between two segments

        self.z_orig = np.array([])  # original z parameter, without merging IC ports, without adding decaps
        self.z_mergeIC_no_decap = np.array([])  # z parameter with merging IC ports, without adding decaps
        self.z_mergeIC_with_decap = np.array([])  # z parameter with merging IC ports and added decaps, port #1 is always IC
        
        #change f
        self.fstart = 10e3
        self.fstop = 200e6
        # self.fstop = 1e9  

        self.nf = 201
        self.freq = rf.frequency.Frequency(start=self.fstart / 1e6, stop=self.fstop / 1e6, npoints=self.nf,
                                            unit='mhz', sweep_type='log')

        self.decap_z = np.array([])  # the z matrix forms by all added decaps
        # self.decap_list = self.init_decap_library()

    def init_decap_library(self):
        decap_list = []
        decap_0, _ = short_1port(rf.Network('decap/0_GRM033C80J104KE84.s2p').interpolate(self.freq))
        decap_list.append(decap_0.z)
        decap_1, _ = short_1port(rf.Network('decap/1_GRM033R60J474KE90.s2p').interpolate(self.freq))
        decap_list.append(decap_1.z)
        decap_2, _ = short_1port(rf.Network('decap/2_GRM155B31C105KA12.s2p').interpolate(self.freq))
        decap_list.append(decap_2.z)
        decap_3, _ = short_1port(rf.Network('decap/3_GRM155C70J225KE11.s2p').interpolate(self.freq))
        decap_list.append(decap_3.z)
        decap_4, _ = short_1port(rf.Network('decap/4_GRM185C81A475KE11.s2p').interpolate(self.freq))
        decap_list.append(decap_4.z)
        decap_5, _ = short_1port(rf.Network('decap/5_GRM188R61A106KAAL.s2p').interpolate(self.freq))
        decap_list.append(decap_5.z)
        decap_6, _ = short_1port(rf.Network('decap/6_GRM188B30J226MEA0.s2p').interpolate(self.freq))
        decap_list.append(decap_6.z)
        decap_7, _ = short_1port(rf.Network('decap/7_GRM219D80E476ME44.s2p').interpolate(self.freq))
        decap_list.append(decap_7.z)
        decap_8, _ = short_1port(rf.Network('decap/8_GRM31CR60J227ME11.s2p').interpolate(self.freq))
        decap_list.append(decap_8.z)
        decap_9, _ = short_1port(rf.Network('decap/9_GRM32EC80E337ME05.s2p').interpolate(self.freq))
        decap_list.append(decap_9.z)
        return decap_list

    def init_para(self):

        self.via_xy = deepcopy(self.ic_via_xy)
        self.via_type = deepcopy(self.ic_via_type)
        self.via_loc = np.ones(self.ic_via_type.shape)

    def seg_bd_node(self, bxy, dl):
        # For outer boundary, it must rotate counter-clockwise !!!
        # note that input matrix bxy must go back to the origin point!!!
        # bxy is the boundary coordinate information,N*2 matrix
        # dl is the length of the segments
        # sxy is a S*4 matrix, x1, y1, x2, y2. S - number of segments

        # if dl is smaller than the interval of the bxy points, do linear
        # interpolation

        '''Another way is to use numpy.append here'''

        bxy_old = deepcopy(bxy)
        if bxy_old[-1, 0] != bxy_old[0, 0] or bxy_old[-1, 1] != bxy_old[0, 1]:
            bxy = np.zeros((bxy_old.shape[0] + 1, bxy_old.shape[1]))
            bxy[0:-1, :] = bxy_old
            bxy[-1, :] = bxy_old[0, :]
        # calculate number of segments needed first
        nseg = 0
        for i in range(0, bxy.shape[0] - 1):
            len_ith = sqrt((bxy[i + 1, 0] - bxy[i, 0]) ** 2 + (bxy[i + 1, 1] - bxy[i, 1]) ** 2)
            if dl <= len_ith:
                ne = np.floor(len_ith / dl)
                if (len_ith - ne * dl) > dl * 0.01:
                    nseg += ne + 1
                else:
                    nseg += ne
            else:
                nseg += 1
        nseg = nseg.astype(int)
        sxy = np.ndarray((nseg, 4))
        s = 0
        for i in range(0, bxy.shape[0] - 1):
            len_ith = sqrt((bxy[i + 1, 0] - bxy[i, 0]) ** 2 + (bxy[i + 1, 1] - bxy[i, 1]) ** 2)
            if dl <= len_ith:
                ne = np.floor(len_ith / dl).astype(int)
                for j in range(0, ne):
                    sxy[s, 0] = bxy[i, 0] + j * dl / len_ith * (bxy[i + 1, 0] - bxy[i, 0])
                    sxy[s, 1] = bxy[i, 1] + j * dl / len_ith * (bxy[i + 1, 1] - bxy[i, 1])
                    sxy[s, 2] = bxy[i, 0] + (j + 1) * dl / len_ith * (bxy[i + 1, 0] - bxy[i, 0])
                    sxy[s, 3] = bxy[i, 1] + (j + 1) * dl / len_ith * (bxy[i + 1, 1] - bxy[i, 1])
                    s += 1
                if (len_ith - ne * dl) > dl * 0.01:
                    sxy[s, 0] = bxy[i, 0] + (j + 1) * dl / len_ith * (bxy[i + 1, 0] - bxy[i, 0])
                    sxy[s, 1] = bxy[i, 1] + (j + 1) * dl / len_ith * (bxy[i + 1, 1] - bxy[i, 1])
                    sxy[s, 2] = bxy[i + 1, 0]
                    sxy[s, 3] = bxy[i + 1, 1]
                    s += 1
            else:
                sxy[s, 0] = bxy[i, 0]
                sxy[s, 1] = bxy[i, 1]
                sxy[s, 2] = bxy[i + 1, 0]
                sxy[s, 3] = bxy[i + 1, 1]
                s += 1
        return sxy

    def seg_port(self, x0, y0, r, n=6):
        # Define a function for port segmentation
        # for port boundary, it must rotate clockwise
        # x0, y0 is the port center location, r is radius
        # n is the number of the segments
        # sxy is a S*4 matrix, x1, y1, x2, y2
        dtheta = 2 * pi / n
        n = int(n)
        sxy = np.ndarray((n, 4))
        for i in range(0, n):
            sxy[i, 0] = x0 + r * cos(-(i) * dtheta)
            sxy[i, 1] = y0 + r * sin(-(i) * dtheta)
            sxy[i, 2] = x0 + r * cos(-(i + 1) * dtheta)
            sxy[i, 3] = y0 + r * sin(-(i + 1) * dtheta)
        return sxy

    def seg_bd(self):
        # segment the outer and inner boundary
        # only support one inner boundary (one void)
        e = 8.85e-12
        self.outer_sxy = self.seg_bd_node(self.outer_bd_node, self.seg_len)
        if self.inner_bd_node.size == 0 and self.inner_sxy.size == 0:
            self.sxy = self.outer_sxy
            self.area = np.array([PolyArea(self.sxy[:, 0], self.sxy[:, 1])])
        elif self.inner_bd_node.size == 0:
            self.sxy = np.concatenate((self.outer_sxy, self.inner_sxy))
            self.area = np.array([PolyArea(self.outer_sxy[:, 0], self.outer_sxy[:, 1])
                                  - PolyArea(self.inner_sxy[:, 0], self.inner_sxy[:, 1])])
        else:
            self.inner_sxy = self.seg_bd_node(self.inner_bd_node, self.seg_len)
            self.sxy = np.concatenate((self.outer_sxy, self.inner_sxy))
            self.area = np.array([PolyArea(self.outer_sxy[:, 0], self.outer_sxy[:, 1])
                                  - PolyArea(self.inner_sxy[:, 0], self.inner_sxy[:, 1])])
        #self.C_pul = self.er * e * self.area / 1
        self.C_pul = np.array([er * e * area / 1 for er, area in zip(self.er_list, self.area)])        

    def calc_z_fast(self, res_matrix=None, verbose: bool = False):
        e = 8.85e-12

        # ---- Via lists (IC + DECAP + optional BURIED) ----
        via_xy   = np.concatenate((self.ic_via_xy, self.decap_via_xy, self.buried_via_xy, self.blind_via_xy), axis=0) 
        via_type = np.concatenate((self.ic_via_type, self.decap_via_type, self.buried_via_type, self.blind_via_type), axis=0) # 1 for pwr, 0 for gnd
        via_loc  = np.concatenate((self.ic_via_loc,  self.decap_via_loc),  axis=0) # 1 for top, 0 for bottom

        via_r   = deepcopy(self.via_r) # via radius
        stackup = deepcopy(self.stackup) # stackup mask (1-pwr, 0-gnd)
        sxy     = deepcopy(self.sxy) # segnented outline of each layer
        die_t   = deepcopy(self.die_t) # dielectric thickness

        # ---- Board area / per-cavity area ----
        def get_overlap_area(shape1, shape2):
            poly1 = Polygon(shape1)
            poly2 = Polygon(shape2)
            inter = poly1.intersection(poly2)
            return inter.area if not inter.is_empty else 0.0

        if len(self.sxy_list) == 1:
            # Single-shape: the same cross-section for all cavities
            common_area = PolyArea(self.sxy_list[0][:, 0], self.sxy_list[0][:, 1])
            self.area = np.full(len(die_t), common_area)
        else:
            # Pairwise overlap of consecutive shapes
            self.area = [
                get_overlap_area(self.bxy[i], self.bxy[i + 1])
                for i in range(len(self.bxy) - 1)
            ]

        self.C_pul = np.array([er * e * area for er, area in zip(self.er_list, self.area)])
        C_pul = deepcopy(self.C_pul)

        # ---- Port mapping containers (top/bottom) ----

        # port number assigned to via i if it terminates in the top cavity
        top_port_num = [[-1] for _ in range(via_xy.shape[0])] # -1 = unset
        
        # same idea for the bottom cavity
        bot_port_num = [[-1] for _ in range(via_xy.shape[0])] # -1 = unset

        # group indices for vias that belong to the same logical port group 
        # (used when merging multiple pins together)
        top_port_grp = -1 * np.ones((via_xy.shape[0]), dtype=int)
        bot_port_grp = -1 * np.ones((via_xy.shape[0]), dtype=int)

        # ---- Heuristic mapping (only used if explicit maps are unusable) ----
        port_num = 1
        pwr_count = gnd_count = 0

        # Assume IC is port 0, and all decaps have 2 connections.
        # And all IC ports are connected using via, (not anchored to the plance for ex)
        
        # Top cavity
        for i in range(via_loc.shape[0]):
            if i < self.ic_via_xy.shape[0]:  # IC via → port 0
                top_port_num[i] = [0]
                continue
            if via_loc[i] == 1:
                top_port_num[i] = [port_num]
                if via_type[i] == 1:
                    pwr_count += 1
                else:
                    gnd_count += 1
                if pwr_count == gnd_count and pwr_count > 0:
                    port_num += 1
                    pwr_count = gnd_count = 0

        # Bottom cavity
        pwr_count = gnd_count = 0
        for i in range(via_loc.shape[0]):
            if via_loc[i] == 0:
                bot_port_num[i] = [port_num]
                if via_type[i] == 1:
                    pwr_count += 1
                else:
                    gnd_count += 1
                if pwr_count == gnd_count and pwr_count > 0:
                    port_num += 1
                    pwr_count = gnd_count = 0

        # Anchor IC PWR group on top, if desired
        top_port_grp[np.where(self.ic_via_type == 1)[0]] = 0

        # ---- Inductance per unit length via BEM ----
        sxy   = np.concatenate(deepcopy(self.sxy_list), axis=0)
        via_r = deepcopy(self.via_r)
        L_pul = calc_lpul_bem(via_xy, via_r, sxy)

        # ---- Merge PDN graph (branches/ports/groups from layer start/stop) ----
        branch, layer_com_node, port_node, port_grp_node_num, branch_merge_list = org_merge_pdn(
            stackup, via_type, self.start_layers, self.stop_layers,
            top_port_num, bot_port_num, top_port_grp, bot_port_grp
        )

        # ---- Assemble L for each cavity ----
        L_big = np.zeros((branch.shape[0], branch.shape[0]))

        if len(self.sxy_list) == 1:
            for c in range(stackup.shape[0] - 1):
                idx = np.where(branch[:, 3] == c)[0]
                bi  = branch[idx, 0].astype(int)
                vi  = branch[idx, 4].astype(int)
                L_big[np.ix_(bi, bi)] = die_t[c] * L_pul[np.ix_(vi, vi)]
        else:
            for c in range(stackup.shape[0] - 1):
                sxy_c   = self.sxy_list[c]
                idx_c   = np.where(branch[:, 3] == c)[0]
                bi      = branch[idx_c, 0].astype(int)
                vi      = branch[idx_c, 4].astype(int)
                L_pul_c = calc_lpul_bem(via_xy[vi], via_r, sxy_c)  # only vias in this cavity
                local   = np.arange(len(vi))
                L_big[np.ix_(bi, bi)] = die_t[c] * L_pul_c[np.ix_(local, local)]

        branch_merge_list = []  # (kept, but empty per your code)

        # ---- Merge L and build incidence ----
        L_new_inv, old_branch_nodes, new_branch_nodes, new_old_node_map = merge_L_big(L_big, branch_merge_list, branch)

        freq = self.freq.f
        new_branch_nodes_w_c = deepcopy(new_branch_nodes)

        new_branch_n = new_branch_nodes.shape[0]    # number of branches
        new_node_n   = new_old_node_map.shape[0]    # number of nodes

        # Add inter-layer capacitor branches (one per cavity)
        for c in range(die_t.shape[0]):
            if layer_com_node[c] == -1 and layer_com_node[c + 1] == -1:
                new_branch_nodes_w_c = np.append(
                    new_branch_nodes_w_c,
                    [[new_branch_n, new_node_n, new_node_n + 1]],
                    axis=0
                )
                new_node_n  += 2
                new_branch_n += 1
            elif layer_com_node[c] != -1 and layer_com_node[c + 1] == -1:
                n1 = new_old_node_map[np.where(new_old_node_map[:, 1] == layer_com_node[c])[0][0], 0]
                new_branch_nodes_w_c = np.append(
                    new_branch_nodes_w_c,
                    [[new_branch_n, n1, new_node_n]],
                    axis=0
                )
                new_node_n  += 1
                new_branch_n += 1
            elif layer_com_node[c] == -1 and layer_com_node[c + 1] != -1:
                n2 = new_old_node_map[np.where(new_old_node_map[:, 1] == layer_com_node[c + 1])[0][0], 0]
                new_branch_nodes_w_c = np.append(
                    new_branch_nodes_w_c,
                    [[new_branch_n, new_node_n, n2]],
                    axis=0
                )
                new_node_n  += 1
                new_branch_n += 1
            else:
                n1 = new_old_node_map[np.where(new_old_node_map[:, 1] == layer_com_node[c])[0][0], 0]
                n2 = new_old_node_map[np.where(new_old_node_map[:, 1] == layer_com_node[c + 1])[0][0], 0]
                new_branch_nodes_w_c = np.append(
                    new_branch_nodes_w_c,
                    [[new_branch_n, n1, n2]],
                    axis=0
                )
                new_branch_n += 1

        # ---- Build branch admittance matrix Yb ----
        Yb = np.zeros((freq.shape[0], new_branch_n, new_branch_n), dtype=complex)

        # Inductive branches (inverse L)
        Yb[:, 0:L_new_inv.shape[0], 0:L_new_inv.shape[0]] = (
            1 / (1j * 2 * np.pi) * np.einsum('i,jk->ijk', 1 / freq, L_new_inv)
        )

        # Capacitive (cavity) branches on the diagonal
        min_len = min(len(self.area), len(self.er_list), len(die_t))
        for c in range(min_len):
            area_c = float(self.area[c])
            er_c   = float(self.er_list[c])
            d_c    = float(die_t[c])
            term   = 1j * 2 * np.pi * freq * er_c * e * area_c / d_c  # (nf,)
            Yb[:, L_new_inv.shape[0] + c, L_new_inv.shape[0] + c] = term

        # ---- Reduced incidence (A) and node admittance (Yn) ----
        A = np.zeros((freq.shape[0], new_node_n, new_branch_n))  # reduced incidence matrix
        for b in range(new_branch_n):
            A[:, int(new_branch_nodes_w_c[b, 1]), b] =  1
            A[:, int(new_branch_nodes_w_c[b, 2]), b] = -1

        Yn = np.einsum('rmn,rnk->rmk', np.einsum('rmn,rnk->rmk', A, Yb), np.transpose(A, (0, 2, 1)))

        # ----------------------------------------------------------------------
        # ########################## debug nk ##########################
        if verbose:
            # 1) Which layers are "anchored"?
            print("[PDN DBG] layer_com_node:", layer_com_node.tolist())
            # -1 means: no via terminating on that layer for the relevant net

            # 2) Which extra branches are the inter-layer capacitors attached to?
            cap_branch_rows = list(range(L_new_inv.shape[0], new_branch_n))
            caps_triplets = new_branch_nodes_w_c[cap_branch_rows, :].astype(int).tolist()
            print("[PDN DBG] cap branches (branch_id, node1, node2):", caps_triplets)

            # 3) Old->new node map (the nodes present in the reduced system)
            map2old_node = new_old_node_map[:, 1].astype(int).tolist()
            print("[PDN DBG] map2old_node:", map2old_node)

            # 4) Port mapping sanity
            print("[PDN DBG] port_node (+, -) per port:\n", port_node.astype(int))
            invalid_ports = np.where((port_node[:, 0] < 0) | (port_node[:, 1] < 0))[0]
            if invalid_ports.size:
                print("[PDN ERR] Ports with invalid +/- node indices:", invalid_ports.tolist())

            # 5) Degenerate branches (node1 == node2)
            bad_branches = np.where(new_branch_nodes_w_c[:, 1] == new_branch_nodes_w_c[:, 2])[0]
            if bad_branches.size:
                print("[PDN WARN] Branches with identical endpoints:", bad_branches.astype(int).tolist())

            # 6) Connectivity from reference (will set below, but we can preview using first port's ground)
            ref_node_dbg = int(port_node[0, 1])
            if ref_node_dbg in map2old_node:
                ref_idx = map2old_node.index(ref_node_dbg)
                Nn = int(A.shape[1])
                adj = np.zeros((Nn, Nn), dtype=bool)
                for bb in range(int(new_branch_nodes_w_c.shape[0])):
                    n1 = int(new_branch_nodes_w_c[bb, 1])
                    n2 = int(new_branch_nodes_w_c[bb, 2])
                    if n1 != n2:
                        adj[n1, n2] = True
                        adj[n2, n1] = True
                seen = np.zeros(Nn, dtype=bool)
                stack = [ref_idx]
                while stack:
                    v = stack.pop()
                    if seen[v]: 
                        continue
                    seen[v] = True
                    for u in np.where(adj[v])[0]:
                        if not seen[u]:
                            stack.append(u)
                disconnected = np.where(~seen)[0]
                if disconnected.size:
                    print("[PDN ERR] Disconnected nodes (not reachable from reference):", disconnected.astype(int).tolist())
            else:
                print("[PDN ERR] ref_node not found in map2old_node. ref_node=", ref_node_dbg, "map:", map2old_node)

            # 7) Optional rank check at a few frequencies
            def _rank(M):
                s = np.linalg.svd(M, compute_uv=False)
                return int((s > 1e-12).sum())
            for fi in (0, len(freq)//2, len(freq)-1):
                try:
                    r = _rank(Yn[fi])
                    print(f"PDN [DBG] rank(Yn[{fi}])={r} size={Yn.shape[1]}")
                except Exception as _:
                    pass

            # 8) Per-via table
            # for i_v in range(len(self.start_layers)):
            #     print(f"[VIA {i_v}] type={via_type[i_v]} start={self.start_layers[i_v]} stop={self.stop_layers[i_v]} xy={via_xy[i_v]}")
        # ########################## end debug nk ##########################
        # ----------------------------------------------------------------------

        # ---- Choose reference node: ground of first port ----``
        map2old_node = new_old_node_map[:, 1].astype(int).tolist()
        ref_node = int(port_node[0, 1])

        Yn_reduce = np.delete(
            np.delete(Yn, [map2old_node.index(ref_node)], axis=2),
            [map2old_node.index(ref_node)], axis=1
        )
        del map2old_node[map2old_node.index(ref_node)]

        # ---- Z of reduced network ----
        ztot = np.linalg.inv(Yn_reduce)

        # Shift to port voltages (V+ - V- for each port)
        for n in range(port_node.shape[0]):
            if port_node[n, 1] != ref_node:
                ztot[:, map2old_node.index(port_node[n, 0]), :] -= ztot[:, map2old_node.index(port_node[n, 1]), :]
                ztot[:, :, map2old_node.index(port_node[n, 0])] -= ztot[:, :, map2old_node.index(port_node[n, 1])]

        port_idx = find_index(map2old_node, port_node[:, 0].tolist())
        z = ztot[np.ix_(list(range(ztot.shape[0])), port_idx, port_idx)]

        # ---- Add port series resistance (optional) ----
        if res_matrix is not None:
            n_ports = len(port_idx)
            if np.isscalar(res_matrix):
                res_matrix_ports = np.eye(n_ports) * float(res_matrix)
            else:
                res_matrix_ports = res_matrix[np.ix_(port_idx, port_idx)]
            z += res_matrix_ports[None, :, :]

        return z
