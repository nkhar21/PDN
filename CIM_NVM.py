import numpy as np
from math import sqrt, pi, log
import time
from itertools import combinations
from utils.geometry import segment_boundary, segment_port

def planesresistance(bxy_b, via_xy_b, via_r, d):
    """
    Compute pairwise **in-plane (sheet)** resistances between via rims on ONE layer,
    using the Contour-Integral Method (CIM).

    Inputs
    ------
    bxy_b     : (Nb,2) polygon of the board outline for this layer.
    via_xy_b  : (Nv,2) real vias + 1 dummy *reference* via appended as the LAST row.
    via_r     : via pad (rim) radius used to draw small circular boundary elements.
    d         : copper thickness of this layer (meters).

    Output
    ------
    rb : (Nv-1, Nv-1) matrix of pairwise spreading resistances between REAL vias
         (the last dummy reference via is removed).
    """

    # --- Physical / meshing params --------------------------------------------
    sigma = 5.8e7          # copper conductivity (S/m)
    dl = 1e-3              # approx. segment length for board outline discretization
    num_via_seg = 6        # number of straight segments per circular via rim

    # --- Build boundary segment list b = [via rims (C′) ; board outline (C)] ---
    # via_sxy: CW arcs around each via center → K segments per via
    via_sxy = np.zeros((via_xy_b.shape[0] * num_via_seg, 4))  # [x1,y1,x2,y2] per segment
    for i in range(via_xy_b.shape[0]):
        via_sxy[i*num_via_seg:(i+1)*num_via_seg, :] = segment_port(via_xy_b[i, 0], via_xy_b[i, 1], via_r, num_via_seg)

    # s_b: CCW segmentation of the outer board boundary
    s_b = segment_boundary(bxy_b, dl)

    # Concatenate: first all *via* segments (C′), then the *outer* boundary segments (C)
    b = np.concatenate((via_sxy, s_b))
    Ntot = b.shape[0]

    # System matrices of CIM (discretized Green’s identity)
    U = np.ndarray((Ntot, Ntot))
    H = np.ndarray((Ntot, Ntot))

    # --- Fill U and H ----------------------------------------------------------
    # Diagonals use the exact self-term; off-diagonals use midpoint/log approximations:
    #   U_mm = 1
    #   H_mm = -(1/(σ π d)) [ ln(wm/2) - 1 ]
    #   U_mn ≈ -(1/π) * w_n * cos(theta)/R
    #   H_mn ≈ -(1/(σ π d)) * ln(R)
    # where w_n is length of segment n, R is distance between segment midpoints,
    # and cos(theta) projects segment-n normal wrt segment-m direction.
    for m in range(Ntot):
        for n in range(Ntot):
            if m == n:
                U[m, n] = 1.0
                wj = sqrt((b[n, 0] - b[n, 2])**2 + (b[n, 1] - b[n, 3])**2)
                H[m, n] = -1.0 / (sigma * pi * d) * (log(wj / 2.0) - 1.0)
            else:
                # Midpoints
                xi0 = (b[m, 0] + b[m, 2]) / 2.0
                yi0 = (b[m, 1] + b[m, 3]) / 2.0
                xj0 = (b[n, 0] + b[n, 2]) / 2.0
                yj0 = (b[n, 1] + b[n, 3]) / 2.0
                # Segment length and midpoint distance
                wj = sqrt((b[n, 0] - b[n, 2])**2 + (b[n, 1] - b[n, 3])**2)
                R  = sqrt((xi0 - xj0)**2 + (yi0 - yj0)**2)
                # Geometric factor (projection)
                cos_theta = ((xj0 - xi0) * (b[n, 3] - b[n, 1]) - (yj0 - yi0) * (b[n, 2] - b[n, 0])) / (R * wj)

                U[m, n] = -1.0 / pi * wj * cos_theta / R              # approx. for off-diagonal U
                H[m, n] = -1.0 / (sigma * pi * d) * log(R)            # approx. for off-diagonal H

    # --- Convert to a segment-level "conductance-like" matrix ------------------
    # D = U^{-1} H ; then G_seg = D^{-1} = (U^{-1} H)^{-1}
    U_H     = np.linalg.inv(U) @ H
    U_H_inv = np.linalg.inv(U_H)    # G_seg (segment space)

    # --- Merge segments that belong to the same via, and lump all outer boundary
    # segments together (equipotential constraints at via rims; boundary is not a port).
    #
    # reduce_list indexes the *first* row/col of each group to be summed:
    #   - for vias: every 'num_via_seg' rows/cols form one group,
    #   - for the outline: all remaining rows/cols are grouped into one block.
    reduce_list = list(range(0, (via_xy_b.shape[0]-1)*num_via_seg + 1, num_via_seg)) \
                + list(range(via_xy_b.shape[0]*num_via_seg, b.shape[0]))

    # Sum rows/cols by group → node-level G (vias + one boundary group)
    U_H_inv_R = np.add.reduceat(np.add.reduceat(U_H_inv, reduce_list, axis=0), reduce_list, axis=1)

    # Convert to a resistance form at node-level: R_tot = (G_node)^{-1}
    R_mat_tot = np.linalg.inv(U_H_inv_R)

    # --- Extract the via-node block and remove the dummy reference -------------
    # The first Nv rows/cols correspond to the Nv via groups (last one is dummy ref).
    R_mat_nodes = R_mat_tot[0:via_xy_b.shape[0], 0:via_xy_b.shape[0]]
    ref_via_num = -1                        # last via = reference

    via_list = list(range(via_xy_b.shape[0]))
    del via_list[ref_via_num]               # keep only REAL vias

    # Gauge-fix: subtract reference row/col so potentials are referenced to the dummy via
    for i in via_list:
        R_mat_nodes[i, :] -= R_mat_nodes[ref_via_num, :]
        R_mat_nodes[:, i] -= R_mat_nodes[:, ref_via_num]

    # Final nodal resistance matrix among REAL vias only
    R = R_mat_nodes[np.ix_(via_list, via_list)]

    # --- Convert nodal R to nodal G, then to pairwise two-terminal rb ----------
    G = np.linalg.inv(R)

    # Build pairwise spreading resistance matrix:
    #   rb[i,i] =  1 / sum_k G[i,k]
    #   rb[i,j] = -1 / G[i,j],  i != j
    rb = np.zeros_like(R)
    for i in range(rb.shape[0]):
        for j in range(rb.shape[1]):
            rb[i, j] = (1.0 / np.sum(G[i, :])) if i == j else (-1.0 / G[i, j])

    return rb


def org_resistance(
    stackup,
    via_type,
    start_layers,
    stop_layers,
    via_xy,
    decap_via_type,
    decap_via_xy,
    decap_via_loc,
    ic_via_xy,
    ic_via_type,
    ic_via_loc,
):
    """
    Build the **resistive branch list** (graph) for the DC network.

    Goal
    ----
    Create all graph edges ("branches") the NVM will solve:
      1) **Vertical branches** along via barrels (through the stack).
      2) **Horizontal branches** between vias on the same metal layer (PWR/GND),
         whose resistances will later be filled by CIM.
      3) **Special branches** for decap PWR-GND links and top-layer stitching.

    Inputs
    ------
    stackup        : (L,) array-like
        Per-layer type mask: 1=PWR, 0=GND, other values are non-conductive.
    via_type       : (N,) array-like
        Net type for every via in `via_xy` (1=PWR, 0=GND). Order must match `via_xy`.
    start_layers   : (N,) array-like
        Starting cavity/layer index of each via's vertical span (inclusive).
    stop_layers    : (N,) array-like
        Ending cavity/layer index of each via's vertical span (inclusive).
    via_xy         : (N,2) array-like
        Via coordinates (meters) in the unified order [IC | decap | buried | blind].
    decap_via_*    : arrays for the **decap subset** (type/xy/location).
        Used to identify/construct decap PWR-GND branches and top-layer stitching.
    ic_via_*       : arrays for the **IC subset** (type/xy/location).
        Used to enforce shared top-layer nodes per net (equipotential pad behavior).

    Output
    ------
    branch : (M, 9) ndarray
        Branch table; each row encodes one edge with columns:
          [0] branch_id     (int)
          [1] node_i        (int)  from-node index
          [2] node_j        (int)  to-node index
          [3] via_i         (int)  via index at node_i (or -1 if none/special)
          [4] via_j         (int)  via index at node_j (or -1 if none/special)
          [5] type          (int)  conductor type: 1=PWR, 0=GND, -1=special (decap/stitch)
          [6] layer         (int)  layer/cavity this branch belongs to
          [7] start_layer   (int)  start of vertical span (for via branches)
          [8] stop_layer    (int)  end   of vertical span (for via branches)

    Notes
    -----
    - Node indices are allocated on-demand; repeated (x,y,layer) reuse the same node.
    - A separate map keeps (x,y,layer) → node_id to ensure node reuse.
    - A list of (layer, node, type, via_idx) is accumulated for later horizontal pairing.
    """

    # --- Containers / counters ------------------------------------------------
    branch = np.zeros((0, 9))   # branch table to be filled and returned
    branch_n = 0                # running branch_id
    node_n = -1                 # last allocated node index; first new node becomes 0
    vertical_nodes = []         # records (layer, node_id, net_type, via_idx) for via nodes
    xy_node_map = {}            # maps (x, y, layer) -> node_id for node reuse

    # Small helper: top-most starting cavity among vias (used to group IC pads)
    top_layer = np.min(start_layers)

    # --- Identify subsets of the unified via list -----------------------------
    # We previously concatenated vias in the order: [IC | decap | buried | blind].
    # Build boolean masks to quickly test membership in the IC/decap blocks.

    is_ic_via = np.zeros(len(via_xy), dtype=bool)
    is_ic_via[:len(ic_via_xy)] = True
    # -> True only for indices belonging to the IC block (prefix of the array).

    is_decap_via = np.zeros(len(via_type), dtype=bool)
    for idx in range(len(decap_via_xy)):
        is_decap_via[idx + len(ic_via_xy)] = True
    # -> True only for indices belonging to the decap block (immediately after IC).

    # For top-layer IC pads we enforce an equipotential node per NET TYPE.
    # This dict will store a *single* node index per {type} (type: 1=PWR, 0=GND)
    # to short together all IC vias of the same net at the top surface.
    shared_top_node_map = {}


    # --- Iterate over every via to create its top node and vertical segments ---
    for via_n in range(len(via_type)):
        current_type = via_type[via_n]  # net type of this via: 1 = PWR, 0 = GND
    
        # Round coordinates to avoid floating-point key mismatches when reusing nodes.
        # (x,y) are in meters and are used together with layer to key nodes.
        x = round(via_xy[via_n][0], 6)
        y = round(via_xy[via_n][1], 6)

        # The via may span multiple cavities/layers from start_cavity to end_cavity.
        # We'll allocate a node at the start, then step layer-by-layer to create
        # vertical branches (via barrel segments) down to end_cavity.
        start_cavity = start_layers[via_n]
        end_cavity   = stop_layers[via_n]

        # Node table key for the node at the via's entry (start) layer.
        key1 = (x, y, start_cavity)


        # Condition 1: IC vias starting on the *top-most* cavity get a shared node
        # Rationale:
        #   - For IC pads on the top layer, all vias of the SAME NET (PWR or GND)
        #     are treated as equipotential (one node), i.e., shorted together.
        #   - This enforces the pad short at DC and reduces node count.
        # Guard:
        #   - Must be an IC via (not a decap via), and must start on 'top_layer'.
        if start_cavity == top_layer and is_ic_via[via_n] and not is_decap_via[via_n]:

            shared_key = (current_type)  # 1=PWR, 0=GND → one shared node per net
            if shared_key in shared_top_node_map:
                # Reuse existing shared node for this net (PWR/GND)
                node1 = shared_top_node_map[shared_key]
            else:
                # Allocate a new shared node for this net on the top layer
                node_n += 1
                node1 = node_n
                shared_top_node_map[shared_key] = node1

            # Map (x,y,layer) → node index and record this vertical-node tuple
            xy_node_map[key1] = node1
            vertical_nodes.append((start_cavity, node1, current_type, via_n))


        # Condition 2: IC vias that start specifically on layer 0 also share the node
        # Why this exists in addition to Condition 1:
        #   - Some datasets define the "top-most" cavity explicitly as layer 0.
        #   - If 'top_layer' != 0 for any reason (e.g., mixed starts), we still want
        #     IC vias that begin at layer 0 to be shorted per net type (PWR/GND).
        # Guard:
        #   - Must be an IC via, not a decap via, and start_cavity == 0.
        elif is_ic_via[via_n] and start_cavity == 0 and not is_decap_via[via_n]:
            shared_key = (current_type)  # net type key → 1=PWR, 0=GND
            if shared_key in shared_top_node_map:
                # Reuse the existing shared top-layer node for this net
                node1 = shared_top_node_map[shared_key]
            else:
                # Allocate a new shared node for this net on layer 0
                node_n += 1
                node1 = node_n
                shared_top_node_map[shared_key] = node1

            # Register the (x,y,layer) → node mapping and record this via-node
            xy_node_map[key1] = node1
            vertical_nodes.append((start_cavity, node1, current_type, via_n))


        # Else: assign an independent node for this via's start location
        # Case: not an IC-top-layer case → do NOT force sharing; each (x,y,layer)
        #       gets its own node so plane/CIM can model distinct potentials.
        else:
            if key1 in xy_node_map:
                # A node at this (x,y,layer) was already created (another via step hit it)
                node1 = xy_node_map[key1]
            else:
                # Allocate a fresh node for this (x,y,layer) and record it
                node_n += 1
                node1 = node_n
                xy_node_map[key1] = node1
                vertical_nodes.append((start_cavity, node1, current_type, via_n))

        # Summary: Assign start-node (x, y, start_layer) for each via.
        # IC vias starting on the top surface share one node per net (PWR/GND).
        # No end nodes or vertical branches yet—those are created in the next “vertical traversal” step.


        # --- Vertical traversal across cavities: build via barrel segments -----
        curr_cavity = start_cavity
        while curr_cavity < end_cavity:
            found_match = False

            # Look ahead to the next cavity boundary where we should place a node.
            # We stop either:
            #   (a) at the final cavity (end_cavity), or
            #   (b) at the next conductive layer matching this via's net (PWR/GND),
            #       so that plane connections can attach at that layer.
            for next_cavity in range(curr_cavity + 1, end_cavity + 1):
                if next_cavity == end_cavity or stackup[next_cavity] == current_type:

                    # Create/reuse the node at (x, y, next_cavity)
                    key2 = (x, y, next_cavity)
                    if key2 in xy_node_map:
                        node2 = xy_node_map[key2]
                    else:
                        node_n += 1
                        node2 = node_n
                        xy_node_map[key2] = node2
                        vertical_nodes.append((next_cavity, node2, current_type, via_n))

                    # Add a vertical branch (via segment) from node1@curr_cavity to node2@next_cavity
                    # Columns: [branch_id, node_i, node_j, via_i, via_j, type, layer, start_layer, stop_layer]
                    new_branch = np.array([[branch_n, node1, node2, via_n, via_n,
                                            current_type, curr_cavity, curr_cavity, next_cavity]])
                    branch = np.vstack([branch, new_branch])
                    branch_n += 1

                    # Advance: the new segment end becomes the start for the next step
                    node1 = node2
                    curr_cavity = next_cavity
                    found_match = True
                    break

            # Safety: if no suitable next_cavity was found (shouldn't happen under
            # normal data), exit to avoid an infinite loop.
            if not found_match:
                print(
                    f"[Warning][CIM org_resistance] No suitable next_cavity found for via #{via_n} "
                    f"at (x={x:.6f}, y={y:.6f}), curr_cavity={curr_cavity}, end_cavity={end_cavity}, "
                    f"type={current_type}. This indicates a dangling via node (no landing copper) "
                    f"on an intermediate layer. Breaking vertical traversal."
                )
                break


    # ========== Add horizontal branches (in-plane spreading connections) =============
    
    # Build a quick lookup: node_id -> via index that created it.
    # (Used to tag each horizontal branch with the corresponding via endpoints.)
    node_to_via_idx = {node: via_idx for (lay, node, vtype, via_idx) in vertical_nodes}

    # Collect all layers that have at least one via node
    unique_layers = sorted({layer for (layer, _, _, _) in vertical_nodes})
    seen_pairs = set()  # avoid duplicate (node_i, node_j) pairs

    for layer in unique_layers:
        layer_type = stackup[layer]
        if layer_type not in [0, 1]:
            # Skip non-conductive layers; only PWR(1) and GND(0) planes host in-plane branches
            continue

        # -- Within this conductive layer: connect same-net nodes pairwise -------
        # Keep only nodes that (a) sit on this layer, and (b) match the layer's net (PWR/GND).
        nodes_on_layer = [
            (node, vtype) for (lay, node, vtype, _) in vertical_nodes
            if lay == layer and vtype == layer_type
        ]
        if len(nodes_on_layer) < 2:
            # Fewer than two eligible nodes → nothing to connect on this layer
            continue

        # Strip to node IDs; these are candidates for in-plane (spreading) branches.
        filtered_nodes = [node for (node, _) in nodes_on_layer]

        # Create unique unordered pairs (ni, nj) with ni != nj
        for (ni, nj) in combinations(filtered_nodes, 2):
            pair = tuple(sorted((ni, nj)))
            if ni == nj or pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Map nodes back to their originating via indices (used later when
            # assigning plane resistances via CIM for this layer).
            vi = node_to_via_idx.get(ni, -1)
            vj = node_to_via_idx.get(nj, -1)

            # Append a horizontal branch: [id, ni, nj, via_i, via_j, type, layer, 0, 0]
            # 'type' is the layer net (1=PWR, 0=GND); start/stop layers unused for horizontals.
            new_branch = np.array([[branch_n, ni, nj, vi, vj, layer_type, layer, 0, 0]])
            branch = np.vstack([branch, new_branch])
            branch_n += 1


    # === Decap PWR–GND pairing (explicit cross-net branches) ==================
    # Decap PWR–GND tie (why we add a branch here)
    # - CIM returns SAME-NET plane spreading resistances only; it does not connect PWR↔GND.
    # - At strict DC, a capacitor is OPEN, so PWR and GND planes would be disconnected and
    #   the nodal system (Yn) becomes singular → no DC resistance can be computed.
    # - We therefore add an explicit branch between each decap’s +/– pads to provide a
    #   RETURN PATH for the DC extraction. Set this branch to a tiny R for a short-circuit
    #   baseline, or to the decap ESR for a more physical low-freq limit.
    # - This models current leaving the IC PWR node, spreading on PWR, returning via the
    #   decap’s GND pad, and flowing on GND back—so copper losses along that loop are included.
    # - If you want “pure copper DC” without component ties, omit this branch (or use a huge R)
    #   and instead add a VRM/short tie; otherwise the DC network is open.

    top_layer = 0                               # index of the first (top) layer
    bottom_layer = stackup.shape[0] - 1         # index of the last (bottom) layer
    used_nodes = set()                          # ensure each decap via is paired only once

    # Outer loop: iterate over decap **PWR** vias (type == 1)
    for pi in range(len(decap_via_type)):
        if decap_via_type[pi] != 1:
            continue
        if pi in used_nodes:
            continue  # already paired with a GND via

        # Inner loop: find a decap **GND** via (type == 0) not yet used
        for gi in range(len(decap_via_type)):
            if decap_via_type[gi] != 0:
                continue
            if gi in used_nodes:
                continue  # skip GND via if already used in another pair

            # Determine which outer surface each decap via lives on:
            # decap_via_loc[*] == 1 → top surface; else → bottom surface.
            layer_pi = top_layer if decap_via_loc[pi] == 1 else bottom_layer
            layer_gi = top_layer if decap_via_loc[gi] == 1 else bottom_layer

            # Look up the pre-created nodes at those (x,y,layer) locations.
            key1 = (round(decap_via_xy[pi][0], 6), round(decap_via_xy[pi][1], 6), layer_pi)
            key2 = (round(decap_via_xy[gi][0], 6), round(decap_via_xy[gi][1], 6), layer_gi)
            
            if key1 not in xy_node_map or key2 not in xy_node_map:
                # Warn and skip if either endpoint node does not exist (e.g., no landing copper)
                missing = []
                if key1 not in xy_node_map:
                    missing.append(f"PWR@L{layer_pi}({decap_via_xy[pi][0]:.6f},{decap_via_xy[pi][1]:.6f})")
                if key2 not in xy_node_map:
                    missing.append(f"GND@L{layer_gi}({decap_via_xy[gi][0]:.6f},{decap_via_xy[gi][1]:.6f})")
                print(f"[Warning][CIM org_resistance] Skipping decap pair pi={pi}, gi={gi}: missing node(s): "
                      + ", ".join(missing))
                continue


            ni = xy_node_map[key1]                  # node for PWR decap via
            nj = xy_node_map[key2]                  # node for GND decap via

            # Tag branch endpoints with their originating via indices (if known)
            vi = node_to_via_idx.get(ni, -1)
            vj = node_to_via_idx.get(nj, -1)

            # Add an explicit cross-net decap branch (type = -1 marks "special/decap") !!!
            # Columns: [id, ni, nj, via_i, via_j, type, layer, start_layer, stop_layer]
            # Note: using layer_pi here (either endpoint's surface would be acceptable).
            new_branch = np.array([[branch_n, ni, nj, vi, vj, -1, layer_pi, 0, 0]])
            branch = np.vstack([branch, new_branch])
            branch_n += 1

            # Prevent reusing these two decap vias in another pair
            used_nodes.update([pi, gi])
            break  # stop searching GND partner for this PWR; move to next PWR
    

    # Top-layer “hub” (temp node) rationale:
    # - Real boards often have very low-R copper tying many top-layer points together (PWR decap pads,
    #   local pours, short traces, nearby GND vias). At DC these behave nearly equipotential.
    # - Instead of modeling a dense web of tiny copper links, we collapse that cluster into ONE
    #   consolidation node (“hub”) and short each participating top-layer node to it (type=-1 branches).
    # - This is ONLY for network assembly (NVM) — CIM still computes same-net plane spreading
    #   resistances between actual via rims. The hub improves conditioning and reduces graph size.
    # - The anchor coordinate is arbitrary; we offset slightly in x to get a unique (x,y,layer) key.
    
    # -- Prepare a top-layer consolidation “temp node” for PWR-side stitching ---
    top_layer = 0
    xy_offset_m = 0.0001  # small x-offset so the temp node doesn't coincide with an existing node

    # 1) Choose an *existing* top-layer PWR node as an anchor.
    #    We scan IC + decap vias (in that combined order). The first one we find
    #    that (a) is PWR (typ==1) and (b) already has a node on layer 0
    #    becomes the reference location. We'll later place the temp node slightly
    #    offset from this coordinate and use its via index (vj_temp) to tag branches.
    for idx, (xy, typ) in enumerate(
        zip(np.vstack([ic_via_xy, decap_via_xy]),
            np.concatenate([ic_via_type, decap_via_type]))
    ):
        node_key = (round(xy[0], 6), round(xy[1], 6), top_layer)
        if typ == 1 and node_key in xy_node_map:
            base_xy = xy       # anchor coordinate near which temp node will be created
            vj_temp = idx      # via index used to label temp-node connections
            break
    else:
        # If no top-layer PWR node exists, we cannot create the consolidation hub.
        raise ValueError("There is no PWR via located on the top layer (layer 0).")


    # 2) Create the “temp node” slightly offset from the anchor PWR node on layer 0.
    #    - Small offset avoids colliding with an existing (x,y,layer) key.
    #    - Allocate a fresh node_id and register it in xy_node_map.
    new_xy = (round(base_xy[0] + xy_offset_m, 9), round(base_xy[1], 9), top_layer)
    temp_node = max(xy_node_map.values()) + 1
    xy_node_map[new_xy] = temp_node

    # Track which nodes we've already stitched to the temp node (avoid duplicates).
    connected_nodes = set()

    # 3) Stitch top-layer decap **PWR-side** nodes to the temp node
    #    Scan existing branches; pick only top-layer **decap branches** (type == -1),
    #    then connect their PWR node (assumed at 'ni') to the consolidation node.
    for b in branch:
        _, ni, nj, vi, vj, typ, lay, _, _ = b
        if lay != top_layer:
            continue                   # only operate on layer 0
        if typ != -1:
            continue                   # only consider special decap branches
        pwr_node = int(ni)             # convention here: 'ni' holds the PWR-side node
        if pwr_node == 0 or pwr_node in connected_nodes:
            continue                   # skip sentinel/ref node 0 and avoid duplicates

        # Add a short/special branch from this PWR node to the temp node
        # Columns: [id, node_i, node_j, via_i, via_j, type, layer, start, stop]
        temp_branch = np.array([[branch_n, pwr_node, temp_node, vi, vj_temp, -1, top_layer, 0, 0]])
        branch = np.vstack([branch, temp_branch])
        branch_n += 1
        connected_nodes.add(pwr_node)


    # 4) Stitch top-layer **GND** vias to the temp node
    #    Iterate over all (IC + decap) vias; pick those that are GND (typ==0) and
    #    have an existing node on layer 0, then connect that node to the temp node.
    for idx, (xy, typ) in enumerate(
        zip(np.vstack([ic_via_xy, decap_via_xy]),
            np.concatenate([ic_via_type, decap_via_type]))
    ):
        if typ != 0:
            continue  # only GND vias
        node_key = (round(xy[0], 6), round(xy[1], 6), top_layer)
        if node_key not in xy_node_map:
            continue  # skip if no landing node at layer 0
        gnd_node = xy_node_map[node_key]
        if gnd_node == 0 or gnd_node in connected_nodes:
            continue  # avoid reference/sentinel or duplicate stitching

        vi = idx  # via index tag for this endpoint
        # Add special (type=-1) branch tying this GND node to the consolidation node
        temp_branch = np.array([[branch_n, gnd_node, temp_node, vi, vj_temp, -1, top_layer, 0, 0]])
        branch = np.vstack([branch, temp_branch])
        branch_n += 1
        connected_nodes.add(gnd_node)

    # Done: all necessary top-layer PWR/GND stitching added; return branch list.
    return branch


def main_res(brd, verbose=False):
    """
    Compute DC impedance using the Contour Integral Method (CIM)
    and the Node Voltage Method (NVM).

    Parameters
    ----------
    brd : PDN
        PDN board object supplying geometry, vias, and stackup data:
          - stackup masks per layer (PWR=1 / GND=0), conductor/dielectric thickness
          - via coordinates/types/locations and start/stop layers
          - board outline polygon(s) and via radius
    """

    # --- Via coordinates and types -------------------------------------------
    # Build a single, flat list of all vias in a **fixed order**:
    #   [ IC vias | decap vias | buried vias | blind vias ]
    # This order is preserved in `via_type` and is relied upon downstream
    # (e.g., for mapping nodes back to via indices and for layer/net filtering).
    #
    # via_xy:  shape (N_total, 2) with (x, y) in meters
    # via_type: shape (N_total,) with net type per via (1=PWR, 0=GND)
    # NOTE: Assumes each component array is present (possibly empty-array, not None).

    via_xy = np.concatenate((brd.ic_via_xy, brd.decap_via_xy, brd.buried_via_xy, brd.blind_via_xy), axis=0)
    via_type = np.concatenate((brd.ic_via_type, brd.decap_via_type, brd.buried_via_type, brd.blind_via_type), axis=0)
    
    # From here on, `via_xy[i]` and `via_type[i]` refer to the same via.
    # Subsequent steps (branch building, plane CIM, and nodal solve) will:
    #   1) create vertical branches for these vias through the stack,
    #   2) create horizontal (in-plane) branches between vias of the same net on each metal layer,
    #   3) assign resistances to branches (via metal + plane spreading via CIM),
    #   4) assemble the nodal system (NVM) to obtain the DC resistance.


    # --- Build branch list (resistive network) ---
    branch = org_resistance(
        stackup=brd.stackup,
        via_type=via_type,
        start_layers=brd.start_layers,
        stop_layers=brd.stop_layers,
        via_xy=via_xy,
        decap_via_type=brd.decap_via_type,
        decap_via_xy=brd.decap_via_xy,
        decap_via_loc=brd.decap_via_loc,
        ic_via_xy=brd.ic_via_xy,
        ic_via_type=brd.ic_via_type,
        ic_via_loc=brd.ic_via_loc,
    )

    # --- Dimensions / helpers --------------------------------------------------
    branch_num = branch.shape[0]                     # total number of branches (edges)
    node_num   = int(np.max(branch[:, [1, 2]]))      # highest node index used (number of nodes) (nodes are 0..node_num)

    # Indices of **vertical** branches (via barrel segments):
    # A branch is vertical if it starts/ends on the same via (via_i == via_j) and is not "special" (-1).
    branch_verti = np.where((branch[:, 3] == branch[:, 4]) & (branch[:, 3] != -1))[0]

    # --- A matrix (node–branch incidence) --------------------------------------
    # A has size (N_nodes) x (N_branches); each column i has +1 at node_i and -1 at node_j of branch i.
    # KCL later: I_nodes = A * I_branches ; and Y_n = A * Y_b * A^T
    A = np.zeros((node_num + 1, branch_num))
    for i in range(branch_num):
        A[int(branch[i, 1]), i] =  1                 # from-node
        A[int(branch[i, 2]), i] = -1                 # to-node
    
    if verbose:
        print(f"[CIM_DC_RES] Constructed A matrix with shape {A.shape}")
        print(f"[CIM_DC_RES] Branch count: {branch_num}, Node count: {node_num}")
        print(f"[CIM_DC_RES] Vertical branch count: {branch_verti.shape[0]}")
        print("reduced incidence matrix (N+1)xB. (hub node included)")
        print(A)



    # --- Initialize branch impedance matrix (Z_b) ------------------------------
    # Vertical via DC resistance uses R = (rho * L) / (pi r^2) = L / (sigma * pi * r^2).
    via_radius = brd.via_r
    sigma      = 5.814e7                              # copper conductivity (S/m)
    via_r      = 1 / (np.pi * sigma * via_radius**2)  # resistance per meter of via barrel (Ω/m)

    # Start Z_b as a tiny diagonal (prevents singularities before we fill real values).
    # Each branch i will eventually have Z_b[i,i] set to:
    #   - via segment R for vertical branches,
    #   - CIM plane-spreading R for horizontal branches,
    #   - very small R (or ESR) for special decap/hub branches.
    zb = np.eye(branch_num) * 1e-5


    # --- Vertical branches (vias): provisional per-branch Z --------------------
    for i in range(branch_num):
        vj = branch[i, 4]

        if vj == -1:
            # Special branches (decap/hub links): keep a tiny placeholder;
            # these are NOT via barrels and will not be length-accumulated below.
            zb[i, i] = 1e-5

        elif i in branch_verti:
            # First-pass seed for via segments using local layer info.
            # NOTE: This is just an initial estimate; the *exact* via resistance
            # (sum of segment lengths × via_r) is computed and overwrites this
            # in the next block (“length accumulation across layers”).
            layer_b = int(branch[i, 6])               # base layer for this segment
            zb[i, i] = brd.die_t[layer_b] + brd.d_r[layer_b] * via_r
            # (Will be replaced by total_length * via_r in the subsequent loop.)

    # --- Refine vertical branch resistances with actual via length -------------
    # For each vertical branch, accumulate the **physical length** the via barrel
    # traverses through the stack (copper foils + dielectric cavities) and set
    # R_via = (total_length) * via_r  where via_r = 1 / (π σ r^2) [Ω/m].
    for i in branch_verti:
        start_lay = int(branch[i, 7])
        stop_lay  = int(branch[i, 8])
        if start_lay > stop_lay:
            start_lay, stop_lay = stop_lay, start_lay  # enforce ascending order

        total = 0.0
        for lay in range(start_lay, stop_lay + 1):
            if 0 <= lay < len(brd.d_r):
                if lay == stop_lay:
                    # At the terminal layer, include only the copper foil thickness.
                    total += brd.d_r[lay]
                elif lay < len(brd.die_t):
                    # Intermediate span: copper foil + dielectric cavity thickness.
                    total += brd.d_r[lay] + brd.die_t[lay]
                else:
                    # If die_t is shorter than d_r, fall back to copper only.
                    total += brd.d_r[lay]
            else:
                print(f"[CIM_NVM Warning] Layer {lay} out of bounds for d/die_t → skipped")

        # Convert length (meters) to resistance (ohms) using Ω/m factor via_r.
        zb[i, i] = total * via_r


    # --- Handle bxy polygons per layer ----------------------------------------
    bxy = brd.bxy
    # If stored as shape (1, N, 2) → unwrap to (N, 2) single polygon
    if isinstance(bxy, np.ndarray) and bxy.ndim == 3 and bxy.shape[0] == 1:
        bxy = bxy[0]
    # If a single (N, 2) polygon is provided, replicate it for every layer so
    # CIM can receive a per-layer outline: bxy[layer] is the polygon for that layer.
    if isinstance(bxy, np.ndarray) and bxy.ndim == 2 and bxy.shape[1] == 2:
        num_layers = len(brd.stackup)
        bxy = [bxy.copy() for _ in range(num_layers)]

    # --- Horizontal (plane) branches ------------------------------------------
    # By construction, vertical branches were appended first; all later branches
    # (indices > last vertical index) are horizontal/special. Select those.
    horizontal_branches = branch[(branch[:, 0] > branch_verti.shape[0] - 1)]
    # Layers that actually host horizontal branches (deduped)
    target_layers = np.unique(horizontal_branches[:, 6].astype(int))

    # --- For each layer that has horizontal branches, fill Z_b via CIM --------
    for i in target_layers:
        # Select all branches that live on this layer
        corres_branches = np.where(branch[:, 6] == i)
        branches_i = branch[corres_branches]

        # Keep only horizontal/special branches (indices after the vertical set)
        branches_i = branches_i[(branches_i[:, 0] > branch_verti.shape[0] - 1)]
        if branches_i.shape[0] == 0:
            continue  # nothing to solve for this layer

        # Collect the *unique via indices* that appear as endpoints on this layer
        combined_values = np.unique(np.concatenate((branches_i[:, 3], branches_i[:, 4])))
        via_xy_b = via_xy[combined_values.astype(int)]   # (Nv, 2) coordinates for CIM ports

        # Per-layer board polygon for CIM (outline of the conductive sheet)
        bxy_b = np.array(bxy[i])

        # Append a tiny-offset “dummy via” near the boundary as **reference node**
        # (planesresistance will remove it internally to fix the DC singularity)
        via_xy_b = np.vstack([via_xy_b, bxy_b[1] + 1e-5])

        # CIM solve on this layer: returns pairwise plane *spreading* resistances
        # between the REAL vias (dummy reference is removed inside the routine).
        # Uses: sigma, d = brd.d_r[i] (sheet thickness) inside planesresistance.
        rb = planesresistance(bxy_b, via_xy_b, via_radius, brd.d_r[i])

        # Map each horizontal branch’s endpoints (via_i, via_j) → indices in rb,
        # and write that resistance on the branch’s diagonal in Z_b.
        for j in range(branches_i.shape[0]):
            idx1 = np.where(combined_values == branches_i[j, 3])[0][0]
            idx2 = np.where(combined_values == branches_i[j, 4])[0][0]
            branch_idx = int(branches_i[j, 0])
            zb[branch_idx, branch_idx] = abs(rb[idx1, idx2])  # guard tiny negatives

    if verbose:
        print(f"[CIM_DC_RES] Completed branch impedance matrix Z_b with shape {zb.shape}")
        print("branch impedance matrix Z_b (formatted, full matrix):")
        np.set_printoptions(precision=3, suppress=False)
        print("zb:\n", zb)

    print_branch_legend(branch, zb, node_num, branch_num, verbose)


    # --- Solve circuit using Node Voltage Method (NVM) -------------------------
    # Build nodal admittance: Y_n = A · Y_b · A^T, where Y_b = Z_b^{-1}.
    at = np.transpose(A)
    yb = np.linalg.inv(zb)        # branch admittances (Z_b is diagonal → cheap & stable)
    yn = np.matmul(A, yb)
    yn1 = np.matmul(yn, at)       # full nodal admittance (includes reference node)

    # Choose a reference node and form the reduced nodal system.
    # Here we use the **highest-index node** (node_num) as the reference by
    # deleting its row/column. This removes the DC gauge (singularity).
    yn2 = np.delete(yn1, node_num, 0)  # drop reference-node row
    yn3 = np.delete(yn2, node_num, 1)  # drop reference-node column

    # Invert reduced admittance to get the reduced nodal impedance matrix.
    zn2 = np.linalg.inv(yn3)

    # Return driving-point resistance seen at node 0 w.r.t. the chosen reference.
    # (Assumes node 0 is the IC PWR node after branch/node construction.)
    
    return zn2[0, 0]


def print_branch_legend(branch, zb, node_num, branch_num, verbose):
    # --- Branch legend (column order == A columns) ----------------------------
    if verbose:
        print("\n[CIM_DC_RES] Branch legend (column order == A columns)")
        hub_id = node_num  # the extra row (index == node_num) is the hub

        def _branch_kind(row):
            typ = int(row[5])           # 1=PWR, 0=GND, -1=special
            s   = int(row[7]); t = int(row[8])
            i   = int(row[1]); j = int(row[2])
            if (j == hub_id) or (i == hub_id):
                return "HUB-TIE"
            if typ in (0, 1) and s != t:
                return "VIA"
            if typ in (0, 1) and s == t:
                return "PLANE"
            return "SPECIAL"

        def _net_name(typ):
            return "PWR" if typ == 1 else ("GND" if typ == 0 else "SPECIAL")

        def _fmt_via(vi, vj):
            return f"({vi},{vj})" if (vi >= 0 and vj >= 0) else "--"

        # Header row (Layer removed)
        header = (
            f"{'Branch':<8} {'Kind':<8} {'Nodes':<9} "
            f"{'Via':<9} {'Net':<8} {'Layer span':<11} {'Z (Ω)':>12}"
        )
        print(header)
        print("-" * len(header))

        kind_counts = {"VIA": 0, "PLANE": 0, "SPECIAL": 0, "HUB-TIE": 0}

        # Iterate strictly in A-column order.
        for k in range(branch_num):
            idx = int(np.where(branch[:, 0] == k)[0][0])  # row with branch_id == k
            row = branch[idx]

            b_id = int(row[0])
            i    = int(row[1]); j = int(row[2])
            vi   = int(row[3]); vj = int(row[4])
            typ  = int(row[5])
            s    = int(row[7]); t   = int(row[8])

            kind = _branch_kind(row)
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            net  = _net_name(typ)
            Zii  = float(zb[b_id, b_id]) if b_id < zb.shape[0] else float('nan')

            # One row (Layer removed)
            print(f"{('b'+str(b_id+1).zfill(2)):<8} {kind:<8} "
                  f"{('n'+str(i+1)+'→'+'n'+str(j+1)):<9} "
                  f"{_fmt_via(vi,vj):<9} {net:<8} "
                  f"{(str(s)+'→'+str(t)):<11} {Zii:12.6e}")

        print(f"[CIM_DC_RES] Branch kind counts: "
              f"VIA={kind_counts.get('VIA',0)}, "
              f"PLANE={kind_counts.get('PLANE',0)}, "
              f"SPECIAL={kind_counts.get('SPECIAL',0)}, "
              f"HUB-TIE={kind_counts.get('HUB-TIE',0)}")
