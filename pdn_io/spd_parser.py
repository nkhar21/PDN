# pdn_io/spd_parser.py
from __future__ import annotations
import re
import numpy as np
from typing import List
import os

# ---- public helpers ---------------------------------------------------------

def canon_node(name: str) -> str:
    """
    Normalize Node labels like 'Node01'/'Node001' → 'Node1'.
    """
    m = re.match(r"Node0*([1-9]\d*)$", name, flags=re.IGNORECASE)
    return f"Node{m.group(1)}" if m else name

def parse_spd(brd, spd_path: str, verbose: bool = False):
    """
    Parse a PowerSI .spd file and populate `brd` with:
      - brd.bxy: list[np.ndarray(N,2)] board polygon(s) in meters
      - brd.ic_node_names, brd.ic_via_xy, brd.ic_via_type, brd.ic_via_loc
      - brd.decap_node_names, brd.decap_via_xy, brd.decap_via_type, brd.decap_via_loc
      - brd.start_layers, brd.stop_layers, brd.via_type   (0-based)
      - brd.buried_via_xy, brd.buried_via_type  (optional, may be empty)
      - brd.top_port_num, brd.bot_port_num  (object arrays of lists)

    Returns (matching your current SPD tuple):
        (
          brd.bxy,
          brd.ic_via_xy, brd.ic_via_type,
          brd.start_layers, brd.stop_layers, brd.via_type,
          brd.decap_via_xy, brd.decap_via_type,
          stackup_mask,
          [optional] brd.buried_via_xy, brd.buried_via_type
        )
    """
    log = print if verbose else (lambda *args, **kwargs: None)

    text = _read(spd_path, verbose=verbose)

    # 1) Board shapes -> brd.bxy
    brd.bxy = _extract_board_polygons(text, debug=verbose)
    log("\n[SPD] Board polygons extracted:\n", brd.bxy)

    # 2) Nodes (coords, type, layer) -> node_info (with both raw+canon keys)
    node_info = _extract_nodes(text)
    log("\n[SPD] Nodes extracted:\n", len(node_info), "examples:\n", list(node_info.items())[:5])

    # 3) .Connect blocks -> ic_blocks, decap_blocks (preserving order)
    #ic_blocks, decap_blocks = _extract_connect_blocks(text)
    ic_blocks, decap_blocks = _extract_port_blocks(text)
    log("\n[SPD] IC blocks:", len(ic_blocks), "Decap blocks:", len(decap_blocks))
    log("\nIC blocks: ", ic_blocks if ic_blocks else "N/A")
    log("\nDecap blocks: ", decap_blocks if decap_blocks else "N/A")

    # 4) IC/Decap vias in Connect order (names + xy + type)
    _fill_ic_decap_vias(brd, node_info, ic_blocks, decap_blocks)
    log("\n[SPD] IC vias:", len(getattr(brd, 'ic_node_names', [])),
        "\nDecap vias:", len(getattr(brd, 'decap_node_names', [])))

    # 5) All via pairs (upper/lower nodes), canonicalized
    via_lines = _extract_via_lines(text)
    log("[SPD] Via lines extracted:\n", len(via_lines), "examples:\n", via_lines[:5])

    # 6) Start/stop/type arrays for ALL vias (IC+DECAP order as in Connect)
    sl, tl, vt = _extract_start_stop_type(via_lines, node_info, ic_blocks, decap_blocks)
    brd.start_layers = np.asarray(sl, np.int32) - 1
    brd.stop_layers  = np.asarray(tl, np.int32) - 1
    brd.via_type     = np.asarray(vt, np.int32)
    log("[SPD] Start/stop/type arrays:\n", brd.start_layers, brd.stop_layers, brd.via_type)
    

    # 7) Buried vias (optional)
    _fill_buried_vias(brd, via_lines, node_info)
    if hasattr(brd, 'buried_via_xy'):
        log("[SPD] Buried vias:", len(brd.buried_via_type))
        log("[SPD] Buried via start layers:", brd.start_layers[-len(brd.buried_via_xy):])
        log("[SPD] Buried via stop layers:", brd.stop_layers[-len(brd.buried_via_xy):])
        log("[SPD] Buried via type layers:", brd.via_type[-len(brd.buried_via_xy):])


    # 8) Via cavity location flags from node layers (top=1, bottom=0)
    _fill_via_locs(brd, node_info)
    log("[SPD] IC via locs:\n", getattr(brd, 'ic_via_loc', "N/A"))
    log("[SPD] Decap via locs:\n", getattr(brd, 'decap_via_loc', "N/A"))

    # 9) Exact per-via → (port, cavity) map for SPD inputs
    _fill_port_cavity_maps(brd, ic_blocks, decap_blocks)
    log("[SPD] IC via port/cavity maps:\n", getattr(brd, 'top_port_num', "N/A"))
    log("[SPD] Decap via port/cavity maps:\n", getattr(brd, 'bot_port_num', "N/A"))

    # --- (3) Assign/snap/cast moved here ---
    SNAP_DEC = 7
    brd.ic_via_xy     = _snap(brd.ic_via_xy, SNAP_DEC)
    brd.decap_via_xy  = _snap(brd.decap_via_xy, SNAP_DEC)
    if getattr(brd, "buried_via_xy", None) is not None and np.size(brd.buried_via_xy):
        brd.buried_via_xy = _snap(brd.buried_via_xy, SNAP_DEC)

    brd.ic_via_type    = np.asarray(brd.ic_via_type,    np.int32)
    brd.decap_via_type = np.asarray(brd.decap_via_type, np.int32)
    if getattr(brd, "buried_via_type", None) is not None:
        brd.buried_via_type = np.asarray(brd.buried_via_type, np.int32)

    # --- (4) Global guards: IC vs Decap vs Buried de-duplication ---
    pre_unique = len(_to_keys(brd.ic_via_xy) | _to_keys(brd.decap_via_xy) | _to_keys(getattr(brd,"buried_via_xy", None)))
    _dedupe_across_groups(brd, eps=1e-7, rdec=9)
    post_unique = len(_to_keys(brd.ic_via_xy) | _to_keys(brd.decap_via_xy) | _to_keys(getattr(brd,"buried_via_xy", None)))
    if post_unique != pre_unique:
        log(f"[SPD] Dedupe adjusted {pre_unique - post_unique} duplicate(s) (eps=1e-7 m)")

    # 10) Infer stackup mask (0=GND-return layer, 1=PWR layer)
    brd.stackup = _infer_stackup_mask(text, node_info=node_info)
    log(f"[SPD] Stackup mask: len={len(brd.stackup)}  PWR={int(np.sum(brd.stackup))}  GND={int(len(brd.stackup)-np.sum(brd.stackup))}")

    # 11) Build return tuple compatible with current main.py
    ret = [
        brd.bxy,
        brd.ic_via_xy, brd.ic_via_type,
        brd.start_layers, brd.stop_layers, brd.via_type,
        brd.decap_via_xy, brd.decap_via_type,
        brd.stackup,
    ]
    if getattr(brd, "buried_via_xy", None) is not None and brd.buried_via_xy.size > 0:
        ret += [brd.buried_via_xy, brd.buried_via_type]
    return tuple(ret)


# ---- private helpers (lift logic from your current SPD branch) ---------------

def _read(path: str, verbose: bool = False) -> str:
    """
    Read an SPD file as text (UTF-8). Raises a clear, actionable error if it fails.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Empty SPD path provided to _read().")

    if not os.path.exists(path):
        raise FileNotFoundError(f"SPD file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            if verbose:
                print(f"[SPD_READ] Successfully read SPD file: {path}")
                print(f"[SPD_READ] File length: {len(text)} characters")
                print(f"[SPD_READ] First 200 chars:\n{text[:200]}")
                print("-------------------------------------------")
            return text
    except UnicodeDecodeError:
        # Fallback: try with 'errors=ignore' but make it explicit in the error
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise OSError(f"Failed to read SPD file (binary/encoding issue): {path}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading SPD file: {path}") from e
    except OSError as e:
        raise OSError(f"Failed to read SPD file '{path}': {e.strerror or e}") from e

def _extract_board_polygons(
    text: str,
    ground_net: str = "gnd",
    power_net: str = "pwr",
    *,
    tol: float = 1e-9,
    debug: bool = False,
) -> List[np.ndarray]:
    """
    Parse .Shape blocks referenced by PatchSignal.. assignments and return a list
    of polygons (each as Nx2 array in meters). Boxes are expanded to rectangles.

    Filtering:
      - If a shape body contains net tags like '::gnd+' or '::pwr1+', only shapes
        whose net matches `ground_net` or `power_net` are kept.
      - If no net tags are present (pure Box/Polygon), keep the shape (backward compatibility).

    Returns
      list[np.ndarray]: polygons/boxes in meters. If multiple identical outlines
      are found, collapse to a single one.
    """

    dprint = (print if debug else (lambda *a, **k: None))

    # --- 1) Capture ALL shape bodies up to .EndShape (robust to + continuations & extra lines like Circle...) ---
    shape_section = re.findall(
        r"\.Shape\s+(\S+)\s*\n(.*?)(?=\.EndShape\b)",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    shape_dict = {name: body for name, body in shape_section}
    dprint(f"[DBG] #shape bodies captured: {len(shape_section)}. Examples: {[n for n,_ in shape_section[:5]]}")

    # --- 2) Map PatchSignal → Shape and keep layer order (SignalXX or Signal$NAME in top→bottom order) ---
    patch_mapping = re.findall(
        r"PatchSignal\S+\s+Shape\s*=\s*(\S+)\s+Layer\s*=\s*(Signal\S+)",
        text
    )

    def _layer_index(sig_name: str) -> int:
        # Try to extract trailing digits (Signal03 → 3). If absent (e.g., Signal$TOP), keep relative order by 0.
        m = re.search(r"(\d+)$", sig_name.replace("Signal", ""))
        return int(m.group(1)) if m else 0

    sorted_patch_mapping = sorted(patch_mapping, key=lambda x: _layer_index(x[1]))
    dprint(f"[DBG] #patch mappings: {len(sorted_patch_mapping)}. First few: {sorted_patch_mapping[:5]}")

    polygon_shapes: List[np.ndarray] = []
    box_shapes: List[np.ndarray] = []

    # --- 3) Iterate shapes referenced by layers; filter by nets, then parse polygon/box coords ---
    for shape_name, layer_name in sorted_patch_mapping:
        body = shape_dict.get(shape_name)
        if body is None:
            dprint(f"[DBG] shape '{shape_name}' missing in shape_dict (skipping).")
            continue

        # Find any net tags like '::gnd+' or '::pwr1-'
        nets = re.findall(r"::([A-Za-z0-9_]+)[\+\-]", body)
        nets_lower = {n.lower() for n in nets}
        keep_shape = (not nets_lower) or (ground_net.lower() in nets_lower) or (power_net.lower() in nets_lower)

        dprint(f"[DBG] shape='{shape_name}' layer='{layer_name}' nets={sorted(nets_lower)} keep={keep_shape}")

        if not keep_shape:
            continue

        # --- Polygons ---
        for poly_block in re.finditer(
            r"Polygon[^\n]*"          
            r"(?:\n\+\s*[^\n]*)*",    
            body,
            flags=re.IGNORECASE
        ):
            # Find net for this specific polygon
            net_match = re.search(r"::([A-Za-z0-9_]+)[\+\-]", poly_block.group(0))
            if net_match:
                net = net_match.group(1).lower()
                if net not in {ground_net.lower(), power_net.lower()}:
                    dprint(f"[DBG]    skipping polygon with net={net}")
                    continue

            mm_tokens = re.findall(r"(-?[\d\.eE\+\-]+)\s*mm", poly_block.group(0))
            if len(mm_tokens) >= 6 and len(mm_tokens) % 2 == 0:
                coords = [(float(mm_tokens[i]), float(mm_tokens[i+1]))
                        for i in range(0, len(mm_tokens), 2)]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                arr = np.array(coords, dtype=float) * 1e-3
                polygon_shapes.append(arr)
                dprint(f"[DBG]  -> polygon points: {arr.shape[0]}")

        
        # --- Boxes ---
        for m in re.finditer(
            r"(Box[0-9A-Za-z:+\-]*)"       # full box token
            r"(?:\:\:([A-Za-z0-9_]+)[\+\-])?" # optional ::net
            r"\s+(-?[\d\.eE\+\-]+)\s*mm"   # x0
            r"\s+(-?[\d\.eE\+\-]+)\s*mm"   # y0
            r"\s+([\d\.eE\+\-]+)\s*mm"     # width
            r"\s+([\d\.eE\+\-]+)\s*mm",    # height
            body, flags=re.IGNORECASE
        ):
            _, net, x0, y0, w, h = m.groups()
            if net:
                net = net.lower()
                if net not in {ground_net.lower(), power_net.lower()}:
                    dprint(f"[DBG]    skipping box with net={net}")
                    continue

            x0, y0, w, h = map(float, (x0, y0, w, h))
            x1, y1 = x0 + w, y0 + h
            arr = np.array(
                [[x0, y0], [x1, y0], [x1, y1],
                [x0, y1], [x0, y0]], dtype=float
            ) * 1e-3
            box_shapes.append(arr)
            dprint(f"[DBG]  -> box parsed (w,h)=({w},{h}) -> 5 pts")


    # --- 4) Collapse identicals (same behavior as before) ---
    def _collapse(shapes: List[np.ndarray]) -> List[np.ndarray]:
        if not shapes:
            return []
        first = shapes[0]
        all_same = all((s.shape == first.shape) and np.allclose(s, first, atol=tol) for s in shapes)
        if all_same:
            dprint("[DBG]  collapsed")
            return [first]
        dprint("[DBG]  shapes returned (not collapsed)")
        return shapes

    if polygon_shapes:
        out = _collapse(polygon_shapes)
        dprint(f"[DBG] polygons kept: {len(out)} (from {len(polygon_shapes)})")
        return out
    if box_shapes:
        out = _collapse(box_shapes)
        dprint(f"[DBG] boxes kept: {len(out)} (from {len(box_shapes)})")
        return out

    dprint("[DBG] no polygons/boxes matched filters.")
    return []




def _extract_nodes(text: str, pwr_net: str = "pwr", gnd_net: str = "gnd") -> dict:
    """
    Build node_info dict with ONLY canonical keys ('Node013' -> 'Node13').

    node_info['Node13'] = {
        'type':  1 for PWR, 0 for GND,
        'net':   raw net tag string (e.g. 'pwr1', 'gnd'),
        'x':     x in mm,
        'y':     y in mm,
        'layer': integer layer number (from 'Signal##')
    }

    Only nodes whose tag exactly matches pwr_net or gnd_net are kept.
    """

    pattern = (
        r"(Node\d+)"                  # raw node name
        r"(?:::)?([A-Za-z0-9_]+)?"    # optional ::NET
        r"\s+X\s*=\s*([-\d\.eE\+]+)mm"
        r"\s+Y\s*=\s*([-\d\.eE\+]+)mm"
        r"\s+Layer\s*=\s*Signal(\d+)"
    )
    node_lines = re.findall(pattern, text, flags=re.IGNORECASE)

    node_info = {}
    for raw, tag, x_str, y_str, layer_str in node_lines:
        tag_l = (tag or "").lower()

        # Only accept nodes that match exactly
        if tag_l == pwr_net.lower():
            node_type = 1
        elif tag_l == gnd_net.lower():
            node_type = 0
        else:
            continue  # skip unrelated nets

        info = {
            "type": node_type,
            "net": tag_l,
            "x": float(x_str),
            "y": float(y_str),
            "layer": int(layer_str),
        }
        node_info[canon_node(raw)] = info

    return node_info

def _extract_connect_blocks(text: str, ic_port_tag: str = "ic_port", decap_port_tag: str = "decap_port") -> tuple:
    """
    Return (ic_blocks, decap_blocks) preserving file order.

    Looks inside the '* Component description lines' section if present; otherwise
    scans the whole file. Each block returned is the inner body between the first
    line after '.Connect ...' and the matching '.EndC'.
    """
    # Try to isolate the component section first (non-greedy up to next '*' line).
    comp_match = re.search(
        r"\*\s*Component description lines\b(.*?)(?:\n\*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    component_block = comp_match.group(1) if comp_match else text

    # .Connect ic_port ... <body> .EndC
    ic_blocks = [
        m.group(1)
        for m in re.finditer(
            rf"(?si)\.Connect\s+{ic_port_tag}[^\n]*\n(.*?)(?:\.EndC\b)",
            component_block,
        )
    ]

    # .Connect decap_portX/cap_portX ... <body> .EndC
    decap_blocks = [
        m.group(1)
        for m in re.finditer(
            rf"(?si)\.Connect\s+(?:{decap_port_tag})\d*\s+[^\n]*\n(.*?)(?:\.EndC\b)",
            component_block,
        )
    ]

    return ic_blocks, decap_blocks


def _extract_port_blocks(
    text: str, ic_port_tag: str = "ic_port", decap_port_tag: str = "decap_port", pwr_net: str = "pwr"
) -> tuple:
    """
    Return (ic_blocks, decap_blocks) preserving file order, based on *Port description lines*.

    Each returned block is the body lines for one Port (PositiveTerminal, NegativeTerminal, etc.).
    Only ports that reference the given pwr_net (e.g. 'pwr', 'pwr1') are returned.
    """

    # Extract the entire .Port ... .EndPort section
    port_match = re.search(
        r"\*+\s*Port description lines\b(.*?)(?:\*+\s*Extraction|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not port_match:
        return [], []

    port_section = port_match.group(1)

    # Split into individual Port definitions
    # Match "PortX_name::tag ..." plus its continuation lines until the next "PortY_" or ".EndPort"
    port_defs = re.findall(
        r"(?mi)^(Port\d+[^\n]*)(?:\n[+].*)*",
        port_section,
    )

    ic_blocks, decap_blocks = [], []

    for match in re.finditer(
        r"(?mi)(Port\d+[^\n]*)(?:\n[+].*)*",
        port_section,
    ):
        port_block = match.group(0)

        # Check if PositiveTerminal contains the requested pwr_net
        if not re.search(rf"\b{pwr_net}\b", port_block, flags=re.IGNORECASE):
            continue

        header_line = match.group(1)

        if re.search(ic_port_tag, header_line, flags=re.IGNORECASE):
            ic_blocks.append(port_block)
        elif re.search(decap_port_tag, header_line, flags=re.IGNORECASE):
            decap_blocks.append(port_block)
        else:
            # fallback: classify by "decap" or "ic" anywhere in block
            if re.search("decap", port_block, flags=re.IGNORECASE):
                decap_blocks.append(port_block)
            elif re.search("ic", port_block, flags=re.IGNORECASE):
                ic_blocks.append(port_block)

    return ic_blocks, decap_blocks


def _extract_via_lines(text: str, pwr_net: str = "pwr", gnd_net: str = "gnd"):
    """
    Return list of (upper_node, lower_node) canonical names, filtered to the requested nets.

    Keeps a via ONLY if:
      - Both UpperNode and LowerNode tags are present and equal, and
      - That tag is EXACTLY one of {pwr_net, gnd_net} (case-insensitive), and
      - If the Via line itself has a tag, it must match the same tag (when present).

    Examples kept (with pwr_net='pwr', gnd_net='gnd'):
      Via..::pwr  UpperNode = Node..::pwr  LowerNode = Node..::pwr
      Via..::GND  UpperNode = Node..::GND  LowerNode = Node..::GND
    Examples dropped:
      Via..::pwr1 ... (nodes ::pwr1)   # not the requested power tag
      Mixed tags between upper/lower.
    """
    # Capture via tag (optional) and node tags (optional), allow newlines between tokens.
    pattern = re.compile(
        r"Via\d+(?:::(?P<vtag>[A-Za-z0-9_]+))?\s+"
        r"UpperNode\s*=\s*(?P<un>Node\d+)(?:::(?P<utag>[A-Za-z0-9_]+))?\s+"
        r"LowerNode\s*=\s*(?P<ln>Node\d+)(?:::(?P<ltag>[A-Za-z0-9_]+))?",
        flags=re.IGNORECASE | re.DOTALL,
    )

    allowed = { (pwr_net or "").lower(), (gnd_net or "").lower() }
    out = []

    for m in pattern.finditer(text):
        un    = m.group("un")
        ln    = m.group("ln")
        vtag  = (m.group("vtag")  or "").lower()
        utag  = (m.group("utag")  or "").lower()
        ltag  = (m.group("ltag")  or "").lower()

        # Must have node tags and they must match each other
        if not utag or not ltag or utag != ltag:
            continue
        # Node tag must be one of the requested nets
        if utag not in allowed:
            continue
        # If via has a tag, it must agree with the node tag
        if vtag and vtag != utag:
            raise ValueError(f"Via tag '{vtag}' disagrees with node tag '{utag}'")

        out.append((canon_node(un), canon_node(ln)))

    return out

def _extract_start_stop_type(via_lines, node_info, ic_blocks, decap_blocks):
    """
    Build (start_layers, stop_layers, via_type) using .Port blocks.

    Inputs
    ------
    via_lines   : list[tuple[str,str]]
        Canonical node pairs for vias, e.g. [('Node1','Node4'), ...]
    node_info   : dict[str, dict]
        Per-node info; keys are canonical names. Must contain:
          - 'layer' : int (Signal## index as int)
          - 'type'  : int (1 for PWR, 0 for GND)  <-- used to derive via_type
    ic_blocks   : list[str]
        Each item is a single-port text block from the .Port section
        containing "... PositiveTerminal ... NegativeTerminal ..." lines.
    decap_blocks: list[str]
        Same as ic_blocks, for decap ports.

    Returns
    -------
    start_layers: list[int]   # 1-based layer indices (SignalN)
    stop_layers : list[int]
    via_type    : list[int]   # 1 if either endpoint node is PWR, else 0
    """

    def find_via_by_node(node_can):
        """Return (start_layer, stop_layer, upper_name, lower_name) for the first via touching node_can."""
        for upper, lower in via_lines:
            if upper == node_can or lower == node_can:
                if (upper in node_info) and (lower in node_info):
                    return (
                        node_info[upper]['layer'],
                        node_info[lower]['layer'],
                        upper,
                        lower,
                    )
        return None

    def _nodes_from_port_block(block_text):
        """
        Parse a .Port block into ordered (pos_nodes, neg_nodes) lists of canonical node names.
        Robust to '+' continuations, mixed case, and multi-line terminals.
        """
        # Capture everything after 'PositiveTerminal' up to the next 'NegativeTerminal' / next 'Port' / '.EndPort' / end
        pos_m = re.search(
            r'(?is)PositiveTerminal\s+(.+?)(?=\n\s*\+\s*NegativeTerminal\b|\n\s*Port\d+|\n\s*\.EndPort\b|$)',
            block_text
        )
        neg_m = re.search(
            r'(?is)NegativeTerminal\s+(.+?)(?=\n\s*\+\s*PositiveTerminal\b|\n\s*Port\d+|\n\s*\.EndPort\b|$)',
            block_text
        )
        pos_chunk = pos_m.group(1) if pos_m else ""
        neg_chunk = neg_m.group(1) if neg_m else ""

        # Find nodes in the order they appear
        pos_raw = re.findall(r'\$Package\.(Node\d+)', pos_chunk, flags=re.IGNORECASE)
        neg_raw = re.findall(r'\$Package\.(Node\d+)', neg_chunk, flags=re.IGNORECASE)

        pos_nodes = [canon_node(n) for n in pos_raw]
        neg_nodes = [canon_node(n) for n in neg_raw]
        return pos_nodes, neg_nodes

    def process_blocks(blocks):
        starts, stops, types = [], [], []
        for blk in blocks:
            pos_nodes, neg_nodes = _nodes_from_port_block(blk)

            # Preserve port order: all positives first, then all negatives
            for node in pos_nodes + neg_nodes:
                if node not in node_info:
                    continue
                hit = find_via_by_node(node)
                if not hit:
                    continue
                s_layer, t_layer, u, l = hit
                starts.append(s_layer)
                stops.append(t_layer)
                # via_type = 1 if either endpoint is a power node, else 0
                vtype = 1 if (node_info[u].get('type', 0) == 1 or node_info[l].get('type', 0) == 1) else 0
                types.append(vtype)
        return starts, stops, types

    ic_s, ic_t, ic_v = process_blocks(ic_blocks)
    dc_s, dc_t, dc_v = process_blocks(decap_blocks)

    start_layers = ic_s + dc_s
    stop_layers  = ic_t + dc_t
    via_type     = ic_v + dc_v

    return start_layers, stop_layers, via_type


def _fill_ic_decap_vias(
    brd,
    node_info: dict,
    ic_blocks: list[str],
    decap_blocks: list[str],
    pwr_net: str = "pwr",
    gnd_net: str = "gnd",
    snap_dec: int = 7,
):
    """
    Works with new .Port blocks:
      PortX_...::net
      + PositiveTerminal $Package.Node... [possibly many, over wrapped '+' lines]
      + NegativeTerminal $Package.Node... [possibly many, over wrapped '+' lines]

    Populates:
        brd.ic_node_names,   brd.ic_via_xy,   brd.ic_via_type
        brd.decap_node_names,brd.decap_via_xy,brd.decap_via_type

    type = 1 if node_info[n]["net"] == pwr_net (case-insensitive)
          = 0 if node_info[n]["net"] == gnd_net
    Other nets are ignored.
    """

    POS_PAT = re.compile(r"^\+?\s*PositiveTerminal\b", re.IGNORECASE)
    NEG_PAT = re.compile(r"^\+?\s*NegativeTerminal\b", re.IGNORECASE)
    NODE_PAT = re.compile(r"\$Package\.(Node\d+)", re.IGNORECASE)

    pwr_net_l = pwr_net.lower()
    gnd_net_l = gnd_net.lower()

    def _collect_from_blocks(blocks: list[str]):
        names_ordered, xy_list, type_list = [], [], []

        for blk in blocks:
            mode = None  # 'pos' or 'neg'
            # IMPORTANT: do NOT strip '+'; only trim trailing newline/spaces
            for raw_ln in blk.splitlines():
                ln = raw_ln.rstrip()

                # Switch mode when we hit a terminal header
                if POS_PAT.search(ln):
                    mode = "pos"
                elif NEG_PAT.search(ln):
                    mode = "neg"

                # Only extract nodes when we are inside a terminal section
                if mode is None:
                    continue

                # Any continuation line that belongs to current terminal can have nodes
                nodes = NODE_PAT.findall(ln)
                if not nodes:
                    continue

                for node_raw in nodes:
                    n = canon_node(node_raw)
                    info = node_info.get(n)
                    if not info:
                        continue

                    net_l = (info.get("net") or "").lower()
                    if net_l == pwr_net_l:
                        t = 1
                    elif net_l == gnd_net_l:
                        t = 0
                    else:
                        # Node is on a different net we don't care about
                        continue

                    names_ordered.append(n)
                    xy_list.append([info["x"] * 1e-3, info["y"] * 1e-3])  # mm → m
                    type_list.append(t)

        if xy_list:
            xy_arr = np.round(np.asarray(xy_list, dtype=float), snap_dec)
            type_arr = np.asarray(type_list, dtype=int)
        else:
            xy_arr = np.zeros((0, 2), dtype=float)
            type_arr = np.zeros((0,), dtype=int)

        return names_ordered, xy_arr, type_arr

    # Parse IC and decap blocks (new .Port format)
    ic_names, ic_xy, ic_type = _collect_from_blocks(ic_blocks)
    decap_names, decap_xy, decap_type = _collect_from_blocks(decap_blocks)

    # Assign to board
    brd.ic_node_names = ic_names
    brd.ic_via_xy = ic_xy
    brd.ic_via_type = ic_type

    brd.decap_node_names = decap_names
    brd.decap_via_xy = decap_xy
    brd.decap_via_type = decap_type

    # Debug
    if ic_type.size:
        print(f"[SPD_MULTI][IC_DBG] total={ic_type.size}, type0={np.sum(ic_type==0)}, type1={np.sum(ic_type==1)}")
    

def _fill_buried_vias(brd, via_lines, node_info):
    """
    Compute brd.buried_via_xy, brd.buried_via_type, and merge their
    start/stop/type into brd.start_layers, brd.stop_layers, brd.via_type.

    Rules (mirrors your main.py behavior):
      - A via is considered *buried* if neither endpoint is on the min or max layer.
      - Midpoint key: ((x_u + x_l)/2, (y_u + y_l)/2) in mm, rounded to 6 decimals.
      - Each unique midpoint produces one buried via with:
          type = 1 if either endpoint is PWR, else 0
          start = min(layer_u, layer_l) - 1   (0-based)
          stop  = max(layer_u, layer_l) - 1   (0-based)
      - Results are sorted by x ascending, then y descending (lexsort on (-y, x)).
      - If any buried vias exist, append their start/stop/type to the existing arrays.
    """
    if not via_lines:
        brd.buried_via_xy = np.array([])
        brd.buried_via_type = np.array([])
        return

    # Determine top/bottom signal layer numbers from all nodes present
    layers = [v['layer'] for v in node_info.values()]
    if not layers:
        brd.buried_via_xy = np.array([])
        brd.buried_via_type = np.array([])
        return

    min_layer = min(layers)
    max_layer = max(layers)

    buried_dict = {}

    for upper, lower in via_lines:
        if upper not in node_info or lower not in node_info:
            raise ValueError("node_info missing entry for via endpoints: {}".format((upper, lower)))

        up = node_info[upper]
        lo = node_info[lower]

        # Skip vias that touch top or bottom signal layer
        if (up['layer'] in (min_layer, max_layer)) or (lo['layer'] in (min_layer, max_layer)):
            continue

        # Midpoint in mm (rounded to 6)
        x_mm = round((up['x'] + lo['x']) / 2.0, 6)
        y_mm = round((up['y'] + lo['y']) / 2.0, 6)
        key = (x_mm, y_mm)

        if key not in buried_dict:
            buried_dict[key] = {
                'type': 1 if (up['type'] == 1 or lo['type'] == 1) else 0,
                'start': min(up['layer'], lo['layer']) - 1,  # 0-based
                'stop':  max(up['layer'], lo['layer']) - 1   # 0-based
            }

    # Build arrays (in meters for XY), sort, and attach to brd
    if buried_dict:
        buried_xy_m   = np.array([[x * 1e-3, y * 1e-3] for (x, y) in buried_dict.keys()], dtype=float)
        buried_type   = np.array([v['type']  for v in buried_dict.values()], dtype=int)
        buried_start  = np.array([v['start'] for v in buried_dict.values()], dtype=int)
        buried_stop   = np.array([v['stop']  for v in buried_dict.values()], dtype=int)

        # Sort by x asc, then y desc (same as lexsort((-y, x)))
        order = np.lexsort((-buried_xy_m[:, 1], buried_xy_m[:, 0]))
        brd.buried_via_xy   = buried_xy_m[order]
        brd.buried_via_type = buried_type[order]
        buried_start        = buried_start[order]
        buried_stop         = buried_stop[order]
    else:
        brd.buried_via_xy   = np.array([])
        brd.buried_via_type = np.array([])
        buried_start        = np.array([], dtype=int)
        buried_stop         = np.array([], dtype=int)

    # Append buried start/stop/type to the existing arrays if any exist
    if buried_start.size > 0 and buried_stop.size > 0:
        brd.start_layers = np.concatenate([np.asarray(brd.start_layers, dtype=int), buried_start])
        brd.stop_layers  = np.concatenate([np.asarray(brd.stop_layers,  dtype=int), buried_stop])
        brd.via_type     = np.concatenate([np.asarray(brd.via_type,     dtype=int), brd.buried_via_type])

def _fill_via_locs(brd, node_info):
    """
    Set brd.ic_via_loc and brd.decap_via_loc using each via's node layer:
      top layer -> 1, bottom layer -> 0.
    For nodes that are on neither extreme, we follow your current behavior and
    treat them as top (1).
    """
    if not getattr(brd, "ic_node_names", None):
        brd.ic_via_loc = np.array([], dtype=int)
    if not getattr(brd, "decap_node_names", None):
        brd.decap_via_loc = np.array([], dtype=int)

    if not node_info:
        # Fallback if node_info is missing; mark everything as top.
        if getattr(brd, "ic_via_xy", None) is not None:
            brd.ic_via_loc = np.ones(len(brd.ic_via_xy), dtype=int)
        if getattr(brd, "decap_via_xy", None) is not None:
            brd.decap_via_loc = np.ones(len(brd.decap_via_xy), dtype=int)
        return

    layers = [v["layer"] for v in node_info.values()]
    top_layer = min(layers) if layers else 1
    bot_layer = max(layers) if layers else top_layer

    # IC locations (by node name)
    ic_locs = []
    for n in getattr(brd, "ic_node_names", []):
        lyr = node_info.get(n, {}).get("layer", top_layer)
        # same logic as in your main.py: if neither top nor bottom, default to top (1)
        ic_locs.append(1 if lyr == top_layer else (0 if lyr == bot_layer else 1))
    brd.ic_via_loc = np.array(ic_locs, dtype=int) if ic_locs else np.array([], dtype=int)

    # Decap locations (by node name)
    dec_locs = []
    for n in getattr(brd, "decap_node_names", []):
        lyr = node_info.get(n, {}).get("layer", top_layer)
        dec_locs.append(1 if lyr == top_layer else (0 if lyr == bot_layer else 1))
    brd.decap_via_loc = np.array(dec_locs, dtype=int) if dec_locs else np.array([], dtype=int)

def _fill_port_cavity_maps(brd, ic_blocks, decap_blocks):
    """
    Build brd.top_port_num / brd.bot_port_num (dtype=object) exactly like your
    working main.py:

      - Total via order in PDN.calc_z_fast is [IC] + [DECAP] + ([BURIED] if any).
      - IC vias map to port 0, assigned to top/bottom by brd.ic_via_loc.
      - Decap ports are 1..M in .Connect order, assigned to cavities by brd.decap_via_loc.

    brd.ic_node_names and brd.decap_node_names must already be populated.
    """
    N_ic   = 0 if getattr(brd, "ic_via_xy", None)     is None else brd.ic_via_xy.shape[0]
    N_dec  = 0 if getattr(brd, "decap_via_xy", None)  is None else brd.decap_via_xy.shape[0]
    N_bury = 0 if not hasattr(brd, "buried_via_xy") or brd.buried_via_xy is None else (
             0 if brd.buried_via_xy.size == 0 else brd.buried_via_xy.shape[0])
    N_total = N_ic + N_dec + N_bury

    top_port_num = [[-1] for _ in range(N_total)]
    bot_port_num = [[-1] for _ in range(N_total)]

    # IC = port 0, assign by cavity using brd.ic_via_loc
    if N_ic:
        for i in range(N_ic):
            if getattr(brd, "ic_via_loc", None) is not None and len(brd.ic_via_loc) > i:
                if int(brd.ic_via_loc[i]) == 1:
                    top_port_num[i] = [0]
                else:
                    bot_port_num[i] = [0]
            else:
                # Fallback to top if loc is missing
                top_port_num[i] = [0]

    # Decaps: ports 1..M in the order of decap_blocks (Connect order).
    # For each block, read the $Package.NodeNNN lines and map those node names
    # back to indices in brd.decap_node_names.
    port_id = 1
    for blk in decap_blocks:
        lines = [ln.strip() for ln in blk.strip().splitlines()]
        plus_nodes  = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1))
                       for ln in lines if ln.startswith("1") and re.search(r"\$Package\.(Node\d+)", ln)]
        minus_nodes = [canon_node(re.search(r"\$Package\.(Node\d+)", ln).group(1))
                       for ln in lines if ln.startswith("2") and re.search(r"\$Package\.(Node\d+)", ln)]

        for n in plus_nodes + minus_nodes:
            if not hasattr(brd, "decap_node_names") or brd.decap_node_names is None:
                continue
            try:
                j = brd.decap_node_names.index(n)  # index within DECAP set
            except ValueError:
                continue  # node not found in decap names (shouldn't happen if SPD is consistent)

            global_idx = N_ic + j  # shift into total via index space
            loc = 1
            if getattr(brd, "decap_via_loc", None) is not None and len(brd.decap_via_loc) > j:
                loc = int(brd.decap_via_loc[j])

            if loc == 1:
                top_port_num[global_idx] = [port_id]
            else:
                bot_port_num[global_idx] = [port_id]

        port_id += 1

    brd.top_port_num = np.array(top_port_num, dtype=object)
    brd.bot_port_num = np.array(bot_port_num, dtype=object)


def _infer_stackup_mask(text: str, node_info=None) -> np.ndarray:
    """
    Return an int mask over Signal layers:
       0 -> GND/return layer, 1 -> PWR layer
    Preferred: use PatchSignal->Shape mapping, scanning shape body for '::pwr+' or '::gnd+'.
    Fallback: if shapes are unavailable, mark a layer as PWR if any node on that layer is type==1.
    """
    # map '.Shape <name> ... .EndShape' bodies
    shape_blocks = dict(re.findall(r"(?si)\.Shape\s+(\S+)\s*\n(.*?)(?:\.EndShape\b)", text))
    # map 'PatchSignalN Shape=<name> Layer=SignalNN' in file order
    patch = re.findall(r"(?mi)PatchSignal\d+\s+Shape\s*=\s*(\S+)\s+Layer\s*=\s*(Signal\d+)", text)

    layer_to_type = {}
    max_sig = 0
    for shape_name, sig in patch:
        try:
            idx = int(re.sub(r"\D", "", sig))
        except Exception:
            continue
        body = shape_blocks.get(shape_name, "")
        s = body.lower()
        has_pwr = bool(re.search(r"::\s*pwr\w*\+", s))
        has_gnd = bool(re.search(r"::\s*gnd\w*\+", s))
        # Decide 0/1; if mixed or ambiguous, prefer PWR (you can warn here if you like)
        if has_pwr and not has_gnd:
            v = 1
        elif has_gnd and not has_pwr:
            v = 0
        elif has_pwr and has_gnd:
            v = 1  # mixed pour (rare for your current boards) -> treat as PWR for mask
        else:
            v = 0  # default to GND if no explicit tag is found
        layer_to_type[idx] = v
        max_sig = max(max_sig, idx)

    # Fallback using node_info if we found nothing in shapes
    if not layer_to_type:
        if node_info is None:
            node_info = _extract_nodes(text)
        for _, inf in node_info.items():
            idx = int(inf["layer"])
            layer_to_type[idx] = max(layer_to_type.get(idx, 0), int(inf["type"]))
            max_sig = max(max_sig, idx)

    # Build dense 1..max_sig mask (1-based -> 0-based array)
    mask = np.zeros(max_sig, dtype=int)
    for i in range(1, max_sig + 1):
        mask[i - 1] = int(layer_to_type.get(i, 0))
    
    return mask

# --- snapping & de-dup helpers ---

def _snap(arr, decimals=7):
    if arr is None or np.size(arr) == 0:
        return arr
    return np.round(np.asarray(arr, dtype=float), decimals)

def _dedupe_group_inplace(arr, eps=1e-7, rdec=9):
    """Make all points in arr unique within the group by nudging duplicates."""
    if arr is None or np.size(arr) == 0:
        return
    a = np.asarray(arr, float)
    seen = set()
    for i in range(len(a)):
        x, y = float(a[i, 0]), float(a[i, 1])
        key = (round(x, rdec), round(y, rdec))
        while key in seen:
            x += eps; y += eps
            key = (round(x, rdec), round(y, rdec))
        seen.add(key)
        a[i, 0], a[i, 1] = x, y
    arr[:] = a  # in place

def _to_keys(arr, rdec=9):
    """Turn an (N,2) array-like into a set of rounded (x,y) tuples (array-safe)."""
    if arr is None:
        return set()
    a = np.asarray(arr, float)
    if a.size == 0:
        return set()
    a = a.reshape(-1, 2)
    return {(round(float(x), rdec), round(float(y), rdec)) for x, y in a}

def _dedupe_across_groups(brd, eps=1e-7, rdec=9):
    """
    Priority: IC -> Decap -> Buried.
    Later groups nudge to avoid collisions with the union of earlier groups.
    """
    _dedupe_group_inplace(brd.ic_via_xy,    eps=eps, rdec=rdec)
    _dedupe_group_inplace(brd.decap_via_xy, eps=eps, rdec=rdec)
    if getattr(brd, "buried_via_xy", None) is not None and np.size(brd.buried_via_xy):
        _dedupe_group_inplace(brd.buried_via_xy, eps=eps, rdec=rdec)

    seen = _to_keys(brd.ic_via_xy, rdec=rdec)

    if brd.decap_via_xy is not None and np.size(brd.decap_via_xy):
        for i in range(len(brd.decap_via_xy)):
            x, y = float(brd.decap_via_xy[i][0]), float(brd.decap_via_xy[i][1])
            key = (round(x, rdec), round(y, rdec))
            while key in seen:
                x += eps; y += eps
                key = (round(x, rdec), round(y, rdec))
            seen.add(key)
            brd.decap_via_xy[i] = [x, y]

    if getattr(brd, "buried_via_xy", None) is not None and np.size(brd.buried_via_xy):
        for i in range(len(brd.buried_via_xy)):
            x, y = float(brd.buried_via_xy[i][0]), float(brd.buried_via_xy[i][1])
            key = (round(x, rdec), round(y, rdec))
            while key in seen:
                x += eps; y += eps
                key = (round(x, rdec), round(y, rdec))
            seen.add(key)
            brd.buried_via_xy[i] = [x, y]
