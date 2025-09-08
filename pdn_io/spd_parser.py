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
          [optional] brd.buried_via_xy, brd.buried_via_type
        )
    """
    log = print if verbose else (lambda *args, **kwargs: None)

    text = _read(spd_path)

    # 1) Board shapes -> brd.bxy
    brd.bxy = _extract_board_polygons(text)
    log("\nBoard polygons extracted:\n", brd.bxy)

    # 2) Nodes (coords, type, layer) -> node_info (with both raw+canon keys)
    node_info = _extract_nodes(text)
    log("\nNodes extracted:\n", len(node_info), "examples:\n", list(node_info.items()))

    # 3) .Connect blocks -> ic_blocks, decap_blocks (preserving order)
    ic_blocks, decap_blocks = _extract_connect_blocks(text)
    log("\nIC blocks:\n", len(ic_blocks), "Decap blocks:\n", len(decap_blocks))
    log("IC block example:\n", ic_blocks[0] if ic_blocks else "N/A")

    # 4) IC/Decap vias in Connect order (names + xy + type)
    _fill_ic_decap_vias(brd, node_info, ic_blocks, decap_blocks)
    log("\nIC vias:\n", len(getattr(brd, 'ic_node_names', [])),
        "\nDecap vias:\n", len(getattr(brd, 'decap_node_names', [])))

    # 5) All via pairs (upper/lower nodes), canonicalized
    via_lines = _extract_via_lines(text)
    log("Via lines extracted:\n", len(via_lines), "examples:\n", via_lines[:5])

    # 6) Start/stop/type arrays for ALL vias (IC+DECAP order as in Connect)
    brd.start_layers, brd.stop_layers, brd.via_type = _extract_start_stop_type(
        via_lines, node_info, ic_blocks, decap_blocks
    )
    # convert to 0-based layer indices
    brd.start_layers = brd.start_layers - 1
    brd.stop_layers  = brd.stop_layers  - 1
    log("Start/stop/type arrays:\n", brd.start_layers, brd.stop_layers, brd.via_type)

    # 7) Buried vias (optional)
    _fill_buried_vias(brd, via_lines, node_info)
    if hasattr(brd, 'buried_via_xy'):
        log("Buried vias:\n", len(brd.buried_via_type), "example types:\n", brd.buried_via_type[:5])

    # 8) Via cavity location flags from node layers (top=1, bottom=0)
    _fill_via_locs(brd, node_info)
    log("IC via locs:\n", getattr(brd, 'ic_via_loc', "N/A"))

    # 9) Exact per-via → (port, cavity) map for SPD inputs
    _fill_port_cavity_maps(brd, ic_blocks, decap_blocks)
    log("IC via port/cavity maps:\n", getattr(brd, 'top_port_num', "N/A"))
    log("Decap via port/cavity maps:\n", getattr(brd, 'bot_port_num', "N/A"))

    # 10) Build return tuple compatible with current main.py
    ret = [
        brd.bxy,
        brd.ic_via_xy, brd.ic_via_type,
        brd.start_layers, brd.stop_layers, brd.via_type,
        brd.decap_via_xy, brd.decap_via_type,
    ]
    if getattr(brd, "buried_via_xy", None) is not None and brd.buried_via_xy.size > 0:
        ret += [brd.buried_via_xy, brd.buried_via_type]
    return tuple(ret)


# ---- private helpers (lift logic from your current SPD branch) ---------------


def _read(path: str) -> str:
    """
    Read an SPD file as text (UTF-8). Raises a clear, actionable error if it fails.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Empty SPD path provided to _read().")

    if not os.path.exists(path):
        raise FileNotFoundError(f"SPD file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
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

def _extract_board_polygons(text: str, *, tol: float = 1e-9) -> List[np.ndarray]:
    """
    Parse .Shape blocks referenced by PatchSignal.. assignments and return a list
    of polygons (each as Nx2 array in meters). Boxes are expanded to rectangles.
    If multiple polygons/boxes are identical within `tol`, the list is collapsed
    to a single instance (to match your original main.py behavior).
    """

    import re
    import numpy as np

    # --- Primary extractor: identical to your original main.py ---
    shape_section = re.findall(
        r"\.Shape\s+(ShapeSignal\d+)\n((?:(?:Box|Polygon).*?\n(?:\+.*?\n)?)?)",
        text
    )
    patch_mapping = re.findall(
        r"PatchSignal\d+\s+Shape\s*=\s*(ShapeSignal\d+)\s+Layer\s*=\s*(Signal\d+)",
        text
    )
    sorted_patch_mapping = sorted(
        patch_mapping, key=lambda x: int(x[1].replace("Signal", ""))
    )
    shape_dict = {name: body for name, body in shape_section}

    polygon_shapes: List[np.ndarray] = []
    box_shapes: List[np.ndarray] = []

    for shape_name, _layer in sorted_patch_mapping:
        shape = shape_dict.get(shape_name)
        if shape is None:
            continue

        if "Polygon" in shape:
            poly_coords = re.findall(r"(-?[\d\.eE\+\-]+)\s*mm", shape)
            if len(poly_coords) >= 6 and len(poly_coords) % 2 == 0:
                coords = [(float(poly_coords[i]), float(poly_coords[i + 1]))
                          for i in range(0, len(poly_coords), 2)]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                polygon_shapes.append(np.array(coords, dtype=float) * 1e-3)

        elif "Box" in shape:
            m = re.search(
                r"Box\d*[:\w+\-]*\s+(-?[\d\.eE\+\-]+)\s*mm\s+(-?[\d\.eE\+\-]+)\s*mm\s+([\d\.eE\+\-]+)\s*mm\s+([\d\.eE\+\-]+)\s*mm",
                shape
            )
            if m:
                x0, y0, w, h = map(float, m.groups())
                x1, y1 = x0 + w, y0 + h
                coords = np.array([[x0, y0], [x1, y0], [x1, y1],
                                   [x0, y1], [x0, y0]], dtype=float) * 1e-3
                box_shapes.append(coords)

    # --- Fallback: if PatchSignal mapping or the tight .Shape regex missed things,
    #     grab the full shape body up to the next dot-directive and parse it. ---
    if not polygon_shapes and not box_shapes:
        fallback_sections = re.findall(
            r"\.Shape\s+(ShapeSignal\d+)\s*\n(.*?)(?=\n\.[A-Za-z]|\Z)",
            text, flags=re.DOTALL
        )
        shape_dict_fb = {name: body for name, body in fallback_sections}

        # If we *do* have a mapping, iterate by it; otherwise iterate by appearance
        iter_keys = [k for k, _ in sorted_patch_mapping] if sorted_patch_mapping else list(shape_dict_fb.keys())

        for shape_name in iter_keys:
            body = shape_dict_fb.get(shape_name)
            if not body:
                continue

            # polygons (collect any mm tokens)
            if re.search(r"\bPolygon\b", body, flags=re.IGNORECASE):
                tok = re.findall(r"(-?[\d\.eE\+\-]+)\s*mm", body)
                if len(tok) >= 6 and len(tok) % 2 == 0:
                    pts = [(float(tok[i]), float(tok[i + 1]))
                           for i in range(0, len(tok), 2)]
                    if pts[0] != pts[-1]:
                        pts.append(pts[0])
                    polygon_shapes.append(np.array(pts, dtype=float) * 1e-3)

            # boxes (there can be multiple box lines)
            for m in re.finditer(
                r"Box\d*[:\w+\-]*\s+(-?[\d\.eE\+\-]+)\s*mm\s+(-?[\d\.eE\+\-]+)\s*mm\s+([\d\.eE\+\-]+)\s*mm\s+([\d\.eE\+\-]+)\s*mm",
                body, flags=re.IGNORECASE
            ):
                x0, y0, w, h = map(float, m.groups())
                x1, y1 = x0 + w, y0 + h
                coords = np.array([[x0, y0], [x1, y0], [x1, y1],
                                   [x0, y1], [x0, y0]], dtype=float) * 1e-3
                box_shapes.append(coords)

    # --- Collapse identicals (as in your main.py) ---
    def _collapse(shapes: List[np.ndarray]) -> List[np.ndarray]:
        if not shapes:
            return []
        first = shapes[0]
        all_same = all(
            (s.shape == first.shape) and np.allclose(s, first, atol=tol)
            for s in shapes
        )
        return [first] if all_same else shapes

    if polygon_shapes:
        return _collapse(polygon_shapes)
    if box_shapes:
        return _collapse(box_shapes)
    return []


def _extract_nodes(text: str):
    """
    Build node_info dict with ONLY canonical keys ('Node013' -> 'Node13'):
      node_info['Node13'] = {'type': 1|0, 'x': mm, 'y': mm, 'layer': int}
    Notes:
      - 'type' is 1 for PWR, 0 otherwise (GND/empty)
      - x, y are kept in millimeters (convert to meters at use-site)
      - 'layer' is the integer from 'Signal##'
    """
    pattern = (
        r"(Node\d+)"                  # raw node name
        r"(?:::)?(PWR|GND)?"          # optional ::PWR or ::GND
        r"\s+X\s*=\s*([-\d\.eE\+]+)mm"
        r"\s+Y\s*=\s*([-\d\.eE\+]+)mm"
        r"\s+Layer\s*=\s*Signal(\d+)"
    )

    node_lines = re.findall(pattern, text, flags=re.IGNORECASE)

    node_info = {}
    for raw, typ, x_str, y_str, layer_str in node_lines:
        info = {
            "type": 1 if (typ and typ.lower() == "pwr") else 0,
            "x": float(x_str),       # mm
            "y": float(y_str),       # mm
            "layer": int(layer_str)  # e.g., Signal03 -> 3
        }
        canon = canon_node(raw)      # e.g., "Node013" -> "Node13"
        node_info[canon] = info      # <-- ONLY canonical key

    return node_info


def _extract_connect_blocks(text: str):
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
            r"(?si)\.Connect\s+ic_port[^\n]*\n(.*?)(?:\.EndC\b)",
            component_block,
        )
    ]

    # .Connect decap_portX/cap_portX ... <body> .EndC
    decap_blocks = [
        m.group(1)
        for m in re.finditer(
            r"(?si)\.Connect\s+(?:decap|cap)_port\d*\s+[^\n]*\n(.*?)(?:\.EndC\b)",
            component_block,
        )
    ]

    return ic_blocks, decap_blocks

def _extract_via_lines(text: str):
    """
    Return list of (upper_node, lower_node) names (canonicalized).

    Matches lines like:
      Via12::Through     UpperNode = Node013::PAD  LowerNode = Node7::VIA
    …as well as variants without the extra qualifiers after 'ViaNNN' or 'NodeNNN'.
    """
    # Allow optional qualifiers after 'ViaNNN' and after each Node token.
    # Use \s so Upper/Lower can be on the same or separate lines.
    pattern = (
        r"Via\d+(?:::\w+)?\s+"
        r"UpperNode\s*=\s*(Node\d+)(?:::\w+)?\s+"
        r"LowerNode\s*=\s*(Node\d+)(?:::\w+)?"
    )
    pairs = re.findall(pattern, text, flags=re.IGNORECASE)

    # Canonicalize (e.g., 'Node013' -> 'Node13') and preserve order
    return [(canon_node(u), canon_node(l)) for (u, l) in pairs]

def _extract_start_stop_type(via_lines, node_info, ic_blocks, decap_blocks):
    """
    Reproduce your extract_start_stop_layers_strict_order(...) logic.
    Returns:
        start_layers: np.ndarray[int]   # Signal layer numbers (1-based)
        stop_layers:  np.ndarray[int]   # Signal layer numbers (1-based)
        via_type:     np.ndarray[int]   # 1 if either endpoint is PWR, else 0
    Notes:
        - Canonicalizes node names before searching (e.g., 'Node013' -> 'Node13').
        - Pairs are collected in IC block order first, then decap blocks.
        - Entries whose nodes don't resolve to a known via are skipped.
    """

    def find_via_by_node(node: str):
        """Return (start_layer, stop_layer, upper_name, lower_name) for the first via touching `node`."""
        for upper, lower in via_lines:
            if upper == node or lower == node:
                if (upper in node_info) and (lower in node_info):
                    return (
                        node_info[upper]['layer'],
                        node_info[lower]['layer'],
                        upper,
                        lower
                    )
        return None

    def _nodes_from_block(block: str):
        """Extract ordered plus/minus node name lists (canonicalized) from a .Connect block body."""
        lines = [ln.strip() for ln in block.strip().splitlines()]
        plus_nodes = []
        minus_nodes = []
        for ln in lines:
            if ln.startswith("1") or ln.startswith("2"):
                m = re.search(r"\$Package\.(Node\d+)", ln)
                if not m:
                    continue
                node = canon_node(m.group(1))
                if ln.startswith("1"):
                    plus_nodes.append(node)
                else:
                    minus_nodes.append(node)
        return plus_nodes, minus_nodes

    def process_blocks(blocks):
        results = []
        for blk in blocks:
            plus_nodes, minus_nodes = _nodes_from_block(blk)
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

    ic_entries    = process_blocks(ic_blocks)
    decap_entries = process_blocks(decap_blocks)

    all_entries   = ic_entries + decap_entries
    start_layers  = np.array([e[0] for e in all_entries], dtype=int)
    stop_layers   = np.array([e[1] for e in all_entries], dtype=int)
    via_type      = np.array([e[2] for e in all_entries], dtype=int)

    return start_layers, stop_layers, via_type

def _fill_ic_decap_vias(brd, node_info, ic_blocks, decap_blocks, snap_dec=7):
    """
    Populate:
        brd.ic_node_names, brd.ic_via_xy, brd.ic_via_type
        brd.decap_node_names, brd.decap_via_xy, brd.decap_via_type

    Behavior:
      - Preserves the exact .Connect order used in your main.py.
      - Canonicalizes node names (e.g., 'Node013' -> 'Node13').
      - Uses both '1 ...' (plus) and '2 ...' (minus) lines in sequence.
      - Converts coordinates from mm to meters and rounds to `snap_dec` decimals.
      - Silently skips entries whose node isn't found in node_info.
    """
    import re
    import numpy as np

    def _collect_from_blocks(blocks):
        names_ordered, xy_list, type_list = [], [], []
        for blk in blocks:
            lines = [ln.strip() for ln in blk.strip().splitlines()]
            plus_nodes = []
            minus_nodes = []
            for ln in lines:
                if ln.startswith("1") or ln.startswith("2"):
                    m = re.search(r"\$Package\.(Node\d+)", ln)
                    if not m:
                        continue
                    node = canon_node(m.group(1))
                    if ln.startswith("1"):
                        plus_nodes.append(node)
                    else:
                        minus_nodes.append(node)

            # Keep the same order as your new main: all plus first, then minus
            for n in plus_nodes + minus_nodes:
                if n in node_info:
                    info = node_info[n]
                    names_ordered.append(n)
                    xy_list.append([info["x"] * 1e-3, info["y"] * 1e-3])  # mm -> m
                    type_list.append(info["type"])

        if len(xy_list) == 0:
            xy_arr = np.zeros((0, 2), dtype=float)
            type_arr = np.zeros((0,), dtype=int)
        else:
            xy_arr = np.round(np.array(xy_list, dtype=float), snap_dec)
            type_arr = np.array(type_list, dtype=int)
        return names_ordered, xy_arr, type_arr

    ic_names, ic_xy, ic_type = _collect_from_blocks(ic_blocks)
    decap_names, decap_xy, decap_type = _collect_from_blocks(decap_blocks)

    brd.ic_node_names = ic_names
    brd.ic_via_xy = ic_xy
    brd.ic_via_type = ic_type

    brd.decap_node_names = decap_names
    brd.decap_via_xy = decap_xy
    brd.decap_via_type = decap_type

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
            continue

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

