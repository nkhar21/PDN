# pdn_io/spd_parser_multi.py
from __future__ import annotations
import re
import numpy as np
from typing import List, Tuple, Dict, Iterable, Optional

# Reuse your existing helpers
from . import spd_parser
from code_pdn_AH import PDN


# -----------------------------
# Low-level extractors (multi-net aware)
# -----------------------------

_NET_WHITELIST = {"gnd"} | {f"pwr{i}" for i in range(0, 6)}  # pwr, pwr1..pwr5 (pwr0==pwr)


def _extract_nodes_multinet(text: str, target_net: str) -> Dict[str, Dict]:
    """
    Extract nodes with explicit net name (gnd, pwr, pwr1..pwr5), keeping
    the fields your existing helpers expect: type (0=gnd,1=pwr*), x,y (mm),
    and numeric 'layer' (SignalN -> N).
    """
    # Node lines sometimes omit "::net" (e.g., only Contact=...), so make net optional.
    # Example:
    #   Node0207634::pwr2 X = 2.758920e+01mm Y = ... Layer = Signal1
    pat = (
        r"(Node\d+)"                # 1: Node name
        r"(?:::)?([A-Za-z0-9_]+)?"  # 2: optional net token
        r"\s+X\s*=\s*([-\d\.eE\+]+)mm"
        r"\s+Y\s*=\s*([-\d\.eE\+]+)mm"
        r"\s+Layer\s*=\s*Signal(\d+)"
    )
    out: Dict[str, Dict] = {}
    for node, net, xs, ys, ls in re.findall(pat, text, flags=re.IGNORECASE):
        net_l = (net or "").lower()

        # Normalize special cases
        if net_l == "pwr0":
            net_l = "pwr"

        # If it looks bogus (like "0" from Contact=0), drop it
        if net_l not in _NET_WHITELIST:
            net_l = None  # let fallback handle it

        # --- NEW: fallback assignment ---
        if net_l is None or net_l == "":
            net_l = target_net  # << pass current board net down here

        ntype = 0 if net_l == "gnd" else 1
        out[node] = {
            "net": net_l,
            "type": ntype,
            "x": float(xs),
            "y": float(ys),
            "layer": int(ls),
        }
    return out


def _shape_blocks(text: str) -> Dict[str, str]:
    """
    Return dict of {shape_name: body} for all .Shape ... .EndShape blocks.
    (Works for both 'ShapeSignalNN' and 'Signal$L...pkgshape'.)
    """
    return dict(re.findall(r"(?si)\.Shape\s+(\S+)\s*\n(.*?)(?:\.EndShape\b)", text))


def _patch_map(text: str) -> Dict[str, int]:
    """
    Map 'PatchSignalN Shape = <shape_name> Layer = SignalN' to {shape_name: N}.
    Handles both old and new SPD styles.
    """
    mapping: Dict[str, int] = {}
    # Examples:
    #   PatchSignal05 Shape = ShapeSignal05 Layer = Signal05
    #   PatchSignal5  Shape = Signal$L5pkgshape Layer = Signal5
    pat = r"(?mi)^\s*PatchSignal(\d+)\s+Shape\s*=\s*(\S+)\s+Layer\s*=\s*Signal(\d+)"
    for psn, shape, sig in re.findall(pat, text):
        try:
            n1, n2 = int(psn), int(sig)
            if n1 == n2:
                mapping[shape] = n2
        except Exception:
            continue
    return mapping


def _nets_in_shape_body(body: str) -> Dict[str, int]:
    """
    Scan a shape body and count how many polygons/boxes belong to each net
    via tags like '::gnd+' or '::pwr3+'.
    Returns {net_name: count}.
    """
    nets: Dict[str, int] = {}
    for m in re.findall(r"::([A-Za-z0-9_]+)\+", body, flags=re.IGNORECASE):
        n = m.lower().replace("pwr0", "pwr")
        nets[n] = nets.get(n, 0) + 1
    return nets


def _layer_mask_for_net(
    shape_bodies: Dict[str, str],
    patchmap: Dict[str, int],
    target_net: str
) -> np.ndarray:
    """
    Build a stackup mask per net:
      1 if layer has any polygons for 'target_net'
      0 if layer has only gnd polygons
    If both appear on same layer, prefer 1 (we're analyzing the PWR view).
    If a layer has neither (for this view), it is left at 0.
    """
    max_sig = max(patchmap.values()) if patchmap else 0
    mask = np.zeros(max_sig, dtype=int) if max_sig else np.array([], dtype=int)
    if max_sig == 0:
        return mask

    for shape_name, body in shape_bodies.items():
        layer = patchmap.get(shape_name)
        if not layer:
            continue
        nets = _nets_in_shape_body(body)
        if target_net in nets:
            mask[layer - 1] = 1
        elif ("gnd" in nets) and (mask[layer - 1] == 0):
            mask[layer - 1] = 0
        # If both target and gnd exist, the 'if target_net' above already set 1.

    return mask


def _connect_ic_blocks_for_net(text: str, target_net: str) -> List[List[str]]:
    """
    Parse '.Connect IC ...' blocks (new SPD) and generate ic_blocks compatible
    with your _fill_ic_decap_vias helper:
      Each block is represented as a list of lines like:
        '1 $Package.NodeXXXXX::pwr2'
        '2 $Package.NodeYYYYY::GND'
    We keep only '1' lines whose net==target_net and '2' lines whose net==gnd.
    """
    ic_blocks: List[List[str]] = []

    # Grab each .Connect ... .EndC block
    for head, body in re.findall(r"(?si)^\s*\.Connect\s+([^\n]*?)\n(.*?)^\s*\.EndC\s*$", text, flags=re.MULTILINE):
        if "ic" not in head.lower():
            continue
        pos_lines: List[str] = []
        neg_lines: List[str] = []
        # Lines look like: "1 $Package.Node0207634::pwr2"
        for side, node, net in re.findall(
            r"(?mi)^\s*([12])\s+\$Package\.(Node[0-9]+)::([A-Za-z0-9_]+)", body
        ):  
            net_l = net.lower().replace("pwr0", "pwr")
            if side == "1":
                # Normalize pwr0 → pwr
                if net_l == "pwr0":
                    net_l = "pwr"
                if net_l == target_net:
                    pos_lines.append(f"1 $Package.{node}::{net_l}")
            elif side == "2":
                if net_l == "gnd":
                    neg_lines.append(f"2 $Package.{node}::gnd")
        # Reparse robustly:
        for line in body.splitlines():
            m = re.search(r"^\s*([12])\s+\$Package\.(Node[0-9]+)::([A-Za-z0-9_]+)", line, flags=re.IGNORECASE)
            if not m:
                continue
            side, node, net = m.groups()
            net_l = net.lower().replace("pwr0", "pwr")
            if side == "1" and net_l == target_net:
                pos_lines.append(f"1 $Package.{node}::{net_l}")
            elif side == "2" and net_l == "gnd":
                neg_lines.append(f"2 $Package.{node}::gnd")

        if pos_lines and neg_lines:
            # Join into one string block with newlines, as expected by _fill_ic_decap_vias
            block_str = "\n".join(pos_lines + neg_lines)
            ic_blocks.append(block_str)

    return ic_blocks


def _board_outline_from_shapes(shape_bodies: Dict[str, str]) -> List[np.ndarray]:
    """
    Synthesize a rectangular board outline from the bounding box of all kept
    polygons/boxes (in meters). This avoids empty bxy on new SPD.
    """
    xs: List[float] = []
    ys: List[float] = []

    # Capture numbers followed by 'mm' inside polygons/boxes
    for body in shape_bodies.values():
        for xs_str, ys_str in re.findall(r"([-\d\.eE\+]+)mm\s+([-\d\.eE\+]+)mm", body):
            try:
                xs.append(float(xs_str) * 1e-3)
                ys.append(float(ys_str) * 1e-3)
            except Exception:
                pass

    if not xs or not ys:
        return []

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Simple CCW rectangle
    rect = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]], dtype=float)
    return [rect]

def _extract_start_stop_type_multi(via_lines, node_info, ic_blocks, decap_blocks, verbose=False):
    """
    Build start/stop layer arrays and via type array.
    Updated to support multiple power nets (pwr, pwr1, pwr2...).
    """
    start_layers, stop_layers, via_types = [], [], []

    for via_name, (up, lo) in enumerate(via_lines):
        up_info = node_info.get(up)
        lo_info = node_info.get(lo)
        if not up_info or not lo_info:
            continue

        # Layers
        start_layers.append(up_info["layer"])
        stop_layers.append(lo_info["layer"])

        # Decide via type from net name
        net = up_info["net"].lower()
        if net == "gnd":
            vtype = 0
        elif net.startswith("pwr"):
            vtype = 1
        else:
            vtype = -1  # Unknown / skip

        via_types.append(vtype)

        if verbose and via_name < 10:  # print first 10 as debug
            print(f"[SPD_MULTI][VIA_DBG] {via_name}: up={up}({up_info['net']}, L{up_info['layer']}) "
                  f"lo={lo}({lo_info['net']}, L{lo_info['layer']}) type={vtype}")

    if len(start_layers) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    return (np.asarray(start_layers, dtype=int),
            np.asarray(stop_layers, dtype=int),
            np.asarray(via_types, dtype=int))

# -----------------------------
# Top-level multi parser
# -----------------------------

def parse_spd_multi(brd_cls, spd_path: str, pwr_nets: List[str], verbose: bool = False) -> List[PDN]:
    """
    Build multiple 'virtual boards' (target PWRx + GND) from a single multi-plane SPD.

    Returns: list of PDNs identical in shape to parse_spd(...) (one per requested power net):
      [bxy,
       ic_via_xy, ic_via_type,
       start_layers, stop_layers, via_type,
       decap_via_xy, decap_via_type,
       stackup,
       (optional) buried_via_xy, buried_via_type]
    """
    text = spd_parser._read(spd_path)

    # 1) Shapes and Patch mapping
    shapes = _shape_blocks(text)
    if verbose:
        print(f"[SPD_MULTI] Found {len(shapes)} shape blocks")
    patchmap = _patch_map(text)
    if verbose:
        print(f"[SPD_MULTI] PatchSignal map entries: {len(patchmap)}")

    # 2) Nodes (net-aware)
    node_info_all = _extract_nodes_multinet(text, "pwr")
    nets_detected = sorted({inf["net"] for inf in node_info_all.values()})
    if verbose:
        print(f"[SPD_MULTI] Nets detected in SPD: {nets_detected}")

    # 3) Pre-extract via topology & connect text (reuse old helpers where possible)
    via_lines_all = spd_parser._extract_via_lines(text)  # pairs of ('NodeXXXX','NodeYYYY')
    # We'll filter this per-net later.
    # For IC blocks, we will build our own (new SPD 'Connect IC') and *replace* old ic_blocks.
    # Decaps: none in new SPD — supply empty list.
    decap_blocks: List[List[str]] = []

    results: List[Tuple] = []

    # Loop over requested power nets
    for net in pwr_nets:
        target = net.lower().replace("pwr0", "pwr")
        if verbose:
            print(f"\n========== Building sub-board for net '{target}' ==========")

        # 3a) Keep shapes that include either ::target+ or ::gnd+ in the body
        kept_shapes = {name: body for name, body in shapes.items()
                       if (f"::{target}+" in body.lower()) or ("::gnd+" in body.lower())}
        if verbose:
            print(f"[SPD_MULTI] Shapes kept for {target}: {list(kept_shapes.keys())}")

        # 3b) Nodes: only target PWRx and GND
        node_info = {n: inf for n, inf in node_info_all.items()
                     if inf["net"] in (target, "gnd")}
        for n, inf in node_info.items():
            if inf["net"] not in (target, "gnd"):
                # force assignment: any stray becomes this board’s target
                inf["net"] = target
                inf["type"] = 1 if target != "gnd" else 0

        if verbose:
            print(f"[SPD_MULTI] Nodes kept for {target}: {len(node_info)}")

        # 3c) Vias: endpoints must both be kept nodes
        via_lines = [(u, l) for (u, l) in via_lines_all if (u in node_info and l in node_info)]
        if verbose:
            print(f"[SPD_MULTI] Via lines kept for {target}: {len(via_lines)}")

        # 3d) IC ports from new '.Connect IC' blocks
        ic_blocks = _connect_ic_blocks_for_net(text, target)
        if verbose:
            print(f"[SPD_MULTI] IC blocks for {target}: {len(ic_blocks)}")

        # 4) Build PDN board object via your existing fill helpers
        brd = brd_cls()

        # Board outline: Try old extractor; if empty, synthesize rectangle
        bxy = spd_parser._extract_board_polygons(text)
        if not bxy:
            bxy = _board_outline_from_shapes(kept_shapes)
            if verbose:
                print(f"[SPD_MULTI] Board outline synthesized: {'yes' if bxy else 'no'}")
        brd.bxy = bxy

        # IC / Decap vias (decap_blocks empty in new SPD)
        spd_parser._fill_ic_decap_vias(brd, node_info, ic_blocks, decap_blocks)

        # --- FIX: Recompute ic_via_type from node_info net tags ---
        if hasattr(brd, "ic_node_names"):
            new_types = []
            for n in brd.ic_node_names:
                info = node_info.get(n)
                if not info:
                    new_types.append(0)
                else:
                    new_types.append(0 if info["net"] == "gnd" else 1)
            brd.ic_via_type = np.array(new_types, dtype=int)

        print(f"[SPD_MULTI][IC_TYPE_FIX] total={len(brd.ic_via_type)}, "
            f"type0={np.sum(brd.ic_via_type==0)}, type1={np.sum(brd.ic_via_type==1)}")

        # Start/stop/type arrays
        sl, tl, vt = _extract_start_stop_type_multi(via_lines, node_info, ic_blocks, decap_blocks)
        brd.start_layers = np.asarray(sl, np.int32) - 1 if len(sl) else np.array([], dtype=int)
        brd.stop_layers  = np.asarray(tl, np.int32) - 1 if len(tl) else np.array([], dtype=int)
        brd.via_type     = np.asarray(vt, np.int32) if len(vt) else np.array([], dtype=int)

        # Buried vias / via locs / maps / dedupe
        spd_parser._fill_buried_vias(brd, via_lines, node_info)
        spd_parser._fill_via_locs(brd, node_info)
        spd_parser._fill_port_cavity_maps(brd, ic_blocks, decap_blocks)
        spd_parser._dedupe_across_groups(brd, eps=1e-7, rdec=9)

        # 5) Stackup mask from kept shapes + patchmap
        brd.stackup = _layer_mask_for_net(kept_shapes, patchmap, target)
        if verbose:
            print(f"[SPD_MULTI] Stackup mask for {target}: {brd.stackup}")

        # 6) Assemble return tuple compatible with parse_spd(...)
        results.append(brd)

        # Helpful debugging snapshots
        if verbose:
            print(f"[SPD_MULTI][{target}] ic_vias: {len(brd.ic_via_xy) if hasattr(brd, 'ic_via_xy') else 0}, "
                  f"vias: {len(brd.via_type)}, buried: {len(brd.buried_via_type) if hasattr(brd, 'buried_via_type') else 0}")

    return results

