import numpy as np
from typing import Tuple, List
from BEM_AC_NVM_PDN import PDN


def generate_segments(
    bxy: np.ndarray,
    d: float
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[np.ndarray]]:
    """
    Generate segmented boundary attributes from board outlines.

    Parameters
    ----------
    bxy : np.ndarray[dtype=object]
        Board outline polygons (N_layers,). Each entry is (M,2).
    d : float
        Segment length (meters).

    Returns
    -------
    sxy : np.ndarray
        Concatenated boundary segments for all polygons (total_segs, 4).
        Each row is [x1, y1, x2, y2].
    sxy_index_ranges : list of (start, stop) indices for each polygon in sxy.
    sxy_list : list of per-polygon segment arrays.
    """
    sxy_list: List[np.ndarray] = []
    sxy_index_ranges: List[Tuple[int, int]] = []
    offset = 0
    pdn = PDN() # probably will have to modify after geometry gets decomposed out of BEM_AC_NVM_PDN.py

    for poly in bxy:
        segs = pdn.seg_bd_node(poly, d)  # imported directly
        sxy_list.append(segs)
        n_seg = segs.shape[0]
        sxy_index_ranges.append((offset, offset + n_seg))
        offset += n_seg

    sxy = np.concatenate(sxy_list, axis=0) if sxy_list else np.zeros((0, 4))
    return sxy, sxy_index_ranges, sxy_list
