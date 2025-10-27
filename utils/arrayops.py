"""
Small, generic helpers for list/index operations and square-matrix merging.
These are solver-agnostic and safe to use from both BEM/CIM NVM code.
"""

from __future__ import annotations
from copy import deepcopy
import numpy as np


def delete_multiple_element(lst: list, indices: list[int]) -> None:
    """
    Remove multiple entries from a list by index, safely (delete high -> low).

    Parameters
    ----------
    lst : list
        List to mutate in place.
    indices : list[int]
        Indices to delete; out-of-range are ignored.
    """
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(lst):
            lst.pop(idx)


def max_value(nested: list[list[int] | list[float]]) -> float:
    """
    Max across a 2-level nested list.

    Examples
    --------
    >>> max_value([[1, 2], [3]]) == 3
    """
    return max(max(sub) for sub in nested)


def find_index(haystack: list[int], needles: list[int]) -> list[int]:
    """
    Map each value in `needles` to its index in `haystack`.

    Parameters
    ----------
    haystack : list[int]
        List to search within.
    needles : list[int]
        Values to locate.

    Returns
    -------
    list[int]
        Indices in `haystack` corresponding to each value in `needles`.

    Raises
    ------
    ValueError
        If a value in `needles` is not found in `haystack`.
    """
    return [haystack.index(int(v)) for v in needles]


def merge_square_mat(Min: np.ndarray,
                     merge_indices: list[int],
                     map2orig_input: list[int] | None = None):
    """
    Electrically merge rows/cols of a square matrix for nodes/ports that
    are equipotential. Keeps the first index; folds the rest into it.

    Parameters
    ----------
    Min : (N, N) ndarray
        Square matrix (e.g., L^{-1}, Y, etc.).
    merge_indices : list[int]
        Indices to merge together; first entry is the survivor.
    map2orig_input : list[int] | None
        Existing port map; if None, assumes identity.

    Returns
    -------
    Mout : (N', N') ndarray
        Merged matrix with |merge_indices| - 1 fewer rows/cols.
    map2orig_output : list[int]
        Updated map with removed indices deleted.

    Notes
    -----
    Merging sums the survivor row/col with the others (KCL/KVL consistent),
    then deletes the merged rows/cols.
    """
    Mout = deepcopy(Min)
    map2orig_output = list(range(Min.shape[0])) if map2orig_input is None else deepcopy(map2orig_input)

    if len(merge_indices) <= 1:
        return Mout, map2orig_output

    keep = merge_indices[0]
    for idx in merge_indices[1:]:
        Mout[keep, :] += Mout[idx, :]
        Mout[:, keep] += Mout[:, idx]

    drop = merge_indices[1:]
    Mout = np.delete(np.delete(Mout, drop, axis=0), drop, axis=1)
    delete_multiple_element(map2orig_output, drop)
    return Mout, map2orig_output
