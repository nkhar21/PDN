# pdn_bridge/test.py
import numpy as np

from generator.pdn_board import PDNBoard
from generator.pdn_outline.gen_outline import OutlineMode
from generator.pdn_stackup.pdn_stackup_model import Stackup
from pdn_bridge.wrapper import PDNFromBoard


def make_square_board() -> PDNBoard:
    board = PDNBoard()
    square = np.array([[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]], dtype=float)
    board.set_outline([square], mode=OutlineMode.COLLAPSED, units="mm")
    board.set_segmentation(seg_len=0.001)  # 1 mm
    stackup = Stackup(
        num_layers=4,
        stackup_mask=[0, 1, 0, 0],
        die_t=[0.0005, 0.00043, 0.00047],
        er_list=[4.0, 3.43, 3.8],
        d_r=[0.000035] * 4,
    )
    board.set_stackup(stackup)
    return board


def _fmt(a: np.ndarray) -> str:
    return np.array2string(a, separator=", ")


def _allclose(a, b, name, rtol=1e-12, atol=1e-12):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{name} mismatch:\n  board={_fmt(a)}\n  pdn  ={_fmt(b)}")


def _equal(a, b, name):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not np.array_equal(a, b):
            raise AssertionError(f"{name} mismatch:\n  board={_fmt(a)}\n  pdn  ={_fmt(b)}")
    else:
        if a != b:
            raise AssertionError(f"{name} mismatch:\n  board={a}\n  pdn  ={b}")


def _as_poly_2col(poly_like) -> np.ndarray:
    """
    Coerce any polygon representation into a strict (N,2) float array.
    Handles lists of pairs, object arrays of small arrays/tuples, etc.
    """
    arr = np.asarray(poly_like, dtype=object)
    # already numeric (N,2)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype != object:
        return arr.astype(float, copy=False)

    # otherwise, build row-wise
    rows = []
    for pt in arr:
        pt_arr = np.asarray(pt, dtype=float).reshape(-1)
        if pt_arr.size != 2:
            raise ValueError(f"Polygon vertex must have 2 elements, got {pt_arr.size} in {pt!r}")
        rows.append(pt_arr)
    return np.vstack(rows)


def compare_board_and_pdn(board: PDNBoard, pdn) -> None:
    # seg_len
    print("seg_len:")
    print(f"  board={board.outline.seg_len} | pdn={getattr(pdn, 'seg_len', None)}")
    _allclose(float(board.outline.seg_len), float(getattr(pdn, "seg_len", np.nan)), "seg_len")

    # polygons (bxy) — count, shapes, and full coords
    print("polygons (bxy): count, shapes, and coords:")
    board_polys = list(board.outline.bxy) if isinstance(board.outline.bxy, (list, tuple)) else list(board.outline.bxy)
    pdn_polys   = list(pdn.bxy)           if isinstance(pdn.bxy, (list, tuple)) else list(pdn.bxy)
    print(f"  board_count={len(board_polys)} | pdn_count={len(pdn_polys)}")
    _equal(len(board_polys), len(pdn_polys), "bxy count")

    for i, (b_poly, p_poly) in enumerate(zip(board_polys, pdn_polys)):
        b_arr = _as_poly_2col(b_poly)
        p_arr = _as_poly_2col(p_poly)
        print(f"  poly[{i}] shapes: board={b_arr.shape} | pdn={p_arr.shape}")
        _equal(b_arr.shape, p_arr.shape, f"bxy[{i}] shape")
        print(f"  poly[{i}] board coords:\n{_fmt(b_arr)}")
        print(f"  poly[{i}] pdn   coords:\n{_fmt(p_arr)}")
        _allclose(b_arr, p_arr, f"bxy[{i}] coords")

    # sxy (concat) shape & values
    print("sxy (concat) shape & values:")
    b_sxy = np.asarray(board.outline.sxy, dtype=float)
    p_sxy = np.asarray(pdn.sxy, dtype=float)
    print(f"  board shape={b_sxy.shape} | pdn shape={p_sxy.shape}")
    _equal(b_sxy.shape, p_sxy.shape, "sxy shape")
    _allclose(b_sxy, p_sxy, "sxy values")

    # sxy_list length & per-entry values
    print("sxy_list length & per-entry values:")
    print(f"  board len={len(board.outline.sxy_list)} | pdn len={len(pdn.sxy_list)}")
    _equal(len(board.outline.sxy_list), len(pdn.sxy_list), "sxy_list length")
    for i, (b_sxy_i, p_sxy_i) in enumerate(zip(board.outline.sxy_list, pdn.sxy_list)):
        b_arr = np.asarray(b_sxy_i, dtype=float)
        p_arr = np.asarray(p_sxy_i, dtype=float)
        _equal(b_arr.shape, p_arr.shape, f"sxy_list[{i}] shape")
        _allclose(b_arr, p_arr, f"sxy_list[{i}] values")

    # stackup mask
    print("stackup mask (per layer):")
    b_mask = np.asarray(board.stackup.stackup_mask, dtype=int)
    p_mask = np.asarray(pdn.stackup, dtype=int)
    print(f"  board={_fmt(b_mask)} | pdn={_fmt(p_mask)}")
    _equal(b_mask, p_mask, "stackup mask")

    # die_t
    print("die_t (per cavity, m):")
    b_die = np.asarray(board.stackup.die_t, dtype=float)
    p_die = np.asarray(pdn.die_t, dtype=float)
    print(f"  board={_fmt(b_die)} | pdn={_fmt(p_die)}")
    _allclose(b_die, p_die, "die_t")

    # er_list
    print("er_list (per cavity):")
    b_er = np.asarray(board.stackup.er_list, dtype=float)
    p_er = np.asarray(pdn.er_list, dtype=float)
    print(f"  board={_fmt(b_er)} | pdn={_fmt(p_er)}")
    _allclose(b_er, p_er, "er_list")

    # d_r
    print("d_r (per layer, m):")
    b_dr = np.asarray(board.stackup.d_r, dtype=float)
    p_dr = np.asarray(pdn.d_r, dtype=float)
    print(f"  board={_fmt(b_dr)} | pdn={_fmt(p_dr)}")
    _allclose(b_dr, p_dr, "d_r")

    print("\nAll checks passed ✔️")


def main():
    board = make_square_board()
    pdn = PDNFromBoard(board).build()
    compare_board_and_pdn(board, pdn)


if __name__ == "__main__":
    main()
