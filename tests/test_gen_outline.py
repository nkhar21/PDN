import numpy as np
import pytest
from generator.gen_outline import OutlineGeneratorFactory, OutlineMode


def make_square(size=50.0, units="mm", closed=True) -> np.ndarray:
    """Return a square polygon as np.ndarray of shape (N,2)."""
    pts = np.array([
        [0.0, 0.0],
        [size, 0.0],
        [size, size],
        [0.0, size],
    ], dtype=float)
    if closed:
        pts = np.vstack([pts, pts[0]])
    if units == "mm":
        return pts  # still in mm; generator will scale
    elif units == "m":
        return pts * 1e-3  # convert to meters
    else:
        raise ValueError("units must be 'mm' or 'm'")


def test_empty_input():
    gen = OutlineGeneratorFactory.create(OutlineMode.COLLAPSED, units="mm")
    result = gen.generate([])
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


def test_single_polygon():
    square = make_square()
    gen = OutlineGeneratorFactory.create(OutlineMode.COLLAPSED, units="mm")
    result = gen.generate([square])
    assert result.shape == (1,)
    # 50 mm -> 0.05 m
    assert np.allclose(result[0][2], [0.05, 0.05])


def test_identical_polygons_collapsed():
    shapes = [make_square(), make_square()]
    gen = OutlineGeneratorFactory.create(OutlineMode.COLLAPSED, units="mm")
    result = gen.generate(shapes)
    assert result.shape == (1,)  # collapsed
    assert np.allclose(result[0][2], [0.05, 0.05])


def test_identical_polygons_noncollapsed():
    shapes = [make_square(), make_square()]
    gen = OutlineGeneratorFactory.create(OutlineMode.NONCOLLAPSED, units="mm")
    result = gen.generate(shapes)
    assert result.shape == (2,)  # both kept
    assert np.allclose(result[0][2], [0.05, 0.05])
    assert np.allclose(result[1][2], [0.05, 0.05])


def test_different_polygons_collapsed():
    shapes = [make_square(50.0), make_square(45.0)]
    gen = OutlineGeneratorFactory.create(OutlineMode.COLLAPSED, units="mm")
    result = gen.generate(shapes)
    assert result.shape == (2,)  # not collapsed, because different
    assert not np.allclose(result[0], result[1])


def test_units_in_meters():
    square_m = make_square(50.0, units="m")
    gen = OutlineGeneratorFactory.create(OutlineMode.COLLAPSED, units="m")
    result = gen.generate([square_m])
    assert result.shape == (1,)
    assert np.allclose(result[0][2], [0.05, 0.05])  # already in meters


def test_auto_close_polygon():
    open_square = make_square(closed=False)
    gen = OutlineGeneratorFactory.create(OutlineMode.COLLAPSED, units="mm", auto_close=True)
    result = gen.generate([open_square])
    assert np.allclose(result[0][0], result[0][-1])  # first == last
