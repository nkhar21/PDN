import numpy as np
from abc import ABC, abstractmethod
from typing import List, Literal
from enum import Enum


class OutlineMode(Enum):
    COLLAPSED = "collapsed"
    NONCOLLAPSED = "noncollapsed"


class OutlineGenerator(ABC):
    """
    Abstract interface for generating board outlines (bxy).
    """

    def __init__(self, mode: OutlineMode):
        self.mode = mode

    @abstractmethod
    def generate(self, shapes: List[np.ndarray]) -> np.ndarray:
        """
        Args:
            shapes: list of polygons; each polygon is an (N,2) ndarray of xy points.
                    Units may be mm or m depending on implementation.
        Returns:
            np.ndarray[dtype=object]: outer shape (num_layers,), each entry is (N,2) array in meters.
        """
        raise NotImplementedError


class CollapsedOutlineGenerator(OutlineGenerator):
    """
    Collapse multiple identical outlines into a single polygon.
    Always returns np.ndarray[dtype=object] with one polygon per entry.
    """

    def __init__(
        self,
        *,
        tol: float = 1e-9,
        units: Literal["mm", "m"] = "mm",
        auto_close: bool = True,
    ):
        super().__init__(OutlineMode.COLLAPSED)
        self.tol = tol
        self.scale = 1e-3 if units == "mm" else 1.0
        self.auto_close = auto_close

    def _to_meters_and_close(self, s: np.ndarray) -> np.ndarray:
        if not isinstance(s, np.ndarray):
            raise TypeError("Each shape must be a numpy.ndarray")
        a = s.astype(float) * self.scale
        if self.auto_close and a.shape[0] >= 2 and not np.allclose(a[0], a[-1], atol=self.tol):
            a = np.vstack([a, a[0]])
        return a

    def generate(self, shapes: List[np.ndarray]) -> np.ndarray:
        if not shapes:
            return np.array([], dtype=object)

        shapes_m = [self._to_meters_and_close(s) for s in shapes]
        first = shapes_m[0]

        all_same = all(
            (s.shape == first.shape) and np.allclose(s, first, atol=self.tol)
            for s in shapes_m
        )

        if all_same:
            out = np.empty(1, dtype=object)
            out[0] = first
            return out
        else:
            out = np.empty(len(shapes_m), dtype=object)
            for i, s in enumerate(shapes_m):
                out[i] = s
            return out


class NonCollapsedOutlineGenerator(OutlineGenerator):
    """
    Keep all outlines exactly as provided (after conversion and optional closing).
    Always returns np.ndarray[dtype=object] with one polygon per layer.
    """

    def __init__(
        self,
        *,
        units: Literal["mm", "m"] = "mm",
        auto_close: bool = True,
        tol: float = 1e-9,
    ):
        super().__init__(OutlineMode.NONCOLLAPSED)
        self.scale = 1e-3 if units == "mm" else 1.0
        self.auto_close = auto_close
        self.tol = tol

    def _to_meters_and_close(self, s: np.ndarray) -> np.ndarray:
        if not isinstance(s, np.ndarray):
            raise TypeError("Each shape must be a numpy.ndarray")
        a = s.astype(float) * self.scale
        if self.auto_close and a.shape[0] >= 2 and not np.allclose(a[0], a[-1], atol=self.tol):
            a = np.vstack([a, a[0]])
        return a

    def generate(self, shapes: List[np.ndarray]) -> np.ndarray:
        if not shapes:
            return np.array([], dtype=object)

        shapes_m = [self._to_meters_and_close(s) for s in shapes]
        out = np.empty(len(shapes_m), dtype=object)
        for i, s in enumerate(shapes_m):
            out[i] = s
        return out


class OutlineGeneratorFactory:
    """
    Factory for creating OutlineGenerator instances.
    """

    @staticmethod
    def create(mode: OutlineMode, **kwargs) -> OutlineGenerator:
        if mode == OutlineMode.COLLAPSED:
            return CollapsedOutlineGenerator(**kwargs)
        elif mode == OutlineMode.NONCOLLAPSED:
            return NonCollapsedOutlineGenerator(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
