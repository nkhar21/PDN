from __future__ import annotations

from typing import List, Literal, Optional
import numpy as np
from numpy.typing import NDArray

from .pdn_outline_model import BoardOutlineModel
from .pdn_outline_policy import (
    set_outline as _set_outline,
    set_segmentation as _set_segmentation,
)


class BoardOutline:
    """
    Thin façade around the model+policy so existing code can keep doing:
        outline = BoardOutline()
        outline.set_outline(...)
        outline.set_segmentation(...)
        outline.summary()
    """
    def __init__(self, model: Optional[BoardOutlineModel] = None) -> None:
        self.model = model or BoardOutlineModel()

    # ---- properties to preserve old attribute access ----
    @property
    def bxy(self):
        return self.model.bxy

    @property
    def sxy(self):
        return self.model.sxy

    @property
    def seg_len(self):
        return self.model.seg_len

    @property
    def sxy_index_ranges(self):
        return self.model.sxy_index_ranges

    @property
    def sxy_list(self):
        return self.model.sxy_list

    # ---- API (delegates to policy) ----
    def set_outline(
        self,
        shapes: List[NDArray[np.float64]],
        *,
        units: Literal["mm", "m"] = "mm",
        auto_close: bool = True,
        tol: float = 1e-9,
    ) -> None:
        """
        Provide exactly one polygon per layer. Units are converted to meters.
        """
        _set_outline(self.model, shapes, units=units, auto_close=auto_close, tol=tol)

    def set_segmentation(self, seg_len: float) -> None:
        _set_segmentation(self.model, seg_len)

    def summary(self) -> None:
        self.model.summary()
