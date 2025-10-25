# generator/pdn_outline/__init__.py

from .pdn_outline_model import BoardOutlineModel
from .pdn_outline_service import BoardOutline
from .pdn_outline_policy import set_outline, set_segmentation

from .gen_outline import generate_bxy, validate_bxy
from .gen_segments import generate_segments

__all__ = [
    "BoardOutlineModel",
    "BoardOutline",
    "set_outline",
    "set_segmentation",
    "generate_bxy",
    "validate_bxy",
    "generate_segments",
]
