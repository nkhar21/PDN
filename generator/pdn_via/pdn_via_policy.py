from __future__ import annotations
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

from generator.pdn_enums import NetType, ViaRole
from generator.pdn_via.pdn_via_model import Via
from generator.pdn_geometry import point_in_polygon

# Tiny protocol to avoid importing PDNBoard (no circulars)
class BoardLike(Protocol):
    stackup: any
    outline: any  # has bxy: list[Polygon]

def validate_via_placement(board: BoardLike, via: Via) -> None:
    """
    - stackup ready
    - layer bounds ok
    - via xy is inside outline polygon(s) on start & stop layers
    - stackup mask contains at least one matching net at endpoints
    """
    if board.stackup is None:
        raise ValueError("Stackup must be defined before adding vias")
    if board.outline is None or board.outline.bxy is None:
        raise ValueError("Board outline must be set before adding vias")

    num_layers: int = int(board.stackup.num_layers)
    if via.start_layer < 0 or via.stop_layer >= num_layers:
        raise ValueError(f"Via layers {via.start_layer}->{via.stop_layer} out of bounds (board has {num_layers} layers)")

    # Geometry containment
    if len(board.outline.bxy) == 1:
        poly = board.outline.bxy[0]
        for layer in (via.start_layer, via.stop_layer):
            if not point_in_polygon(via.xy, poly):
                raise ValueError(f"Via at {via.xy} is outside board outline on layer {layer}")
    else:
        for layer in (via.start_layer, via.stop_layer):
            poly = board.outline.bxy[layer]
            if not point_in_polygon(via.xy, poly):
                raise ValueError(f"Via at {via.xy} is outside board outline on layer {layer}")

    # Stackup mask check (requires mask codes 0=GND,1=PWR)
    mask: NDArray[np.int_] = board.stackup.stackup_mask
    layer_types = {int(mask[via.start_layer]), int(mask[via.stop_layer])}
    if via.via_type == NetType.PWR:
        if NetType.PWR.value not in layer_types:
            raise ValueError(f"Power via at {via.xy} must connect to at least one PWR layer (layers {via.start_layer}->{via.stop_layer})")
    else:  # GND
        if NetType.GND.value not in layer_types:
            raise ValueError(f"Ground via at {via.xy} must connect to at least one GND layer (layers {via.start_layer}->{via.stop_layer})")

def assign_via_role(via: Via, num_layers: int) -> ViaRole:
    a, b = sorted((via.start_layer, via.stop_layer))
    if a == 0 and b == num_layers - 1:
        return ViaRole.THROUGH
    if a == 0 or b == num_layers - 1:
        return ViaRole.BLIND
    return ViaRole.BURIED
