from __future__ import annotations
from typing import Sequence
from generator.pdn_enums import NetType, PortSide
from generator.pdn_interfaces import BoardView

def _touches_side(board: BoardView, start_layer: int, stop_layer: int, side: PortSide) -> bool:
    top = 0
    bot = board.stackup.num_layers - 1
    if side == PortSide.TOP:
        return (start_layer == top) or (stop_layer == top)
    else:
        return (start_layer == bot) or (stop_layer == bot)

def validate_terminal_ids(
    board: BoardView,
    via_ids: Sequence[int],
    required_net: NetType,
    side: PortSide,
) -> None:
    if board.stackup is None:
        raise ValueError("Stackup must be defined before port checks")
    if not via_ids:
        raise ValueError("Terminal must contain at least one via id")

    for vid in via_ids:
        if not board.vias.has_id(vid):
            raise KeyError(f"Via id {vid} not found in collection")
        v = board.vias.get_by_id(vid)
        if v.via_type != required_net:
            raise ValueError(
                f"Via id={vid} at {v.xy} has net {v.via_type.name}; expected {required_net.name}"
            )
        if not _touches_side(board, v.start_layer, v.stop_layer, side):
            s = "TOP" if side == PortSide.TOP else "BOTTOM"
            raise ValueError(
                f"Via id={vid} at {v.xy} must touch the {s} skin to be used in this port"
            )
