from enum import IntEnum

class NetType(IntEnum):
    """Type for ground or power connection."""
    GND = 0
    PWR = 1

class ViaRole(IntEnum):
    """Functional role of a via in the PDN."""
    UNSET = -1
    BURIED = 0
    BLIND = 1
    THROUGH = 2


class PortRole(IntEnum):
    """ Type of port for PDN"""
    IC = 0
    DECAP = 1
