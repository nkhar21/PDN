from __future__ import annotations
from copy import deepcopy
import numpy as np
import skrf as rf

def short_1port_z(z, map2orig_input, shorted_port):
    """
    Short one port in Z-matrix (with mapping).
    
    Parameters
    ----------
    z : ndarray (n, n)
        Original Z-matrix
    map2orig_input : list
        Mapping of reduced ports to original port indices
    shorted_port : int
        Port index to short (zero-based)
    """
    output_net = deepcopy(z)
    output_net = np.linalg.inv(
        np.delete(
            np.delete(np.linalg.inv(output_net), shorted_port, axis=1),
            shorted_port, axis=0
        )
    )
    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[shorted_port]

    return output_net, map2orig_output


def short_1port_toolz(z_full, shorted_port):
    """
    Short one port in Z-matrix (no mapping, just reduced Z).
    
    Parameters
    ----------
    z_full : ndarray (n, n)
        Original Z-matrix
    shorted_port : int
        Port index to short (zero-based)
    """
    z_inv = np.linalg.inv(z_full)
    z_shorted = np.delete(np.delete(z_inv, shorted_port, axis=0), shorted_port, axis=1)
    return np.linalg.inv(z_shorted)


def short_ports_reduce_z(Zf, ports_to_short):
    """
    Reduce a multiport Z(f) by shorting a set of ports (multiport Schur via Y-deletion).

    Parameters
    ----------
    Zf : ndarray, shape (nf, n, n)
        Frequency-dependent Z-matrix.
    ports_to_short : sequence[int]
        Zero-based indices of ports to short to ground.

    Returns
    -------
    Zred : ndarray, shape (nf, n', n')
        Reduced Z over remaining (unshorted) ports, in original order minus the removed ports.
    keep_idx : list[int]
        The kept (unshorted) port indices mapping to rows/cols of Zred.
    """
    ports_to_short = sorted(set(ports_to_short))
    nf, n, _ = Zf.shape
    keep = [p for p in range(n) if p not in ports_to_short]

    Zred = np.empty((nf, len(keep), len(keep)), dtype=complex)
    for k in range(nf):
        Z = Zf[k]
        Y = np.linalg.inv(Z)
        # delete shorted rows/cols in Y (v_short = 0)
        Yuu = np.delete(np.delete(Y, ports_to_short, axis=0), ports_to_short, axis=1)
        Zred[k] = np.linalg.inv(Yuu)
    return Zred, keep


def z_in_with_shorts(Zf, drive_port, ports_to_short):
    """
    Driving-point input impedance at 'drive_port' with all 'ports_to_short' shorted.

    Returns
    -------
    Zin : ndarray, shape (nf,)
        Input impedance vs frequency.
    """
    # If the drive port is among the shorts, make the drive the only kept port:
    ports_to_short = sorted(set(ports_to_short))
    if drive_port in ports_to_short:
        raise ValueError("drive_port cannot be in ports_to_short")

    Zred, keep = short_ports_reduce_z(Zf, ports_to_short)
    # Find where the drive_port landed among kept indices
    try:
        i = keep.index(drive_port)
    except ValueError:
        raise RuntimeError("Drive port not found after reduction")

    # If only one kept port remains, Zred is 1x1 and Zin is that scalar
    return Zred[:, i, i]


def reduce_with_shunt_loads(Zf, loads):



    """
    Optional: terminate non-drive ports with finite shunt impedances instead of shorts.

    loads : dict[int, complex or ndarray(nf,)]
        Map of port index -> Z_load (per frequency allowed).
        A short is the limit Z_load -> 0 (i.e., Y_load -> inf).
        An open is Z_load -> inf (i.e., Y_load -> 0).

    Returns
    -------
    Zeq : ndarray, shape (nf, n, n)
        Equivalent Z(f) after adding shunt loads to ground at the specified ports.
    """
    nf, n, _ = Zf.shape
    Zeq = np.empty_like(Zf)
    for k in range(nf):
        Z = Zf[k]
        Y = np.linalg.inv(Z)
        # Add shunt admittances on diagonal of Y
        Yadd = np.zeros((n, n), dtype=complex)
        for p, ZL in loads.items():
            ZLk = ZL[k] if hasattr(ZL, "__len__") else ZL
            if np.isinf(ZLk):      # open -> no change
                Yp = 0.0
            elif ZLk == 0:         # short -> infinite; handle by very large number
                Yp = 1e20
            else:
                Yp = 1.0 / ZLk
            Yadd[p, p] += Yp
        Yeq = Y + Yadd
        Zeq[k] = np.linalg.inv(Yeq)
    return Zeq



def short_1port(input_net: rf.Network,
                map2orig_input: list[int] | None = None,
                shorted_port: int = 1):
    """
    Create a *shorted* 1-port and connect it to `input_net` at `shorted_port`,
    effectively shorting that port and returning the reduced network + port map.

    Parameters
    ----------
    input_net : skrf.Network
        Multiport network to short one port on.
    map2orig_input : list[int] | None
        Mapping from current port indices to original port indices.
        If None, defaults to [0, 1, ..., N-1].
    shorted_port : int
        Port index (0-based) in `input_net` to short.

    Returns
    -------
    output_net : skrf.Network
        New network with `shorted_port` shorted and removed from the external ports.
    map2orig_output : list[int]
        Updated port mapping with the shorted port removed.

    Notes
    -----
    - We build a 1-port ideal short (S = −1) and connect it using
      `skrf.network.connect(input_net, k, short_net, 0)`.
    """
    if map2orig_input is None:
        map2orig_input = list(range(input_net.nports))

    short_net = deepcopy(input_net.s11)
    short_net.s = -1.0 * np.ones(short_net.f.shape[0])
    output_net = rf.network.connect(input_net, shorted_port, short_net, 0)

    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[shorted_port]
    return output_net, map2orig_output


def connect_1decap(input_net_z: np.ndarray,
                   map2orig_input: list[int],
                   connect_port: int,
                   decap_z11: np.ndarray):
    """
    Schur-complement a 1-port decap (impedance `decap_z11`) onto `connect_port`
    of a multiport Z-matrix, then remove that port.

    Parameters
    ----------
    input_net_z : (F, P, P) complex ndarray
        Frequency-by-ports impedance tensor of the network.
    map2orig_input : list[int]
        Port map before connection.
    connect_port : int
        Port index (0-based) to attach the decap to (and then eliminate).
    decap_z11 : (F, 1, 1) or (F,) complex ndarray
        The decap’s input impedance per frequency.

    Returns
    -------
    output_net_z : (F, P-1, P-1) complex ndarray
        New Z after connecting the decap at `connect_port` and eliminating it.
    map2orig_output : list[int]
        Updated port map (without `connect_port`).

    Math
    ----
    Partition Z = [[Zaa, Zap], [Zpa, Zpp]], then
      Z' = Zaa - Zap (Zpp + Zqq)^{-1} Zpa,
    where Zqq = decap_z11.
    """
    Zaa = np.delete(np.delete(input_net_z, connect_port, axis=1), connect_port, axis=2)

    Zpp = input_net_z[:, connect_port, connect_port].reshape((-1, 1, 1))
    Zqq = decap_z11.reshape((-1, 1, 1)) if decap_z11.ndim == 1 else decap_z11

    Zap = input_net_z[:, :, connect_port].reshape((input_net_z.shape[0], input_net_z.shape[1], 1))
    Zap = np.delete(Zap, connect_port, axis=1)

    Zpa = input_net_z[:, connect_port, :].reshape((input_net_z.shape[0], 1, input_net_z.shape[2]))
    Zpa = np.delete(Zpa, connect_port, axis=2)

    inv = np.linalg.inv(Zpp + Zqq)
    output_net_z = Zaa - np.einsum('fij,fjk,fkl->fil', Zap, inv, Zpa)

    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[connect_port]
    return output_net_z, map2orig_output