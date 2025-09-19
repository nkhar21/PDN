import numpy as np
from copy import deepcopy

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