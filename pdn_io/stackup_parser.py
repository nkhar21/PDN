from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple, Union

import numpy as np
import pandas as pd

# Accept either a CSV path (str/Path) or a preloaded DataFrame
CsvLike = Union[str, Path, pd.DataFrame]

_SIG_RE = re.compile(r"Signal0*(\d+)$", re.IGNORECASE)


def _coerce_to_df(obj: CsvLike, kind: str) -> pd.DataFrame:
    """
    If `obj` is a DataFrame, return a copy. Otherwise treat it as a file path and read CSV.
    `kind` is just for nicer error messages ("stackup" or "layer-type").
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    # Treat as path
    p = Path(str(obj)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{kind.capitalize()} CSV not found: {p}")
    return pd.read_csv(p)


def read_stackup(stack_up_csv: CsvLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the stackup (e.g., b3_stackup.csv) from a path or a DataFrame.

    Returns:
        die_t: np.ndarray[float]   # dielectric thicknesses (Medium rows), meters
        ers:   np.ndarray[float]   # relative permittivities (Medium rows)
        d_r:   np.ndarray[float]   # signal layer thicknesses (Signal rows), meters
    """
    df = _coerce_to_df(stack_up_csv, kind="stackup")
    # Keep original casing since files use "Layer Name", "Thickness(mm)", "Er"
    df.columns = df.columns.str.strip()

    if "Layer Name" not in df.columns or "Thickness(mm)" not in df.columns:
        raise ValueError("Stackup CSV must contain 'Layer Name' and 'Thickness(mm)' columns.")

    df_medium = df[df["Layer Name"].astype(str).str.startswith("Medium", na=False)]
    df_signal = df[df["Layer Name"].astype(str).str.startswith("Signal", na=False)]

    if df_medium.empty:
        raise ValueError("No dielectric 'Medium' rows found in stackup CSV.")

    # mm -> m
    die_t = df_medium["Thickness(mm)"].astype(float).to_numpy() * 1e-3
    d_r = (
        df_signal["Thickness(mm)"].astype(float).to_numpy() * 1e-3
        if not df_signal.empty else np.array([])
    )

    if "Er" not in df_medium.columns:
        raise ValueError("Stackup CSV dielectric section must include an 'Er' column.")
    ers = df_medium["Er"].astype(float).to_numpy()

    return die_t, ers, d_r


def read_layer_type(layer_type_csv: CsvLike) -> pd.DataFrame:
    """
    Read the layer-type (e.g., b3_layer_type.csv) from a path or a DataFrame.

    Input format example (columns can be named flexibly; we'll normalize):
        layer,type
        Signal01,0
        Signal02,0
        Signal03,1
        ...

    Returns a normalized DataFrame with columns:
        layer (str)         -> 'SignalNN' (zero-padded)
        type (int)          -> 1 (power) / 0 (ground/return)
        signal_index (int)  -> NN as integer (1-based)
    """
    df = _coerce_to_df(layer_type_csv, kind="layer-type")
    # Normalize to lower-case for flexible matching
    df.columns = df.columns.str.strip().str.lower()

    # Normalize column names
    col_map = {}
    if "layer" not in df.columns:
        for c in df.columns:
            if c.startswith("layer"):
                col_map[c] = "layer"
                break
    if "type" not in df.columns:
        for c in df.columns:
            if c.startswith("type"):
                col_map[c] = "type"
                break
    if col_map:
        df = df.rename(columns=col_map)

    missing = [c for c in ("layer", "type") if c not in df.columns]
    if missing:
        raise ValueError(f"Layer-type CSV missing columns: {missing}")

    # Coerce type -> 0/1; accept strings like 'PWR'/'GND'
    def _coerce_type(v):
        if pd.isna(v):
            return 0
        if isinstance(v, (int, np.integer)):
            return int(v)
        s = str(v).strip().lower()
        if s in ("1", "pwr", "power", "powerplane", "power_plane"):
            return 1
        if s in ("0", "gnd", "ground", "return", "returnplane", "ground_plane"):
            return 0
        try:
            return int(float(s))
        except Exception:
            return 0

    df["type"] = df["type"].map(_coerce_type).astype(int)

    # Normalize layer strings and extract numeric indices
    def _norm_layer(s):
        s = str(s).strip()
        m = _SIG_RE.search(s)
        if not m:
            raise ValueError(f"Invalid layer string '{s}', expected like 'Signal03'.")
        n = int(m.group(1))
        return f"Signal{n:02d}", n

    norm = df["layer"].apply(_norm_layer)
    df["layer"], df["signal_index"] = zip(*norm)

    return df[["layer", "type", "signal_index"]]


def build_stackup_mask(layer_type_df: pd.DataFrame) -> np.ndarray:
    """
    Build a 0/1 mask over Signal layers, where index i corresponds to Signal(i+1).
    Uses the largest layer index to size the array.
    """
    if "signal_index" not in layer_type_df.columns or "type" not in layer_type_df.columns:
        raise ValueError("layer_type_df must include 'signal_index' and 'type' columns.")

    max_idx = int(layer_type_df["signal_index"].max())
    if max_idx <= 0:
        return np.zeros(0, dtype=int)

    mask = np.zeros(max_idx, dtype=int)
    for _, row in layer_type_df.iterrows():
        idx = int(row["signal_index"]) - 1
        if 0 <= idx < mask.size and int(row["type"]) == 1:
            mask[idx] = 1
    return mask


def load_layer_and_stackup(layer_type_csv: CsvLike, stack_up_csv: CsvLike):
    """
    Convenience loader that returns exactly what main.py needs:
        die_t, er_list, d_r, stackup_mask, layer_type_df
    Accepts either paths or DataFrames for both inputs.
    """
    die_t, er_list, d_r = read_stackup(stack_up_csv)
    lt_df = read_layer_type(layer_type_csv)
    stackup_mask = build_stackup_mask(lt_df)
    return die_t, er_list, d_r, stackup_mask, lt_df
