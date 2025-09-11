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

