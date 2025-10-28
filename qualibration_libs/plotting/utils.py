from __future__ import annotations

from typing import Tuple, Any, Mapping, Iterable

import numpy as np
import xarray as xr


def get_axis_label(ds: xr.Dataset, coord: str) -> str:
    if coord not in ds.coords:
        return coord
    attrs = ds.coords[coord].attrs or {}
    name = attrs.get("long_name", coord)
    units = attrs.get("units")
    return f"{name} [{units}]" if units else name


def get_qubit_title(qubit_name: str) -> str:
    return f"{qubit_name}"


def parse_grid_location(loc: str) -> tuple[int, int]:
    col_str, row_str = loc.split(",")
    col = int(col_str)
    row = int(row_str)
    return row, col


def label_from_attrs(name: str, attrs: Mapping[str, Any]) -> str:
    long_name = attrs.get("long_name")
    units = attrs.get("units")
    base = long_name or name
    
    # Check if the label contains LaTeX syntax (starts and ends with $)
    if isinstance(base, str) and base.startswith('$') and base.endswith('$'):
        # Return as-is for LaTeX rendering
        label = base
    else:
        # Regular text formatting
        label = base
    
    if units:
        if isinstance(units, str) and units.startswith('$') and units.endswith('$'):
            # LaTeX units
            return f"{label} {units}"
        else:
            # Regular units
            return f"{label} [{units}]"
    else:
        return label


def map_hue_value(dim_name: str, value: Any) -> str:
    if dim_name.lower() in ("detuning_signs", "sign"):
        if value in ("+", "+1", 1, True):
            return "Δ = +"
        if value in ("-", "-1", -1, False):
            return "Δ = −"
    return f"{dim_name} = {value}"


def compute_secondary_ticks(
    primary_vals: Iterable[float], secondary_vals: Iterable[float]
) -> Tuple[list, list]:
    p = list(primary_vals)
    s = list(secondary_vals)
    if not p or not s or len(p) != len(s):
        return [], []
    n = len(p)

    # Target approximately 6-7 ticks evenly distributed
    num_ticks = min(7, n)

    # Use numpy linspace to generate evenly spaced indices
    # This ensures ticks are evenly distributed from start to end
    idxs = np.linspace(0, n - 1, num_ticks, dtype=int).tolist()

    # Remove duplicates while preserving order (can happen with small n)
    seen = set()
    idxs = [i for i in idxs if not (i in seen or seen.add(i))]

    tickvals = [p[i] for i in idxs]

    # Format tick text - use fewer digits for floating point numbers
    ticktext = []
    for i in idxs:
        val = s[i]
        if isinstance(val, (float, np.floating)):
            # Format with 4 significant figures
            ticktext.append(f"{val:.4g}")
        else:
            ticktext.append(str(val))

    return tickvals, ticktext
