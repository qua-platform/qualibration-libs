from __future__ import annotations

from typing import Tuple, Any, Mapping, Iterable

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
    return f"{base} [{units}]" if units else base


def map_hue_value(dim_name: str, value: Any) -> str:
    if dim_name.lower() in ("detuning_signs", "sign"):
        if value in ("+", "+1", 1, True):
            return "Δ = +"
        if value in ("-", "-1", -1, False):
            return "Δ = −"
    return f"{dim_name} = {value}"


def compute_secondary_ticks(
    primary_vals: Iterable[float],
    secondary_vals: Iterable[float]
) -> Tuple[list, list]:
    p = list(primary_vals)
    s = list(secondary_vals)
    if not p or not s or len(p) != len(s):
        return [], []
    n = len(p)
    step = max(1, n // 6)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    tickvals = [p[i] for i in idxs]
    ticktext = [str(s[i]) for i in idxs]
    return tickvals, ticktext
