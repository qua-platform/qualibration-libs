from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Iterator
import matplotlib.pyplot as plt
import xarray as xr


class QubitGrid:
    def __init__(self, coords_or_ds=None, grid_locations=None, shape=None):
        if isinstance(coords_or_ds, dict):
            self.coords = coords_or_ds
            self.shape = shape
        elif isinstance(coords_or_ds, xr.Dataset) and grid_locations is not None:
            qubit_names = list(coords_or_ds.coords.get('qubit', []).values) if 'qubit' in coords_or_ds.coords else []
            if not qubit_names and hasattr(coords_or_ds, 'qubit'):
                qubit_names = list(coords_or_ds.qubit.values)
            
            coords = {}
            for i, loc in enumerate(grid_locations):
                if i < len(qubit_names):
                    qubit_name = str(qubit_names[i])
                    if isinstance(loc, str):
                        # Parse "col,row" format
                        col_str, row_str = loc.split(",")
                        coords[qubit_name] = (int(row_str), int(col_str))
                    elif isinstance(loc, (tuple, list)) and len(loc) == 2:
                        coords[qubit_name] = tuple(loc)
            
            self.coords = coords
            self.shape = shape
            self._create_matplotlib_figure(coords_or_ds, qubit_names)
        else:
            self.coords = coords_or_ds or {}
            self.shape = shape
    
    def _create_matplotlib_figure(self, ds: xr.Dataset, qubit_names: list):
        """Create matplotlib figure and axes for backward compatibility."""
        n_rows, n_cols, positions = self.resolve(qubit_names)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Turn off unused axes to match old behavior
        used_positions = set(positions.values())  # (row, col) are 1-based
        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                if (r, c) in used_positions:
                    continue
                try:
                    ax = axes[r - 1, c - 1]
                except Exception:
                    ax = axes[r - 1][c - 1]
                ax.axis("off")

        self.fig = fig
        self._axes = axes
        self._ds = ds
        self._qubit_names = qubit_names
        self._positions = positions

    def resolve(self, present_qubits: Sequence[str]) -> tuple[int, int, dict[str, tuple[int, int]]]:
        positions: dict[str, tuple[int, int]] = {}

        rows_min = None
        rows_max = None
        cols_min = None
        cols_max = None

        present_with_coords = []
        for q in present_qubits:
            if q not in self.coords:
                continue
            r, c = self.coords[q]
            present_with_coords.append((q, r, c))
            rows_min = r if rows_min is None else min(rows_min, r)
            rows_max = r if rows_max is None else max(rows_max, r)
            cols_min = c if cols_min is None else min(cols_min, c)
            cols_max = c if cols_max is None else max(cols_max, c)

        if rows_min is None or cols_min is None:
            return 0, 0, {}

        if self.shape is not None:
            n_rows, n_cols = self.shape
        else:
            n_rows = (rows_max - rows_min + 1)
            n_cols = (cols_max - cols_min + 1)

        for q, r, c in present_with_coords:
            r_norm = r - rows_min
            c_norm = c - cols_min
            r_flipped = (n_rows - 1) - r_norm
            positions[q] = (r_flipped + 1, c_norm + 1)

        return n_rows, n_cols, positions


def grid_iter(grid: QubitGrid) -> Iterator[tuple[Any, dict[str, str]]]:
    if not hasattr(grid, '_axes') or not hasattr(grid, '_qubit_names'):
        raise ValueError("grid_iter requires a QubitGrid created with the old interface (ds, grid_locations)")

    # Build position -> qubit_name mapping for quick lookup
    pos_to_qubit: dict[tuple[int, int], str] = {
        (row, col): qname for qname, (row, col) in getattr(grid, '_positions', {}).items()
    }

    if not pos_to_qubit:
        return

    # Determine grid bounds from positions
    max_row = max(p[0] for p in pos_to_qubit.keys())
    max_col = max(p[1] for p in pos_to_qubit.keys())

    # Determine the key name (match old interface using ds.qubit.name when available)
    key_name = 'qubit'
    if hasattr(grid, '_ds') and hasattr(grid._ds, 'qubit') and getattr(grid._ds.qubit, 'name', None):
        key_name = grid._ds.qubit.name  # e.g., 'qubit'

    # Iterate row-major over the axes and yield only used positions (backward-compatible order)
    for row in range(1, max_row + 1):
        for col in range(1, max_col + 1):
            qname = pos_to_qubit.get((row, col))
            if qname is None:
                continue
            # Support both numpy ndarray indexing and list-of-lists
            try:
                ax = grid._axes[row - 1, col - 1]
            except Exception:
                ax = grid._axes[row - 1][col - 1]
            yield ax, {key_name: qname}