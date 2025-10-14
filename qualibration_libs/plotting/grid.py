from __future__ import annotations

from typing import Dict, Tuple, Sequence, Iterator, Any
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
        
        self.fig = fig
        self._axes = axes
        self._ds = ds
        self._qubit_names = qubit_names
        self._positions = positions

    def resolve(self, present_qubits: Sequence[str]) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
        positions: Dict[str, Tuple[int, int]] = {}
        rows_max = 0
        cols_max = 0
        for q in present_qubits:
            if q not in self.coords:
                continue
            r, c = self.coords[q]
            r_idx, c_idx = r + 1, c + 1
            positions[q] = (r_idx, c_idx)
            rows_max = max(rows_max, r_idx)
            cols_max = max(cols_max, c_idx)

        if self.shape is not None:
            n_rows, n_cols = self.shape
        else:
            n_rows, n_cols = rows_max, cols_max

        return n_rows, n_cols, positions


def grid_iter(grid: QubitGrid) -> Iterator[Tuple[Any, Dict[str, str]]]:
    if not hasattr(grid, '_axes') or not hasattr(grid, '_qubit_names'):
        raise ValueError("grid_iter requires a QubitGrid created with the old interface (ds, grid_locations)")
    
    for qubit_name in grid._qubit_names:
        if qubit_name not in grid._positions:
            continue
        
        row, col = grid._positions[qubit_name]
        ax = grid._axes[row - 1, col - 1]
        
        qubit_info = {
            "qubit": qubit_name,
            "row": row - 1,
            "col": col - 1
        }
        
        yield ax, qubit_info