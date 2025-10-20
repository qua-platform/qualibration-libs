from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Iterator
import matplotlib.pyplot as plt
import xarray as xr


class QubitGrid:
    """A grid layout manager for organizing qubit data in 2D spatial arrangements.

    QubitGrid maps qubit names to (row, column) positions in a grid and can
    automatically create matplotlib figure layouts for visualization. It supports
    initialization from coordinate dictionaries or xarray Datasets with grid locations.

    The `resolve` method provides grid dimension and position information that is used
    by both the matplotlib backend (via grid_iter) and the Plotly backend in
    ./plotting/figure.py for creating subplot layouts.

    Attributes:
        coords (dict): Mapping of qubit names to (row, col) tuples representing
            grid positions.
        shape (tuple[int, int] | None): Optional fixed (n_rows, n_cols) shape for
            the grid layout.
        fig (matplotlib.figure.Figure): Matplotlib figure (created lazily by grid_iter).
        _axes (numpy.ndarray): 2D array of matplotlib axes (created lazily by grid_iter).
        _qubit_names (list): List of qubit names (only when created from Dataset).
        _coord_name (str): Coordinate name for selectors (only when created from Dataset).
        _size (int): Size multiplier for figure dimensions (only when created from Dataset).
        _positions (dict): Resolved positions mapping (created lazily by grid_iter).

    Note:
        When created from a Dataset, the matplotlib figure is not created immediately.
        It is created lazily on the first call to grid_iter() for better performance.
    """

    def __init__(
        self,
        coords_or_ds: dict[str, tuple[int, int]] | xr.Dataset | None = None,
        grid_locations: list[str | tuple[int, int] | list[int]] | None = None,
        shape: tuple[int, int] | None = None,
        size: int = 3,
    ) -> None:
        """Initialize a QubitGrid from coordinates or an xarray Dataset.

        Args:
            coords_or_ds (dict | xr.Dataset | None): Either a dictionary mapping
                qubit names to (row, col) tuples, or an xarray Dataset containing
                qubit data. If None, creates an empty grid.
            grid_locations (list | None): List of grid locations corresponding to
                qubits in the Dataset. Each location can be:
                - A string in "col,row" format (e.g., "0,1")
                - A tuple/list of (col, row) integers
                Only used when coords_or_ds is a Dataset.
            shape (tuple[int, int] | None): Optional fixed (n_rows, n_cols) shape
                for the grid. If None, shape is automatically determined from the
                coordinate bounds.
            size (int): Size multiplier for matplotlib figure creation. The figure
                size will be (n_cols * size, n_rows * size). Default is 3.

        Examples:
            >>> # Create from coordinate dictionary (using row, col format)
            >>> grid = QubitGrid({'q0': (0, 0), 'q1': (0, 1), 'q2': (1, 0)})

            >>> # Create from Dataset with grid locations (col,row format)
            >>> grid = QubitGrid(ds, grid_locations=["0,0", "0,1", "1,0"])

            >>> # Or using tuples (col, row format)
            >>> grid = QubitGrid(ds, grid_locations=[(0, 0), (0, 1), (1, 0)])

            >>> # Create with custom figure size (each subplot will be 5x5 inches)
            >>> grid = QubitGrid(ds, grid_locations=["0,0", "0,1"], size=5)

            >>> # Create with fixed shape
            >>> grid = QubitGrid({'q0': (0, 0)}, shape=(3, 3))
        """
        if isinstance(coords_or_ds, dict):
            self.coords = coords_or_ds
            self.shape = shape
        elif isinstance(coords_or_ds, xr.Dataset) and grid_locations is not None:
            qubit_names = (
                list(coords_or_ds.coords.get("qubit", []).values)
                if "qubit" in coords_or_ds.coords
                else []
            )
            if not qubit_names and hasattr(coords_or_ds, "qubit"):
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
                        # Convert (col, row) to (row, col) for internal storage
                        coords[qubit_name] = (loc[1], loc[0])

            self.coords = coords
            self.shape = shape

            # Store minimal info for lazy matplotlib figure creation in grid_iter
            self._qubit_names = [str(q) for q in qubit_names]
            self._size = size

            # Determine the coordinate name for selectors
            self._coord_name = "qubit"
            if hasattr(coords_or_ds, "qubit") and getattr(
                coords_or_ds.qubit, "name", None
            ):
                self._coord_name = coords_or_ds.qubit.name
        else:
            self.coords = coords_or_ds or {}
            self.shape = shape

    def resolve(
        self, present_qubits: Sequence[str]
    ) -> tuple[int, int, dict[str, tuple[int, int]]]:
        """Resolve grid dimensions and normalized positions for a set of qubits.

        Calculates the minimum grid dimensions needed to accommodate the given
        qubits and returns their normalized 1-indexed positions. The grid coordinates
        are normalized by shifting to start from (1, 1) and rows are flipped so
        that higher row indices in the original coords appear at the top.

        Args:
            present_qubits (Sequence[str]): Sequence of qubit names to include
                in the grid. Only qubits with defined coordinates in self.coords
                will be included in the output.

        Returns:
            tuple[int, int, dict[str, tuple[int, int]]]: A tuple containing:
                - n_rows (int): Number of rows in the grid
                - n_cols (int): Number of columns in the grid
                - positions (dict): Mapping of qubit names to normalized (row, col)
                  positions (1-indexed). Returns (0, 0, {}) if no qubits have coords.

        Examples:
            >>> grid = QubitGrid({'q0': (0, 0), 'q1': (0, 2), 'q2': (1, 1)})
            >>> n_rows, n_cols, positions = grid.resolve(['q0', 'q1', 'q2'])
            >>> print(n_rows, n_cols)
            2 3
            >>> print(positions)
            {'q0': (2, 1), 'q1': (2, 3), 'q2': (1, 2)}
        """
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
            n_rows = rows_max - rows_min + 1
            n_cols = cols_max - cols_min + 1

        for q, r, c in present_with_coords:
            r_norm = r - rows_min
            c_norm = c - cols_min
            r_flipped = (n_rows - 1) - r_norm
            positions[q] = (r_flipped + 1, c_norm + 1)

        return n_rows, n_cols, positions


def grid_iter(grid: QubitGrid) -> Iterator[tuple[Any, dict[str, str]]]:
    """Iterate over matplotlib axes and their corresponding qubit selectors in grid order.

    Yields axes and qubit selectors for each occupied position in the grid,
    iterating in row-major order (left-to-right, top-to-bottom). This function
    creates the matplotlib figure on first call (lazy initialization).

    Args:
        grid (QubitGrid): A QubitGrid instance created with an xarray Dataset
            and grid_locations (i.e., having _qubit_names attribute).

    Yields:
        tuple[matplotlib.axes.Axes, dict[str, str]]: A tuple containing:
            - ax: The matplotlib Axes object for this grid position
            - selector: A dictionary mapping coordinate name to qubit name,
              e.g., {'qubit': 'q0'}

    Raises:
        ValueError: If the grid was not created with the Dataset interface
            (i.e., missing _qubit_names attribute).

    Examples:
        >>> grid = QubitGrid(ds, grid_locations=["0,0", "0,1", "1,0"])
        >>> for ax, selector in grid_iter(grid):
        ...     # Plot data for this qubit on this axis
        ...     data = ds.sel(selector)
        ...     ax.plot(data)

    Note:
        This function requires a QubitGrid created from an xarray Dataset
        with grid_locations, not from a simple coordinate dictionary.
        The matplotlib figure is created lazily on the first call.
    """
    if not hasattr(grid, "_qubit_names"):
        raise ValueError(
            "grid_iter requires a QubitGrid created with the Dataset interface (ds, grid_locations)"
        )

    # Create matplotlib figure lazily if not already created
    if not hasattr(grid, "_axes"):
        n_rows, n_cols, positions = grid.resolve(grid._qubit_names)

        # Get size parameter for figure dimensions
        size = getattr(grid, "_size", 3)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * size, n_rows * size), squeeze=False
        )

        # Turn off unused axes
        used_positions = set(positions.values())  # (row, col) are 1-based
        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                if (r, c) in used_positions:
                    continue
                ax = axes[r - 1][c - 1]
                ax.axis("off")

        # Cache the figure and axes
        grid.fig = fig
        grid._axes = axes
        grid._positions = positions

    # Build position -> qubit_name mapping for quick lookup
    pos_to_qubit: dict[tuple[int, int], str] = {
        (row, col): qname for qname, (row, col) in grid._positions.items()
    }

    if not pos_to_qubit:
        return

    # Determine grid bounds from positions
    max_row = max(p[0] for p in pos_to_qubit.keys())
    max_col = max(p[1] for p in pos_to_qubit.keys())

    # Get the coordinate name for selectors
    key_name = getattr(grid, "_coord_name", "qubit")

    # Iterate row-major over the axes and yield only used positions (backward-compatible order)
    for row in range(1, max_row + 1):
        for col in range(1, max_col + 1):
            qname = pos_to_qubit.get((row, col))
            if qname is None:
                continue
            # Works for both numpy ndarray and list-of-lists
            ax = grid._axes[row - 1][col - 1]
            yield ax, {key_name: qname, 'row': row, 'col': col}
