from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from . import config as _config
from . import typing as _typing
from .grid import QubitGrid
from .overlays import Overlay
from .utils import compute_secondary_ticks, label_from_attrs, map_hue_value

DataLike = _typing.DataLike


def _set_palette_from_value(palette):
    """Helper function to set palette from string name or color list.
    
    Args:
        palette: Palette name (string) or color list/tuple
    """
    if isinstance(palette, str):
        # Handle predefined palette names
        palette_map = {
            'viridis': ('#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'),
            'plasma': ('#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#f0f921'),
            'tab10': ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'),
            'tab20': ('#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d3', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'),
            'set1': ('#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'),
            'set2': ('#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'),
            'set3': ('#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'),
            'pastel1': ('#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2'),
            'pastel2': ('#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc'),
            'dark2': ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'),
            'paired': ('#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'),
            'accent': ('#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666'),
            'spectral': ('#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'),
            'coolwarm': ('#3b4cc0', '#6884d1', '#8fa3e0', '#b5c2ed', '#d9e0f7', '#f0f2fa', '#f7f8fc', '#fefefe', '#fef7f7', '#fce8e8', '#f5d0d0', '#e8b8b8', '#d6a0a0', '#c08888', '#a87070', '#905858', '#784040', '#602828', '#481010', '#300000'),
            'rdylbu': ('#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'),
            'rdylgn': ('#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9641', '#006837'),
            'rdbu': ('#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'),
            'piyg': ('#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'),
            'prgn': ('#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'),
            'brbg': ('#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30'),
            'puor': ('#7f3b08', '#b35806', '#e08214', '#fdb863', '#fee0b6', '#f7f7f7', '#d8daeb', '#b2abd2', '#8073ac', '#542788', '#2d004b'),
            'rdgy': ('#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#ffffff', '#e0e0e0', '#bababa', '#878787', '#4d4d4d', '#1a1a1a'),
            'terrain': ('#00a2ed', '#00c4ed', '#00e6ed', '#00ffed', '#00ffcc', '#00ffaa', '#00ff88', '#00ff66', '#00ff44', '#00ff22', '#00ff00', '#22ff00', '#44ff00', '#66ff00', '#88ff00', '#aaff00', '#ccff00', '#eeff00', '#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff3300', '#ff0000'),
            'ocean': ('#000080', '#0000aa', '#0000d4', '#0000ff', '#1a1aff', '#3333ff', '#4d4dff', '#6666ff', '#8080ff', '#9999ff', '#b3b3ff', '#ccccff', '#e6e6ff', '#ffffff'),
            'rainbow': ('#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00', '#00ff80', '#00ffff', '#0080ff', '#0000ff', '#8000ff', '#ff00ff', '#ff0080'),
        }
        if palette.lower() in palette_map:
            _config.CURRENT_PALETTE = palette_map[palette.lower()]
        else:
            raise ValueError(f"Unknown palette name: {palette}. Available palettes: {list(palette_map.keys())}")
    else:
        # Handle tuple/list of colors
        _config.CURRENT_PALETTE = tuple(palette) if isinstance(palette, list) else palette


@dataclass
class PlotParams:
    """Container for plot parameters extracted from kwargs.

    Attributes:
        x: Name of x coordinate
        data_var: Name of data variable to plot
        y: Name of y coordinate (None for 1D plots)
        hue: Optional hue dimension for grouping
        x2: Optional secondary x coordinate
        qubit_dim: Name of qubit dimension
        qubit_names: Optional list of qubit names
        grid: Optional QubitGrid configuration
        overlays: Optional overlays specification
        residuals: Whether to include residual subplots
        title: Optional plot title
        colorbar_tolerance: Tolerance for heatmap colorbar optimization
        vertical_spacing: Optional vertical spacing between subplot rows
        horizontal_spacing: Optional horizontal spacing between subplot columns
        style_overrides: Dictionary of style overrides
    """

    x: str
    data_var: str | None
    y: str | None
    hue: str | None
    x2: str | None
    qubit_dim: str
    qubit_names: Sequence[str] | None
    grid: QubitGrid | None
    overlays: (
        Sequence[Overlay]
        | dict[str, Sequence[Overlay]]
        | Callable[[str, Any], Sequence[Overlay]]
    ) | None
    residuals: bool
    title: str | None
    colorbar_tolerance: float
    vertical_spacing: float | None
    horizontal_spacing: float | None
    style_overrides: dict[str, Any]


class QualibrationFigure:
    """Interactive Plotly-based figure for qualibrate calibration data visualization.

    This class provides a high-level interface for creating multi-panel plots of qualibrate
    calibration data, supporting both 1D (line/scatter) and 2D (heatmap) visualizations.
    It handles qubit grid layouts, dual x-axes, overlays (fit curves, markers), and
    residual subplots.

    The class is designed to work seamlessly with xarray Datasets containing calibration
    measurements, automatically managing subplot layouts and axis configurations.

    Examples
    --------
    Basic 1D plot with dual x-axes:

    >>> # dataset is an xr.Dataset with dimensions ['qubit', 'amp_prefactor']
    >>> # and coordinates 'amp_prefactor', 'amp_mV', and data variable 'I'
    >>> fig = QualibrationFigure.plot(
    ...     dataset,
    ...     x='amp_prefactor',
    ...     x2='amp_mV',
    ...     data_var='I',
    ...     qubit_dim='qubit',
    ...     title='Power Rabi Calibration'
    ... )
    >>> fig.figure.show()

    2D heatmap with overlays:

    >>> # dataset is an xr.Dataset with dimensions ['qubit', 'detuning', 'power']
    >>> # and data variable 'IQ_abs_norm'
    >>> def create_overlays(qubit_name, qubit_data):
    ...     return [RefLine(x=optimal_value, name="Optimal")]
    >>>
    >>> fig = QualibrationFigure.plot(
    ...     dataset,
    ...     x='detuning',
    ...     y='power',
    ...     data_var='IQ_abs_norm',
    ...     overlays=create_overlays,
    ...     robust=True
    ... )

    Multiple qubits with custom grid:

    >>> # dataset is an xr.Dataset with dimensions ['qubit', 'amp_prefactor']
    >>> # and data variable 'state'
    >>> # grid is a QubitGrid from plotting/grid.py defining the 2D layout
    >>> grid = QubitGrid(dataset, [q.grid_location for q in qubits])
    >>> fig = QualibrationFigure.plot(
    ...     dataset,
    ...     x='amp_prefactor',
    ...     data_var='state',
    ...     grid=grid,
    ...     qubit_dim='qubit'
    ... )

    Notes
    -----
    - All plots are created using the `plot` classmethod
    - The class automatically determines whether to create 1D or 2D plots based on the `y` parameter
    - If `grid` is None, all qubit subplots are arranged in a single row
    - Secondary x-axes (x2) are supported for showing related coordinates
    - Overlays can be added via callable, dict, or sequence of Overlay objects
    - Residuals can be computed automatically if overlays provide fit data

    See Also
    --------
    QubitGrid : Grid layout manager for multi-qubit visualizations
    Overlay : Base class for plot overlays (fit curves, markers, etc.)
    FitOverlay : Overlay for fitted curves
    RefLine : Overlay for reference lines
    """

    def __init__(self):
        self._fig = go.Figure()
        self._color_index = 0
        self._legend_shown: set[str] = set()

    @property
    def figure(self) -> go.Figure:
        return self._fig

    def reset_color_index(self) -> None:
        """Reset the internal color cycle index.

        Call this before starting a new logical group (e.g., a new subplot) to
        ensure colors start from the first palette color again.
        """
        self._color_index = 0

    def _next_color(self) -> str:
        palette = _config.CURRENT_PALETTE or _config.CURRENT_THEME.colorway
        color = palette[self._color_index % len(palette)]
        self._color_index += 1
        return color

    @classmethod
    def plot(
        cls,
        data: DataLike,
        *,
        x: str,
        data_var: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        x2: str | None = None,
        qubit_dim: str = "qubit",
        qubit_names: Sequence[str] | None = None,
        grid: QubitGrid | None = None,
        overlays: (
            Sequence[Overlay]
            | dict[str, Sequence[Overlay]]
            | Callable[[str, Any], Sequence[Overlay]]
        ) | None = None,
        residuals: bool = False,
        title: str | None = None,
        colorbar_tolerance: float = 0.05,
        vertical_spacing: float | None = None,
        horizontal_spacing: float | None = None,
        **style_overrides: Any,
    ) -> QualibrationFigure:
        """Create an interactive calibration data plot with automatic layout.

        This is the main entry point for creating QualibrationFigure plots. It automatically
        determines whether to create 1D (line/scatter) or 2D (heatmap) plots based on whether
        the `y` parameter is provided, and handles multi-qubit layouts, dual x-axes, overlays,
        and residual subplots.

        Parameters
        ----------
        data : DataLike
            Input data to plot. Can be:
            - xr.Dataset: Multi-dimensional labeled dataset
            - xr.DataArray: Single data array (converted to Dataset)
            - dict: Dictionary of arrays (converted to Dataset)
            - Any object with a `to_xarray()` method
        x : str
            Name of the coordinate or variable to use for the x-axis. This is the primary
            x-axis and is required for all plots.
        data_var : str, optional
            Name of the data variable to plot. If None, uses the first data variable found
            in the dataset. For Datasets with multiple data variables, this should be specified.
        y : str, optional
            Name of the coordinate or variable to use for the y-axis. If provided, creates
            2D heatmap plots. If None (default), creates 1D line/scatter plots.
        hue : str, optional
            Name of a dimension to use for color grouping in 1D plots. Each unique value
            along this dimension will be plotted as a separate series with different colors.
            Only applicable for 1D plots (when y=None).
        x2 : str, optional
            Name of a secondary x-axis coordinate to display on top of the primary x-axis.
            Useful for showing related units or transformations (e.g., 'amp_prefactor' as
            primary and 'amp_mV' as secondary). Creates dual x-axes with automatic tick
            placement. If x2 is present and there are multiple rows of subplots, vertical
            spacing is automatically increased to prevent overlap.
        qubit_dim : str, default="qubit"
            Name of the dimension in the dataset that represents different qubits. This
            dimension is used to create separate subplots for each qubit.
        qubit_names : Sequence[str], optional
            Explicit list of qubit names to plot. If None, all qubits found in the dataset
            along `qubit_dim` will be plotted. Use this to plot a subset of qubits or to
            control the plotting order.
        grid : QubitGrid, optional
            A QubitGrid instance from plotting/grid.py that defines the 2D layout of qubit
            subplots. If None, all qubit subplots are arranged in a single row. The grid
            allows you to arrange qubits in a custom 2D layout matching physical chip topology.
        overlays : Sequence[Overlay] or dict[str, Sequence[Overlay]] or Callable, optional
            Overlays to add to the plots (fit curves, reference lines, markers, etc.). Can be:
            - Sequence[Overlay]: Same overlays applied to all qubits
            - dict[str, Sequence[Overlay]]: Qubit-specific overlays (keys are qubit names)
            - Callable[[str, Any], Sequence[Overlay]]: Function that takes (qubit_name, qubit_data)
              and returns overlays for that qubit
            Common overlay types include FitOverlay, RefLine, LineOverlay, and ScatterOverlay.
        residuals : bool, default=False
            If True, adds residual subplots below each main plot. Residuals are automatically
            computed if overlays provide fit data (e.g., FitOverlay with y_fit attribute).
            The residual is calculated as data - fit.
        title : str, optional
            Overall title for the entire figure. If x2 is present, the title is automatically
            positioned higher to avoid overlap with secondary axes.
        colorbar_tolerance : float, default=0.05
            Tolerance for determining if multiple heatmaps have the same scaling for colorbar
            optimization. Heatmaps with min/max values differing by less than this fraction of
            the data range are considered to have "same scaling" and will share a single colorbar.
            The tolerance is calculated as max(colorbar_tolerance * range_size, 1e-6).
        vertical_spacing : float, optional
            Vertical spacing between subplot rows, passed through to ``plotly.subplots.make_subplots``.
            If None, a default value is chosen automatically (larger when a secondary x-axis is present).
        horizontal_spacing : float, optional
            Horizontal spacing between subplot columns, passed through to ``plotly.subplots.make_subplots``.
            If None, a default value is used.
        palette : str, list, tuple, optional
            Color palette to use for the plot. Can be:
            - A predefined palette name (e.g., 'viridis', 'plasma', 'tab10', 'set1', 'set2', 'set3')
            - A list/tuple of color strings (hex codes or named colors)
            - None (uses current global palette)
            The palette is temporarily set during plot creation and automatically restored afterward.
            See the README for a complete list of available predefined palettes.
        **style_overrides : Any
            Additional style parameters to customize plot appearance. Common overrides include:
            - marker_size : int - Size of scatter plot markers
            - line_width : float - Width of lines
            - color : str - Color for traces
            - mode : str - Plotly mode ('markers', 'lines', 'lines+markers')
            - colorscale : str - Colorscale for heatmaps ('Viridis', 'RdBu', etc.)
            - showscale : bool - Whether to show colorbar for heatmaps
        """

        # Extract palette from style_overrides if provided
        palette = style_overrides.pop('palette', None)
        
        # Temporarily set palette if provided
        original_palette = None
        if palette is not None:
            import qualibration_libs.plotting.config as _config
            original_palette = _config.CURRENT_PALETTE
            _set_palette_from_value(palette)
        
        try:
            obj = cls()
            obj._build(
                data,
                x=x,
                data_var=data_var,
                y=y,
                hue=hue,
                x2=x2,
                qubit_dim=qubit_dim,
                qubit_names=qubit_names,
                grid=grid,
                overlays=overlays,
                residuals=residuals,
                title=title,
                colorbar_tolerance=colorbar_tolerance,
                vertical_spacing=vertical_spacing,
                horizontal_spacing=horizontal_spacing,
                **style_overrides,
            )
            return obj
        finally:
            # Restore original palette
            if original_palette is not None:
                import qualibration_libs.plotting.config as _config
                _config.CURRENT_PALETTE = original_palette

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _extract_plot_params(
        self,
        *,
        x: str,
        data_var: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        x2: str | None = None,
        qubit_dim: str = "qubit",
        qubit_names: Sequence[str] | None = None,
        grid: QubitGrid | None = None,
        overlays: (
            Sequence[Overlay]
            | dict[str, Sequence[Overlay]]
            | Callable[[str, Any], Sequence[Overlay]]
        ) | None = None,
        residuals: bool = False,
        title: str | None = None,
        colorbar_tolerance: float = 0.05,
        vertical_spacing: float | None = None,
        horizontal_spacing: float | None = None,
        **style_overrides: Any,
    ) -> PlotParams:
        """Extract and validate plotting parameters into a structured container.

        Returns a PlotParams dataclass instance, handling defaults and passing
        through style overrides. This centralizes parameter handling for the
        builder and keeps the plot() API thin.
        """
        if not isinstance(x, str) or not x:
            raise KeyError("x coordinate must be provided")

        # Normalize style overrides to a plain dict
        style = dict(style_overrides) if style_overrides else {}

        return PlotParams(
            x=x,
            data_var=data_var,
            y=y,
            hue=hue,
            x2=x2,
            qubit_dim=qubit_dim,
            qubit_names=qubit_names,
            grid=grid,
            overlays=overlays,
            residuals=bool(residuals),
            title=title,
            colorbar_tolerance=colorbar_tolerance,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            style_overrides=style,
        )

    def _normalize_data(self, data: DataLike) -> xr.Dataset:
        """Convert various data types to xarray.Dataset.

        Args:
            data: Input data (xr.Dataset, xr.DataArray, dict, or object with to_xarray method)

        Returns:
            xr.Dataset: Normalized dataset

        Raises:
            TypeError: If data type is not supported
        """
        if isinstance(data, xr.Dataset):
            return data
        if isinstance(data, xr.DataArray):
            return data.to_dataset(name=data.name or "value")
        if isinstance(data, dict):
            # Convert dict to xarray Dataset
            # All arrays should have the same length for proper conversion
            arrays = {k: np.asarray(v) for k, v in data.items()}
            
            # Find the length of the first array to use as dimension
            first_key = next(iter(arrays.keys()))
            length = len(arrays[first_key])
            
            # Create dataset with all arrays as 1D data variables
            data_vars = {}
            for key, arr in arrays.items():
                if len(arr) == length:
                    data_vars[key] = ("index", arr)
                else:
                    # If lengths don't match, treat as coordinate
                    data_vars[key] = arr
            
            return xr.Dataset(data_vars)
        if hasattr(data, "to_xarray"):
            return data.to_xarray()
        raise TypeError("Unsupported data type for plotting")

    def _setup_subplot_grid(
        self,
        ds: xr.Dataset,
        qubit_dim: str,
        qubit_names: Sequence[str] | None,
        grid: QubitGrid | None,
        residuals: bool,
        x2: str | None,
        vertical_spacing: float | None,
        horizontal_spacing: float | None,
    ) -> tuple[Sequence[str], int, int, dict[str, tuple[int, int]]]:
        # Determine qubit names
        if qubit_names is None:
            if qubit_dim in ds.dims:
                qubit_names = [str(q) for q in ds[qubit_dim].values]
            else:
                qubit_names = ["q0"]

        # Determine grid shape
        if grid is None:
            n_cols = len(qubit_names)
            n_rows = 1
            positions = {q: (1, i + 1) for i, q in enumerate(qubit_names)}
        else:
            # Use QubitGrid.resolve to normalize to 1-based positions and compute shape
            n_rows, n_cols, positions = grid.resolve(qubit_names)

        subplot_titles = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        
        for qubit_name in qubit_names:
            if qubit_name in positions:
                row, col = positions[qubit_name]
                subplot_titles[row - 1][col - 1] = qubit_name
        flat_titles = [title for row in subplot_titles for title in row]

        # Determine spacing values, allowing explicit overrides from PlotParams
        if vertical_spacing is not None:
            vspace = vertical_spacing
        else:
            # Default: increase spacing when secondary x-axis is present and multiple rows
            vspace = 0.2 if (x2 and n_rows > 1) else 0.1

        hspace = horizontal_spacing if horizontal_spacing is not None else 0.05
        self._fig = make_subplots(
            rows=n_rows * (2 if residuals else 1),
            cols=n_cols,
            subplot_titles=flat_titles,
            vertical_spacing=vspace,
            horizontal_spacing=hspace,
        )
        return qubit_names, n_rows, n_cols, positions

    def _get_row_indices(self, row: int, residuals: bool) -> tuple[int, int | None]:
        """Get main and residual row indices for a given subplot row.

        Args:
            row: Original row number
            residuals: Whether residuals are enabled

        Returns:
            tuple: (row_main, row_resid) where row_resid is None if residuals disabled
        """
        if residuals:
            row_main = (row - 1) * 2 + 1
            row_resid = row_main + 1
        else:
            row_main = row
            row_resid = None
        return row_main, row_resid

    def _apply_style_overrides(
        self,
        base_kwargs: dict[str, Any],
        style_overrides: dict[str, Any],
        trace_type: str = "scatter",
    ) -> dict[str, Any]:
        """Apply style overrides to trace kwargs.

        Args:
            base_kwargs: Base kwargs for the trace
            style_overrides: Dictionary of style overrides
            trace_type: Type of trace ("scatter" or "heatmap")

        Returns:
            dict: Updated kwargs with style overrides applied
        """
        kwargs = base_kwargs.copy()

        if trace_type == "scatter":
            if "marker_size" in style_overrides:
                kwargs["marker"]["size"] = style_overrides["marker_size"]
            if "line_width" in style_overrides:
                kwargs["line"]["width"] = style_overrides["line_width"]
            if "color" in style_overrides:
                kwargs["marker"]["color"] = style_overrides["color"]
                kwargs["line"]["color"] = style_overrides["color"]
            if "mode" in style_overrides:
                kwargs["mode"] = style_overrides["mode"]
            # Legend control if provided
            if "legendgroup" in style_overrides:
                kwargs["legendgroup"] = style_overrides["legendgroup"]
            if "showlegend" in style_overrides:
                kwargs["showlegend"] = style_overrides["showlegend"]
        elif trace_type == "heatmap":
            if "colorscale" in style_overrides:
                kwargs["colorscale"] = style_overrides["colorscale"]
            if "colorbar" in style_overrides:
                kwargs["colorbar"].update(style_overrides["colorbar"])
            if "showscale" in style_overrides:
                kwargs["showscale"] = style_overrides["showscale"]
            if "robust" in style_overrides and style_overrides["robust"] is True:
                z_vals = kwargs["z"]
                zmin, zmax = np.percentile(z_vals, [2, 98])
                kwargs["zmin"] = zmin
                kwargs["zmax"] = zmax

        return kwargs

    def _plot_1d_data(
        self,
        sel: xr.Dataset,
        x: str,
        data_var: str | None,
        hue: str | None,
        x2: str | None,
        name: str,
        row_main: int,
        col: int,
        n_rows: int,
        n_cols: int,
        style_overrides: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        """Plot 1D data (line/scatter plot).

        Args:
            sel: Selected dataset for this qubit
            x: Name of x coordinate
            data_var: Name of data variable to plot
            hue: Optional hue dimension for grouping
            x2: Optional secondary x coordinate
            name: Qubit name
            row_main: Main plot row index
            col: Column index
            n_rows: Total number of rows
            n_cols: Total number of columns
            style_overrides: Style override dictionary

        Returns:
            tuple: (x_vals, y_vals, xlab, ylab) for potential use in residuals
        """
        x_vals = sel.coords[x].values if x in sel.coords else np.asarray(sel[x].values)
        var = data_var or next(iter(sel.data_vars))
        y_vals = np.asarray(sel[var].values)

        # Reset color cycle at the start of each subplot to keep colors consistent
        self.reset_color_index()

        if hue and hue in sel.dims:
            for hv in sel.coords[hue].values:
                y_h = np.asarray(sel[var].sel({hue: hv}).values)
                label = map_hue_value(hue, hv)
                color = self._next_color()
                # Show each hue label only once across all subplots
                show_lgd = label not in self._legend_shown
                self._legend_shown.add(label)
                scatter_kwargs = {
                    "x": x_vals,
                    "y": y_h,
                    "name": label,
                    "mode": "markers",
                    "marker": dict(size=_config.CURRENT_THEME.marker_size, color=color),
                    "line": dict(width=_config.CURRENT_THEME.line_width, color=color),
                    "legendgroup": label,
                    "showlegend": show_lgd,
                }
                scatter_kwargs = self._apply_style_overrides(
                    scatter_kwargs, style_overrides, "scatter"
                )
                self._fig.add_trace(go.Scatter(**scatter_kwargs), row=row_main, col=col)
        else:
            color = self._next_color()
            scatter_kwargs = {
                "x": x_vals,
                "y": y_vals,
                "name": name,
                "mode": "markers",
                "marker": dict(size=_config.CURRENT_THEME.marker_size, color=color),
                "line": dict(width=_config.CURRENT_THEME.line_width, color=color),
                "showlegend": False,  # Hide data traces from legend - only show overlays
            }
            scatter_kwargs = self._apply_style_overrides(
                scatter_kwargs, style_overrides, "scatter"
            )
            self._fig.add_trace(go.Scatter(**scatter_kwargs), row=row_main, col=col)

        xlab = label_from_attrs(x, (sel.coords[x].attrs if x in sel.coords else {}))
        ylab = label_from_attrs(
            var, sel[var].attrs if hasattr(sel[var], "attrs") else {}
        )
        self._fig.update_xaxes(title_text=xlab, row=row_main, col=col)
        self._fig.update_yaxes(title_text=ylab, row=row_main, col=col)

        if x2 and x2 in sel.coords:
            xv2 = np.asarray(sel.coords[x2].values)
            tickvals, ticktext = compute_secondary_ticks(x_vals, xv2)

            if tickvals and ticktext:
                # Calculate axis indices for this subplot
                idx = (row_main - 1) * n_cols + col
                total_subplots = n_rows * n_cols
                axis_idx_secondary = idx + total_subplots

                # Build axis names
                primary_x_ref = f"x{idx}" if idx > 1 else "x"
                secondary_x_name = f"xaxis{axis_idx_secondary}"
                secondary_x_ref = f"x{axis_idx_secondary}"
                y_axis_ref = f"y{idx}" if idx > 1 else "y"

                # Add dummy trace to activate the secondary axis
                self._fig.add_trace(
                    go.Scatter(
                        x=[min(tickvals), max(tickvals)],
                        y=[None, None],
                        xaxis=secondary_x_ref,
                        yaxis=y_axis_ref,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Create layout configuration for this secondary axis
                layout_config = {
                    secondary_x_name: {
                        "overlaying": primary_x_ref,
                        "side": "top",
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                        "tickangle": 0,
                        "showline": True,
                        "ticks": "outside",
                        "anchor": y_axis_ref,
                        "title": dict(
                            text=label_from_attrs(x2, sel.coords[x2].attrs), standoff=5
                        ),
                    }
                }
                self._fig.update_layout(**layout_config)

        return x_vals, y_vals, xlab, ylab

    def _plot_2d_data(
        self,
        sel: xr.Dataset,
        x: str,
        y: str,
        data_var: str | None,
        x2: str | None,
        n_rows: int,
        n_cols: int,
        row_main: int,
        col: int,
        style_overrides: dict[str, Any],
    ) -> tuple[float, float]:
        """Plot 2D data (heatmap).

        Args:
            sel: Selected dataset for this qubit
            x: Name of x coordinate
            y: Name of y coordinate
            data_var: Name of data variable to plot
            x2: Optional secondary x coordinate
            n_rows: Total number of rows
            n_cols: Total number of columns
            row_main: Main plot row index
            col: Column index
            style_overrides: Style override dictionary
        """
        var = data_var or next(iter(sel.data_vars))

        # find actual dimensions from auxiliary coordinate
        x_dim = sel.coords[x].dims[0]
        y_dim = sel.coords[y].dims[0]

        # transpose the data to match the plotting axis
        z_vals = np.asarray(sel[var].transpose(y_dim, x_dim).values)
        x_vals = np.asarray(sel.coords[x].values)
        y_vals = np.asarray(sel.coords[y].values)

        heatmap_kwargs = {
            "x": x_vals,
            "y": y_vals,
            "z": z_vals,
            "colorbar": dict(title=var),
            "colorscale": "Viridis",
        }
        heatmap_kwargs = self._apply_style_overrides(
            heatmap_kwargs, style_overrides, "heatmap"
        )

        self._fig.add_trace(go.Heatmap(**heatmap_kwargs), row=row_main, col=col)
        xlab = label_from_attrs(x, sel.coords[x].attrs)
        ylab = label_from_attrs(y, sel.coords[y].attrs)
        self._fig.update_xaxes(title_text=xlab, row=row_main, col=col)
        self._fig.update_yaxes(title_text=ylab, row=row_main, col=col)

        # Add secondary x-axis if specified
        # The second axis only shows up if a trace is attached to it
        if x2 and x2 in sel.coords:
            xv2 = np.asarray(sel.coords[x2].values)
            tickvals, ticktext = compute_secondary_ticks(x_vals, xv2)

            if tickvals and ticktext:
                # Calculate axis indices for this subplot
                # Primary axis index: subplots are numbered from 1
                idx = (row_main - 1) * n_cols + col
                # Secondary axis index: after all primary axes
                total_subplots = n_rows * n_cols
                axis_idx_secondary = idx + total_subplots

                # Build axis names
                primary_x_ref = f"x{idx}" if idx > 1 else "x"
                secondary_x_name = f"xaxis{axis_idx_secondary}"
                secondary_x_ref = f"x{axis_idx_secondary}"
                y_axis_ref = f"y{idx}" if idx > 1 else "y"

                # Add dummy trace to activate the secondary axis
                self._fig.add_trace(
                    go.Scatter(
                        x=[min(tickvals), max(tickvals)],
                        y=[None, None],
                        xaxis=secondary_x_ref,
                        yaxis=y_axis_ref,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Create layout configuration for this secondary axis
                layout_config = {
                    secondary_x_name: {
                        "overlaying": primary_x_ref,
                        "side": "top",
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                        "tickangle": 0,
                        "showline": True,
                        "ticks": "outside",
                        "anchor": y_axis_ref,
                        "title": dict(
                            text=label_from_attrs(x2, sel.coords[x2].attrs), standoff=5
                        ),
                    }
                }
                self._fig.update_layout(**layout_config)
        
        # Calculate and return scaling information for colorbar optimization
        z_min = float(np.min(z_vals))
        z_max = float(np.max(z_vals))
        return z_min, z_max

    def _add_overlays(
        self,
        overlays: Any,
        name: str,
        sel: xr.Dataset,
        x_vals: np.ndarray | None,
        row_main: int,
        col: int,
        style_overrides: dict[str, Any],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Add overlays to the plot and track fit data for residuals.

        Args:
            overlays: Overlay specification (callable, dict, or sequence)
            name: Qubit name
            sel: Selected dataset for this qubit
            x_vals: X values for overlay
            row_main: Main plot row index
            col: Column index
            style_overrides: Style override dictionary

        Returns:
            tuple: (fit_data, fit_x_vals) for residuals calculation
        """
        fit_data = None
        fit_x_vals = None

        panel_overlays: Sequence[Overlay]
        if callable(overlays):
            panel_overlays = overlays(name, sel)
        elif isinstance(overlays, dict):
            panel_overlays = overlays.get(name, [])
        else:
            panel_overlays = overlays

        # Track overlay index per subplot for consistent color assignment
        overlay_index = 0
        
        for ov in panel_overlays:
            # Deduplicate legend entries across subplots by legendgroup
            group_label = getattr(ov, "name", None) or "overlay"
            show_lgd = group_label not in self._legend_shown
            self._legend_shown.add(group_label)

            # Assign a color if not overridden by style or overlay-specific color
            ov_style = dict(style_overrides)
            # If overlay defines its own color attribute, use it by default
            if "color" not in ov_style and hasattr(ov, "color") and getattr(ov, "color") is not None:
                ov_style["color"] = ov.color

            if "color" not in ov_style:
                # Use a distinct color for overlays by using a different color scheme
                # For small palettes, use darker/lighter variants or different colors
                palette = _config.CURRENT_PALETTE or _config.CURRENT_THEME.colorway

                if len(palette) >= 4:
                    # Large palette: skip first 2 entries (reserved for data traces)
                    # and use subsequent colors per overlay within this subplot.
                    base_index = 2
                    ov_style["color"] = palette[
                        (base_index + overlay_index) % len(palette)
                    ]
                else:
                    # Small palette: use distinct colors that won't conflict with data
                    # Use colors from a different part of the palette or fallback colors
                    overlay_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]  # Green, Orange, Red, Purple
                    ov_style["color"] = overlay_colors[overlay_index % len(overlay_colors)]
            
            # Increment overlay index for next overlay in this subplot
            overlay_index += 1
            # Pass legend grouping to overlay implementation
            ov_style["legendgroup"] = group_label
            ov_style["showlegend"] = show_lgd

            ov.add_to(
                self._fig,
                row=row_main,
                col=col,
                theme=_config.CURRENT_THEME,
                x=x_vals,
                **ov_style,
            )

            # Check if this overlay provides fit data for residuals
            if hasattr(ov, "y_fit") and ov.y_fit is not None:
                fit_data = ov.y_fit
                fit_x_vals = x_vals

        return fit_data, fit_x_vals

    def _add_residuals(
        self,
        name: str,
        x_vals: np.ndarray | None,
        y_vals: np.ndarray | None,
        fit_data: np.ndarray | None,
        fit_x_vals: np.ndarray | None,
        xlab: str | None,
        row_resid: int,
        col: int,
        style_overrides: dict[str, Any],
    ) -> None:
        """Add residual subplot.

        Args:
            name: Qubit name
            x_vals: X values for residuals
            y_vals: Y values for residuals
            fit_data: Fit data from overlays
            fit_x_vals: X values for fit data
            xlab: X axis label
            row_resid: Residual subplot row index
            col: Column index
            style_overrides: Style override dictionary
        """
        # Add zero reference line
        self._fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            xref="paper",
            yref="y",
            line={"dash": "dot"},
            row=row_resid,
            col=col,
        )
        self._fig.update_yaxes(title_text="Residuals", row=row_resid, col=col)

        # Plot residuals if we have fit data
        if fit_data is not None:
            residual_vals = y_vals - fit_data
            color = self._next_color()
            residual_kwargs = {
                "x": fit_x_vals if fit_x_vals is not None else x_vals,
                "y": residual_vals,
                "name": f"{name} residuals",
                "mode": "markers",
                "marker": dict(size=_config.CURRENT_THEME.marker_size, color=color),
                "line": dict(width=_config.CURRENT_THEME.line_width, color=color),
            }
            residual_kwargs = self._apply_style_overrides(
                residual_kwargs, style_overrides, "scatter"
            )

            self._fig.add_trace(go.Scatter(**residual_kwargs), row=row_resid, col=col)

            # Update x-axis label for residuals (should match main plot)
            self._fig.update_xaxes(title_text=xlab, row=row_resid, col=col)

    def _optimize_colorbars(self, scaling_info: list[tuple[float, float, int, int]], tolerance: float = 0.05) -> None:
        """Optimize colorbar display for multiple heatmaps.
        
        Args:
            scaling_info: List of (z_min, z_max, row, col) tuples for each heatmap
            tolerance: Fraction of data range for determining same scaling (default: 0.05 = 5%)
        """
        if len(scaling_info) <= 1:
            # For single heatmap, ensure colorbar is shown
            heatmap_traces = [trace for trace in self._fig.data if trace.type == 'heatmap']
            for trace in heatmap_traces:
                trace.showscale = True
            return
        
        # Check if all heatmaps have the same scaling (within tolerance)
        # For tolerance >= 1.0, always consider as same scaling (force colorbars)
        if tolerance >= 1.0:
            all_same_scaling = True
        else:
            # Use a more reasonable tolerance for real data (tolerance% of the range)
            first_z_min, first_z_max = scaling_info[0][:2]
            range_size = first_z_max - first_z_min
            tolerance_value = max(tolerance * range_size, 1e-6)  # tolerance% of range or 1e-6, whichever is larger
            
            all_same_scaling = all(
                abs(z_min - first_z_min) < tolerance_value and abs(z_max - first_z_max) < tolerance_value
                for z_min, z_max, _, _ in scaling_info
            )
        
        # Get all heatmap traces (in order they were added)
        heatmap_traces = [trace for trace in self._fig.data if trace.type == 'heatmap']
        
        if not heatmap_traces:
            # No heatmaps found, nothing to do
            return
        
        if all_same_scaling:
            # Same scaling: show only one colorbar (on the last subplot)
            for i, trace in enumerate(heatmap_traces):
                # Show colorbar only on the last heatmap trace
                show_colorbar = (i == len(heatmap_traces) - 1)
                trace.showscale = show_colorbar
                
                # For the last trace with colorbar, ensure it's positioned correctly
                if show_colorbar:
                    # Position colorbar to the right of the plot area
                    trace.colorbar.x = 1.02
                    trace.colorbar.xanchor = "left"
                    trace.colorbar.y = 0.5
                    trace.colorbar.yanchor = "middle"
            
            # Note: Margin adjustment moved to end of _build method
        else:
            # Different scaling: hide all colorbars
            for trace in heatmap_traces:
                trace.showscale = False

    def _build(self, data: DataLike, **kwargs) -> None:
        # Normalize data to xarray.Dataset
        ds = self._normalize_data(data)

        # Extract parameters
        params = self._extract_plot_params(**kwargs)

        # Setup subplot grid
        qubit_names, n_rows, n_cols, positions = self._setup_subplot_grid(
            ds,
            params.qubit_dim,
            params.qubit_names,
            params.grid,
            params.residuals,
            params.x2,
            params.vertical_spacing,
            params.horizontal_spacing,
        )

        # Main plotting loop
        scaling_info = []  # Collect scaling information for colorbar optimization
        
        for name in qubit_names:
            if name not in positions:
                continue

            row, col = positions[name]
            row_main, row_resid = self._get_row_indices(row, params.residuals)
            sel = (
                ds.sel({params.qubit_dim: name}) if params.qubit_dim in ds.dims else ds
            )

            # Initialize fit data tracking
            fit_data = None
            fit_x_vals = None
            x_vals = None
            y_vals = None
            xlab = None
            ylab = None

            # Plot data (1D or 2D)
            if params.y is None:
                x_vals, y_vals, xlab, ylab = self._plot_1d_data(
                    sel,
                    params.x,
                    params.data_var,
                    params.hue,
                    params.x2,
                    name,
                    row_main,
                    col,
                    n_rows,
                    n_cols,
                    params.style_overrides,
                )
            else:
                z_min, z_max = self._plot_2d_data(
                    sel,
                    params.x,
                    params.y,
                    params.data_var,
                    params.x2,
                    n_rows,
                    n_cols,
                    row_main,
                    col,
                    params.style_overrides,
                )
                scaling_info.append((z_min, z_max, row_main, col))

            # Add overlays if specified
            if params.overlays:
                # Use x_vals from 1D plot if available
                x_vals_for_overlay = x_vals if x_vals is not None else None
                fit_data, fit_x_vals = self._add_overlays(
                    params.overlays,
                    name,
                    sel,
                    x_vals_for_overlay,
                    row_main,
                    col,
                    params.style_overrides,
                )

            # Add residuals if enabled
            if params.residuals and row_resid is not None:
                self._add_residuals(
                    name,
                    x_vals,
                    y_vals,
                    fit_data,
                    fit_x_vals,
                    xlab,
                    row_resid,
                    col,
                    params.style_overrides,
                )

        # Optimize colorbar display for 2D heatmaps
        if scaling_info:
            self._optimize_colorbars(scaling_info, params.colorbar_tolerance)

        _config.apply_theme_to_layout(self._fig.layout)
        if _config.CURRENT_PALETTE:
            self._fig.update_layout(colorway=list(_config.CURRENT_PALETTE))

        # If x2 is present, add top margin and adjust subplot titles
        if params.x2:
            # Get configurable margin and annotation offset from style_overrides
            top_margin = params.style_overrides.get('x2_top_margin', 120)
            annotation_offset = params.style_overrides.get('x2_annotation_offset', 0.08)
            
            # Increase top margin to accommodate secondary x-axis and title
            self._fig.update_layout(margin=dict(t=top_margin))

            # Adjust all subplot title positions to avoid overlap with secondary axes
            # This is done after all plotting to ensure all annotations exist
            for annotation in self._fig.layout.annotations:
                if annotation.y is not None and annotation.text:
                    # Move all subplot title annotations higher to avoid overlap with secondary x-axis
                    annotation.update(y=annotation.y + annotation_offset)

        if params.title:
            title_config = dict(
                text=params.title, font=dict(size=_config.CURRENT_THEME.title_size)
            )
            # If x2 is present, move title up within the expanded margin space
            if params.x2:
                title_config["y"] = 0.98
                title_config["yanchor"] = "top"
            self._fig.update_layout(title=title_config)

        if _config.CURRENT_RC.values.get("showlegend") is not None:
            self._fig.update_layout(
                showlegend=bool(_config.CURRENT_RC.values["showlegend"])
            )
        
        # Final margin adjustment for colorbar (if needed)
        # Check if any heatmap traces have showscale=True and adjust margin accordingly
        heatmap_traces = [trace for trace in self._fig.data if trace.type == 'heatmap']
        if heatmap_traces and any(trace.showscale for trace in heatmap_traces):
            # Get current margin settings and add right margin for colorbar
            current_margin = self._fig.layout.margin
            if current_margin:
                # Preserve existing margins and add right margin for colorbar
                margin_dict = dict(
                    t=current_margin.t or 60,
                    b=current_margin.b or 60, 
                    l=current_margin.l or 60,
                    r=100  # Right margin for colorbar
                )
            else:
                margin_dict = dict(r=100)
            self._fig.update_layout(margin=margin_dict)
