from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Callable, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

from . import config as _config
from .grid import QubitGrid
from .overlays import Overlay
from . import typing as _typing

DataLike = _typing.DataLike
from .utils import label_from_attrs, compute_secondary_ticks, map_hue_value


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
    style_overrides: dict[str, Any]


class QualibrationFigure:

    def __init__(self):
        self._fig = go.Figure()

    @property
    def figure(self) -> go.Figure:
        return self._fig

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
        **style_overrides: Any,
    ) -> QualibrationFigure:
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
            **style_overrides,
        )
        return obj

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
        elif isinstance(data, xr.DataArray):
            return data.to_dataset(name=data.name or "value")
        elif hasattr(data, "to_xarray"):
            return data.to_xarray()
        elif isinstance(data, dict):
            return xr.Dataset({k: ("index", np.asarray(v)) for k, v in data.items()})
        else:
            raise TypeError("Unsupported data type for QualibrationFigure.plot")

    def _extract_plot_params(self, **kwargs) -> PlotParams:
        """Extract and validate plot parameters from kwargs.

        Args:
            **kwargs: Plot configuration parameters

        Returns:
            PlotParams: Container with all extracted plot parameters
        """
        x = kwargs.get("x")
        data_var = kwargs.get("data_var")
        y = kwargs.get("y")
        hue = kwargs.get("hue")
        x2 = kwargs.get("x2")
        qubit_dim = kwargs.get("qubit_dim", "qubit")
        qubit_names = kwargs.get("qubit_names")
        grid: QubitGrid = kwargs.get("grid")
        overlays = kwargs.get("overlays")
        residuals = kwargs.get("residuals", False)
        title = kwargs.get("title")
        style_overrides = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "x",
                "data_var",
                "y",
                "hue",
                "x2",
                "qubit_dim",
                "qubit_names",
                "grid",
                "overlays",
                "residuals",
                "title",
            }
        }

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
            residuals=residuals,
            title=title,
            style_overrides=style_overrides,
        )

    def _setup_subplot_grid(
        self,
        ds: xr.Dataset,
        qubit_dim: str,
        qubit_names: Sequence[str] | None,
        grid: QubitGrid | None,
        residuals: bool,
    ) -> tuple[Sequence[str], int, int, dict[str, tuple[int, int]]]:
        """Setup subplot grid and create figure with subplots.

        Args:
            ds: Dataset to extract qubit names from
            qubit_dim: Name of qubit dimension
            qubit_names: Optional list of qubit names
            grid: Optional QubitGrid configuration
            residuals: Whether to include residual subplots

        Returns:
            tuple: (qubit_names, n_rows, n_cols, positions)
        """
        if qubit_names is None and qubit_dim in ds.dims:
            qubit_names = list(map(str, ds.coords[qubit_dim].values))
        qubit_names = qubit_names or ["qubit"]

        if grid is None:
            coords = {name: (0, i) for i, name in enumerate(qubit_names)}
            grid = QubitGrid(coords, shape=(1, len(coords)))
        n_rows, n_cols, positions = grid.resolve(qubit_names)

        if residuals:
            total_rows = n_rows * 2
            rratio = float(
                getattr(_config.CURRENT_THEME, "residuals_height_ratio", 0.35)
            )
            main_h = max(0.0, min(1.0, 1.0 - rratio))
            row_heights = []
            for _ in range(n_rows):
                row_heights.extend([main_h, rratio])
            titles = [""] * (total_rows * n_cols)
            for name, (r, c) in positions.items():
                row_main = (r - 1) * 2 + 1
                idx = (row_main - 1) * n_cols + (c - 1)
                titles[idx] = name
            self._fig = make_subplots(
                rows=total_rows,
                cols=n_cols,
                subplot_titles=titles,
                row_heights=row_heights,
            )
        else:
            titles = [""] * (n_rows * n_cols)
            for name, (r, c) in positions.items():
                idx = (r - 1) * n_cols + (c - 1)
                titles[idx] = name
            self._fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)

        return qubit_names, n_rows, n_cols, positions

    def _get_row_indices(self, row: int, residuals: bool) -> tuple[int, int | None]:
        """Calculate row indices for main plot and residuals subplot.

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
            if "mode" in style_overrides:
                kwargs["mode"] = style_overrides["mode"]
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
            n_cols: Total number of columns
            style_overrides: Style override dictionary

        Returns:
            tuple: (x_vals, y_vals, xlab, ylab) for potential use in residuals
        """
        x_vals = sel.coords[x].values if x in sel.coords else np.asarray(sel[x].values)
        var = data_var or next(iter(sel.data_vars))
        y_vals = np.asarray(sel[var].values)

        if hue and hue in sel.dims:
            for hv in sel.coords[hue].values:
                y_h = np.asarray(sel[var].sel({hue: hv}).values)
                label = map_hue_value(hue, hv)
                scatter_kwargs = {
                    "x": x_vals,
                    "y": y_h,
                    "name": label,
                    "mode": "markers",
                    "marker": dict(size=_config.CURRENT_THEME.marker_size),
                    "line": dict(width=_config.CURRENT_THEME.line_width),
                }
                scatter_kwargs = self._apply_style_overrides(
                    scatter_kwargs, style_overrides, "scatter"
                )
                self._fig.add_trace(go.Scatter(**scatter_kwargs), row=row_main, col=col)
        else:
            scatter_kwargs = {
                "x": x_vals,
                "y": y_vals,
                "name": name,
                "mode": "markers",
                "marker": dict(size=_config.CURRENT_THEME.marker_size),
                "line": dict(width=_config.CURRENT_THEME.line_width),
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
            idx = (row_main - 1) * n_cols + col
            if idx == 1 and tickvals and ticktext:
                self._fig.update_layout(
                    xaxis2={
                        "overlaying": "x",
                        "side": "top",
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                    }
                )

        return x_vals, y_vals, xlab, ylab

    def _plot_2d_data(
        self,
        sel: xr.Dataset,
        x: str,
        y: str,
        data_var: str | None,
        x2: str | None,
        n_cols: int,
        row_main: int,
        col: int,
        style_overrides: dict[str, Any],
    ) -> None:
        """Plot 2D data (heatmap).

        Args:
            sel: Selected dataset for this qubit
            x: Name of x coordinate
            y: Name of y coordinate
            data_var: Name of data variable to plot
            x2: Optional secondary x coordinate
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
            idx = (row_main - 1) * n_cols + col
            if idx == 1 and tickvals and ticktext:
                # add a dummy trace linked to x2 so the top axis shows up
                self._fig.add_trace(
                    go.Scatter(
                        x=[min(tickvals), max(tickvals)],  # span the axis you want
                        y=[None, None],  # invisible
                        xaxis="x2",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                self._fig.update_layout(
                    xaxis2={
                        "overlaying": "x",
                        "side": "top",
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                        "showline": True,
                        "ticks": "outside",
                        "title": dict(
                            text=label_from_attrs(x2, sel.coords[x2].attrs), standoff=5
                        ),
                    }
                )
                self._fig.layout.annotations[0].update(y=1.2)

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

        for ov in panel_overlays:
            ov.add_to(
                self._fig,
                row=row_main,
                col=col,
                theme=_config.CURRENT_THEME,
                x=x_vals,
                **style_overrides,
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
            residual_kwargs = {
                "x": fit_x_vals if fit_x_vals is not None else x_vals,
                "y": residual_vals,
                "name": f"{name} residuals",
                "mode": "markers",
                "marker": dict(size=_config.CURRENT_THEME.marker_size),
                "line": dict(width=_config.CURRENT_THEME.line_width),
            }
            residual_kwargs = self._apply_style_overrides(
                residual_kwargs, style_overrides, "scatter"
            )

            self._fig.add_trace(go.Scatter(**residual_kwargs), row=row_resid, col=col)

            # Update x-axis label for residuals (should match main plot)
            self._fig.update_xaxes(title_text=xlab, row=row_resid, col=col)

    def _build(self, data: DataLike, **kwargs) -> None:
        # Normalize data to xarray.Dataset
        ds = self._normalize_data(data)

        # Extract parameters
        params = self._extract_plot_params(**kwargs)

        # Setup subplot grid
        qubit_names, n_rows, n_cols, positions = self._setup_subplot_grid(
            ds, params.qubit_dim, params.qubit_names, params.grid, params.residuals
        )

        # Main plotting loop
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
                    n_cols,
                    params.style_overrides,
                )
            else:
                self._plot_2d_data(
                    sel,
                    params.x,
                    params.y,
                    params.data_var,
                    params.x2,
                    n_cols,
                    row_main,
                    col,
                    params.style_overrides,
                )

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

        _config.apply_theme_to_layout(self._fig.layout)
        if _config.CURRENT_PALETTE:
            self._fig.update_layout(colorway=list(_config.CURRENT_PALETTE))
        if params.title:
            self._fig.update_layout(
                title=dict(
                    text=params.title, font=dict(size=_config.CURRENT_THEME.title_size)
                )
            )

        if _config.CURRENT_RC.values.get("showlegend") is not None:
            self._fig.update_layout(
                showlegend=bool(_config.CURRENT_RC.values["showlegend"])
            )
