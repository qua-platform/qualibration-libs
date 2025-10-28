from __future__ import annotations

from typing import Optional, Sequence, Callable, Any

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
        data_var: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        x2: Optional[str] = None,
        qubit_dim: str = "qubit",
        qubit_names: Optional[Sequence[str]] = None,
        grid: Optional[QubitGrid] = None,
        overlays: Optional[
            Sequence[Overlay] | dict[str, Sequence[Overlay]]
            | Callable[[str, Any], Sequence[Overlay]]
            ] = None,
        residuals: bool = False,
        title: Optional[str] = None,
        **style_overrides,
    ) -> "QualibrationFigure":
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

    def _build(self, data: DataLike, **kwargs) -> None:
        # Initialize color tracking per subplot
        self._subplot_color_counters = {}
        
        if isinstance(data, xr.Dataset):
            ds = data
        elif isinstance(data, xr.DataArray):
            ds = data.to_dataset(name=data.name or "value")
        elif hasattr(data, "to_xarray"):
            ds = data.to_xarray()
        elif isinstance(data, dict):
            ds = xr.Dataset({k: ("index", np.asarray(v)) for k, v in data.items()})
        else:
            raise TypeError("Unsupported data type for QualibrationFigure.plot")

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
        style_overrides = {k: v for k, v in kwargs.items() if
                           k not in {"x", "data_var", "y", "hue", "x2", "qubit_dim", "qubit_names", "grid", "overlays",
                                     "residuals", "title"}}

        if qubit_names is None and qubit_dim in ds.dims:
            qubit_names = list(map(str, ds.coords[qubit_dim].values))
        qubit_names = qubit_names or ["qubit"]

        if grid is None:
            coords = {name: (0, i) for i, name in enumerate(qubit_names)}
            grid = QubitGrid(coords, shape=(1, len(coords)))
        n_rows, n_cols, positions = grid.resolve(qubit_names)
        if residuals:
            total_rows = n_rows * 2
            rratio = float(getattr(_config.CURRENT_THEME, "residuals_height_ratio", 0.35))
            main_h = max(0.0, min(1.0, 1.0 - rratio))
            row_heights = []
            for _ in range(n_rows):
                row_heights.extend([main_h, rratio])
            titles = [""] * (total_rows * n_cols)
            for name, (r, c) in positions.items():
                row_main = (r - 1) * 2 + 1
                idx = (row_main - 1) * n_cols + (c - 1)
                titles[idx] = name
            self._fig = make_subplots(rows=total_rows, cols=n_cols, subplot_titles=titles, row_heights=row_heights)
        else:
            titles = [""] * (n_rows * n_cols)
            for name, (r, c) in positions.items():
                idx = (r - 1) * n_cols + (c - 1)
                titles[idx] = name
            self._fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)

        for name in qubit_names:
            if name not in positions:
                continue
            row, col = positions[name]
            if residuals:
                row_main = (row - 1) * 2 + 1
                row_resid = row_main + 1
            else:
                row_main = row
                row_resid = None
            sel = ds.sel({qubit_dim: name}) if qubit_dim in ds.dims else ds

            if y is None:
                x_vals = sel.coords[x].values if x in sel.coords else np.asarray(sel[x].values)
                var = data_var or next(iter(sel.data_vars))
                y_vals = np.asarray(sel[var].values)

                if hue and hue in sel.dims:
                    for hv in sel.coords[hue].values:
                        y_h = np.asarray(sel[var].sel({hue: hv}).values)
                        label = map_hue_value(hue, hv)
                        # Apply styling to scatter trace
                        scatter_kwargs = {
                            "x": x_vals, 
                            "y": y_h, 
                            "name": label, 
                            "mode": "markers",
                            "marker": dict(size=_config.CURRENT_THEME.marker_size),
                            "line": dict(width=_config.CURRENT_THEME.line_width)
                        }
                        # Apply style overrides if provided
                        if "marker_size" in style_overrides:
                            scatter_kwargs["marker"]["size"] = style_overrides["marker_size"]
                        if "line_width" in style_overrides:
                            scatter_kwargs["line"]["width"] = style_overrides["line_width"]
                        if "color" in style_overrides:
                            scatter_kwargs["marker"]["color"] = style_overrides["color"]
                        else:
                            # Assign consistent color per subplot
                            color = self._get_next_color(row_main, col)
                            scatter_kwargs["marker"]["color"] = color
                        if "mode" in style_overrides:
                            scatter_kwargs["mode"] = style_overrides["mode"]
                        self._fig.add_trace(go.Scatter(**scatter_kwargs), row=row_main, col=col)
                else:
                    # Apply styling to scatter trace
                    scatter_kwargs = {
                        "x": x_vals, 
                        "y": y_vals, 
                        "name": name, 
                        "mode": "markers",
                        "marker": dict(size=_config.CURRENT_THEME.marker_size),
                        "line": dict(width=_config.CURRENT_THEME.line_width)
                    }
                    # Apply style overrides if provided
                    if "marker_size" in style_overrides:
                        scatter_kwargs["marker"]["size"] = style_overrides["marker_size"]
                    if "line_width" in style_overrides:
                        scatter_kwargs["line"]["width"] = style_overrides["line_width"]
                    if "color" in style_overrides:
                        scatter_kwargs["marker"]["color"] = style_overrides["color"]
                    else:
                        # Assign consistent color per subplot
                        color = self._get_next_color(row_main, col)
                        scatter_kwargs["marker"]["color"] = color
                    if "mode" in style_overrides:
                        scatter_kwargs["mode"] = style_overrides["mode"]
                    self._fig.add_trace(go.Scatter(**scatter_kwargs), row=row_main, col=col)

                xlab = label_from_attrs(x, (sel.coords[x].attrs if x in sel.coords else {}))
                ylab = label_from_attrs(var, sel[var].attrs if hasattr(sel[var], "attrs") else {})
                self._fig.update_xaxes(title_text=xlab, row=row_main, col=col)
                self._fig.update_yaxes(title_text=ylab, row=row_main, col=col)

                if x2 and x2 in sel.coords:
                    xv2 = np.asarray(sel.coords[x2].values)
                    tickvals, ticktext = compute_secondary_ticks(x_vals, xv2)
                    idx = (row_main - 1) * n_cols + col
                    if idx == 1 and tickvals and ticktext:
                        self._fig.update_layout(
                            xaxis2={"overlaying": "x", "side": "top", "tickvals": tickvals, "ticktext": ticktext})

            else:
                var = data_var or next(iter(sel.data_vars))
                z_vals = np.asarray(sel[var].values)
                x_vals = np.asarray(sel.coords[x].values)
                y_vals = np.asarray(sel.coords[y].values)
                # Apply styling to heatmap trace
                heatmap_kwargs = {
                    "x": x_vals, 
                    "y": y_vals, 
                    "z": z_vals, 
                    "colorbar": dict(title=var),
                    "colorscale": "Viridis"  # Default colorscale
                }
                # Apply style overrides if provided
                if "colorscale" in style_overrides:
                    heatmap_kwargs["colorscale"] = style_overrides["colorscale"]
                if "colorbar" in style_overrides:
                    heatmap_kwargs["colorbar"].update(style_overrides["colorbar"])
                self._fig.add_trace(go.Heatmap(**heatmap_kwargs), row=row_main, col=col)
                xlab = label_from_attrs(x, sel.coords[x].attrs)
                ylab = label_from_attrs(y, sel.coords[y].attrs)
                self._fig.update_xaxes(title_text=xlab, row=row_main, col=col)
                self._fig.update_yaxes(title_text=ylab, row=row_main, col=col)

            # Track fit data for residuals calculation
            fit_data = None
            fit_x_vals = None
            
            if overlays:
                panel_overlays: Sequence[Overlay]
                if callable(overlays):
                    panel_overlays = overlays(name, sel)
                elif isinstance(overlays, dict):
                    panel_overlays = overlays.get(name, [])
                else:
                    panel_overlays = overlays
                for ov in panel_overlays:
                    # Pass x values for fit overlays
                    x_vals_for_overlay = x_vals if 'x_vals' in locals() else None
                    
                    # Add color to style overrides for consistent coloring
                    overlay_style = style_overrides.copy()
                    if "color" not in overlay_style:
                        overlay_style["color"] = self._get_next_color(row_main, col)
                    
                    ov.add_to(self._fig, row=row_main, col=col, theme=_config.CURRENT_THEME, x=x_vals_for_overlay, **overlay_style)
                    
                    # Check if this overlay provides fit data for residuals
                    if hasattr(ov, 'y_fit') and ov.y_fit is not None:
                        fit_data = ov.y_fit
                        fit_x_vals = x_vals_for_overlay if x_vals_for_overlay is not None else x_vals

            if residuals and row_resid is not None:
                # Add zero reference line
                self._fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0, xref="paper", yref="y", line={"dash": "dot"},
                                    row=row_resid, col=col)
                self._fig.update_yaxes(title_text="Residuals", row=row_resid, col=col)
                
                # Plot residuals if we have fit data
                if fit_data is not None and 'y_vals' in locals():
                    residual_vals = y_vals - fit_data
                    # Plot residuals as scatter points
                    residual_kwargs = {
                        "x": fit_x_vals if fit_x_vals is not None else x_vals,
                        "y": residual_vals,
                        "name": f"{name} residuals",
                        "mode": "markers",
                        "marker": dict(size=_config.CURRENT_THEME.marker_size),
                        "line": dict(width=_config.CURRENT_THEME.line_width)
                    }
                    # Apply style overrides if provided
                    if "marker_size" in style_overrides:
                        residual_kwargs["marker"]["size"] = style_overrides["marker_size"]
                    if "line_width" in style_overrides:
                        residual_kwargs["line"]["width"] = style_overrides["line_width"]
                    if "color" in style_overrides:
                        residual_kwargs["marker"]["color"] = style_overrides["color"]
                    else:
                        # Assign consistent color per subplot
                        color = self._get_next_color(row_resid, col)
                        residual_kwargs["marker"]["color"] = color
                    if "mode" in style_overrides:
                        residual_kwargs["mode"] = style_overrides["mode"]
                    
                    self._fig.add_trace(go.Scatter(**residual_kwargs), row=row_resid, col=col)
                    
                    # Update x-axis label for residuals (should match main plot)
                    if 'xlab' in locals():
                        self._fig.update_xaxes(title_text=xlab, row=row_resid, col=col)

        _config.apply_theme_to_layout(self._fig.layout)
        if _config.CURRENT_PALETTE:
            self._fig.update_layout(colorway=list(_config.CURRENT_PALETTE))
        if title:
            self._fig.update_layout(title=dict(text=title, font=dict(size=_config.CURRENT_THEME.title_size)))

        if _config.CURRENT_RC.values.get("showlegend") is not None:
            self._fig.update_layout(showlegend=bool(_config.CURRENT_RC.values["showlegend"]))

    def _get_next_color(self, row: int, col: int) -> str:
        """Get the next color in the sequence for the given subplot."""
        subplot_key = f"{row},{col}"
        palette = list(_config.CURRENT_PALETTE) if _config.CURRENT_PALETTE else list(_config.CURRENT_THEME.colorway)
        
        if subplot_key not in self._subplot_color_counters:
            self._subplot_color_counters[subplot_key] = 0
        
        color_idx = self._subplot_color_counters[subplot_key] % len(palette)
        color = palette[color_idx]
        self._subplot_color_counters[subplot_key] += 1
        
        return color
