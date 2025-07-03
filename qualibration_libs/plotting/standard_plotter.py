from typing import Dict, List, Literal, Optional, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
from qualibration_libs.analysis import lorentzian_dip
from qualibration_libs.plotting.grids import (PlotlyQubitGrid, QubitGrid,
                                              grid_iter, plotly_grid_iter)
from quam_builder.architecture.superconducting.qubit import AnyTransmon


class TraceConfig(BaseModel):
    plot_type: Literal["scatter", "heatmap", "line"]
    x_source: str
    y_source: str
    z_source: Optional[str] = None  # For heatmaps
    name: str
    mode: Optional[str] = "lines+markers"
    style: Dict = Field(default_factory=dict)  # e.g., {"color": "blue", "dash": "dot"}
    hover_template: Optional[str] = None
    custom_data_sources: List[str] = Field(default_factory=list)
    visible: bool = True

class LayoutConfig(BaseModel):
    title: str
    x_axis_title: str
    y_axis_title: str
    legend: Dict = Field(default_factory=dict)

class PlotConfig(BaseModel):
    layout: LayoutConfig
    traces: List[TraceConfig]
    fit_traces: List[TraceConfig] = Field(default_factory=list)


def create_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: List[PlotConfig],
    ds_fit: Optional[xr.Dataset] = None,
) -> go.Figure:
    """
    Creates a Plotly figure from raw data, fit data, and a list of plot configurations.

    Args:
        ds_raw: The raw data from the experiment.
        qubits: A list of qubits to plot.
        plot_configs: A list of plot configurations.
        ds_fit: The fitted data from the analysis step.

    Returns:
        A Plotly figure object.
    """
    if not plot_configs:
        return go.Figure()

    config = plot_configs[0] # For now, assume one config for the whole figure.
    grid = PlotlyQubitGrid(ds_raw, [q.grid_location for q in qubits])

    fig = make_subplots(
        rows=grid.n_rows,
        cols=grid.n_cols,
        subplot_titles=grid.get_subplot_titles(),
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    for i, ((grid_row, grid_col), name_dict) in enumerate(plotly_grid_iter(grid)):
        row = grid_row + 1  # Convert to 1-based indexing for Plotly
        col = grid_col + 1  # Convert to 1-based indexing for Plotly
        qubit_id = list(name_dict.values())[0]

        ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
        ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None

        for trace_config in config.traces + config.fit_traces:
            if not trace_config.visible:
                continue

            is_fit_trace = trace_config in config.fit_traces
            ds_source = ds_qubit_fit if is_fit_trace else ds_qubit_raw

            if ds_source is None:
                continue

            # Check that all required data sources for this trace exist in the dataset
            required_sources = [trace_config.x_source, trace_config.y_source]
            if trace_config.z_source:
                required_sources.append(trace_config.z_source)
            if not all(source in ds_source for source in required_sources):
                continue

            if trace_config.custom_data_sources:
                custom_data = np.stack([ds_source[src].values for src in trace_config.custom_data_sources], axis=-1)
            else:
                custom_data = None

            trace_props = {
                "name": trace_config.name,
                "hovertemplate": trace_config.hover_template,
                "customdata": custom_data,
                "legendgroup": trace_config.name,
                "showlegend": i == 0
            }

            if trace_config.plot_type == "heatmap":
                trace = go.Heatmap(
                    x=ds_source[trace_config.x_source].values,
                    y=ds_source[trace_config.y_source].values,
                    z=ds_source[trace_config.z_source].values,
                    **trace_props,
                    **trace_config.style,
                )
            else: # scatter, line
                trace = go.Scatter(
                    x=ds_source[trace_config.x_source].values,
                    y=ds_source[trace_config.y_source].values,
                    mode=trace_config.mode,
                    line=trace_config.style,
                    **trace_props,
                )
            fig.add_trace(trace, row=row, col=col)

        fig.update_xaxes(title_text=config.layout.x_axis_title, row=row, col=col)
        fig.update_yaxes(title_text=config.layout.y_axis_title, row=row, col=col)

    fig.update_layout(
        title_text=config.layout.title,
        legend=config.layout.legend,
        height=350 * grid.n_rows,
        width=max(800, 350 * grid.n_cols)
    )
    return fig


def create_matplotlib_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: List[PlotConfig],
    ds_fit: Optional[xr.Dataset] = None,
) -> MatplotlibFigure:
    """
    Creates a static Matplotlib figure from raw data, fit data, and a list of plot configurations.
    """
    if not plot_configs:
        fig, _ = plt.subplots()
        return fig

    config = plot_configs[0]
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    grid.fig.suptitle(config.layout.title)

    for i, (ax, name_dict) in enumerate(grid_iter(grid)):
        qubit_id = list(name_dict.values())[0]
        ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
        ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None

        # Plot raw data traces
        for trace_config in config.traces:
            if not trace_config.visible:
                continue
                
            if trace_config.plot_type == "heatmap":
                # Handle heatmap plots for matplotlib using xarray's built-in plotting
                if all(s in ds_qubit_raw for s in [trace_config.x_source, trace_config.y_source, trace_config.z_source]):
                    # Use xarray's built-in plot method which handles 2D data properly
                    ds_qubit_raw[trace_config.z_source].plot(
                        ax=ax,
                        x=trace_config.x_source,
                        y=trace_config.y_source,
                        add_colorbar=(i == 0),  # Only add colorbar for first subplot
                        cbar_kwargs={"label": trace_config.name} if i == 0 else {}
                    )
                        
            elif all(s in ds_qubit_raw for s in [trace_config.x_source, trace_config.y_source]):
                # Handle line/scatter plots
                ax.plot(
                    ds_qubit_raw[trace_config.x_source].values,
                    ds_qubit_raw[trace_config.y_source].values,
                    marker='.', linestyle='-',
                    label=trace_config.name if i == 0 else ""
                )

        # Plot fit traces
        if ds_qubit_fit:
            for trace_config in config.fit_traces:
                if trace_config.visible and all(s in ds_qubit_fit for s in [trace_config.x_source, trace_config.y_source]):
                    # Translate plotly dash styles to matplotlib linestyle
                    linestyle_map = {"solid": "-", "dot": ":", "dash": "--", "longdash": "-.", "dashdot": "-."}
                    linestyle = linestyle_map.get(trace_config.style.get("dash"), "--")
                    
                    ax.plot(
                        ds_qubit_fit[trace_config.x_source].values,
                        ds_qubit_fit[trace_config.y_source].values,
                        linestyle=linestyle,
                        label=trace_config.name if i == 0 else "",
                        color=trace_config.style.get("color", "red")
                    )

        ax.set_xlabel(config.layout.x_axis_title)
        ax.set_ylabel(config.layout.y_axis_title)
        ax.set_title(f"Qubit {qubit_id}")

    if any(trace.name for trace in config.traces + config.fit_traces):
        grid.fig.legend()
        
    grid.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return grid.fig


def create_specialized_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon], 
    plot_config: Any,  # Union of various config types
    ds_fit: Optional[xr.Dataset] = None,
    ds_prepared: Optional[xr.Dataset] = None
) -> go.Figure:
    """
    Creates a specialized Plotly figure for complex plot types (heatmaps, dual axes, overlays).
    
    Args:
        ds_raw: The raw data from the experiment
        qubits: A list of qubits to plot
        plot_config: Enhanced plot configuration (HeatmapConfig, ChevronConfig, etc.)
        ds_fit: The fitted data from the analysis step
        ds_prepared: Optional pre-prepared data with standardized coordinates
        
    Returns:
        A Plotly figure object
    """
    # Import configs here to avoid circular imports
    from .configs import HeatmapConfig, ChevronConfig, SpectroscopyConfig
    
    # Use prepared data if available, otherwise use raw data
    ds_plot = ds_prepared if ds_prepared is not None else ds_raw
    
    # Handle different config types
    if isinstance(plot_config, HeatmapConfig):
        return _create_heatmap_figure(ds_plot, qubits, plot_config, ds_fit)
    elif isinstance(plot_config, ChevronConfig):
        return _create_chevron_figure(ds_plot, qubits, plot_config, ds_fit)
    elif isinstance(plot_config, SpectroscopyConfig):
        return _create_spectroscopy_figure(ds_plot, qubits, plot_config, ds_fit)
    else:
        # Fallback to basic plotter
        return create_plotly_figure(ds_raw, qubits, [plot_config], ds_fit)


def _create_heatmap_figure(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    config: Any,  # HeatmapConfig
    ds_fit: Optional[xr.Dataset] = None
) -> go.Figure:
    """Create a heatmap figure with overlays and dual axes."""
    grid = PlotlyQubitGrid(ds, [q.grid_location for q in qubits])
    
    # Create subplots with custom spacing
    spacing = config.subplot_spacing
    fig = make_subplots(
        rows=grid.n_rows,
        cols=grid.n_cols,
        subplot_titles=grid.get_subplot_titles(),
        horizontal_spacing=spacing.get("horizontal", 0.15),
        vertical_spacing=spacing.get("vertical", 0.12),
        shared_xaxes=False,
        shared_yaxes=False
    )
    
    heatmap_info = []
    
    for idx, ((grid_row, grid_col), name_dict) in enumerate(plotly_grid_iter(grid)):
        row = grid_row + 1
        col = grid_col + 1
        qubit_id = list(name_dict.values())[0]
        
        # Add heatmap traces
        for trace_config in config.traces:
            if trace_config.plot_type == "heatmap":
                _add_heatmap_trace(fig, ds, qubit_id, trace_config, row, col)
                heatmap_info.append((len(fig.data) - 1, row, col))
        
        # Add overlays if fit data available
        if ds_fit is not None and hasattr(config, 'overlays'):
            _add_overlays(fig, ds_fit, qubit_id, config.overlays, row, col)
        
        # Update axes
        fig.update_xaxes(title_text=config.layout.x_axis_title, row=row, col=col)
        fig.update_yaxes(title_text=config.layout.y_axis_title, row=row, col=col)
    
    # Position colorbars
    _position_colorbars(fig, heatmap_info, grid.n_cols, config.traces[0].colorbar if hasattr(config.traces[0], 'colorbar') else None)
    
    # Final layout
    fig.update_layout(
        title_text=config.layout.title,
        showlegend=False,
        height=400 * grid.n_rows,
        width=max(1000, 400 * grid.n_cols)
    )
    
    return fig


def _create_chevron_figure(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    config: Any,  # ChevronConfig  
    ds_fit: Optional[xr.Dataset] = None
) -> go.Figure:
    """Create a chevron-style figure for power Rabi plots."""
    grid = PlotlyQubitGrid(ds, [q.grid_location for q in qubits])
    
    spacing = config.subplot_spacing
    fig = make_subplots(
        rows=grid.n_rows,
        cols=grid.n_cols,
        subplot_titles=grid.get_subplot_titles(),
        horizontal_spacing=spacing.get("horizontal", 0.1),
        vertical_spacing=spacing.get("vertical", 0.2),
        shared_xaxes=False,
        shared_yaxes=False
    )
    
    for idx, ((grid_row, grid_col), name_dict) in enumerate(plotly_grid_iter(grid)):
        row = grid_row + 1
        col = grid_col + 1
        qubit_id = list(name_dict.values())[0]
        
        # Determine if 1D or 2D based on data dimensions
        is_2d = "nb_of_pulses" in ds.dims and ds.sizes.get("nb_of_pulses", 1) > 1
        
        if is_2d:
            _add_2d_chevron_traces(fig, ds, qubit_id, config, row, col)
        else:
            _add_1d_chevron_traces(fig, ds, qubit_id, config, row, col)
        
        # Add dual axis if configured
        if config.dual_axis and config.dual_axis.enabled:
            _add_dual_axis(fig, ds, qubit_id, config.dual_axis, row, col, grid.n_cols)
        
        # Add overlays for 2D plots
        if is_2d and ds_fit is not None and hasattr(config, 'overlays'):
            _add_overlays(fig, ds_fit, qubit_id, config.overlays, row, col)
        
        fig.update_xaxes(title_text=config.layout.x_axis_title, row=row, col=col)
        fig.update_yaxes(title_text=config.layout.y_axis_title, row=row, col=col)
    
    fig.update_layout(
        title_text=config.layout.title,
        showlegend=False,
        height=900,
        width=1500
    )
    
    return fig


def _create_spectroscopy_figure(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    config: Any,  # SpectroscopyConfig
    ds_fit: Optional[xr.Dataset] = None
) -> go.Figure:
    """Create a spectroscopy figure with dual axes."""
    # Use existing create_plotly_figure but add dual axis support
    fig = create_plotly_figure(ds, qubits, [config], ds_fit)
    
    # Add dual axis if configured
    if config.dual_axis and config.dual_axis.enabled:
        grid = PlotlyQubitGrid(ds, [q.grid_location for q in qubits])
        for idx, ((grid_row, grid_col), name_dict) in enumerate(plotly_grid_iter(grid)):
            row = grid_row + 1
            col = grid_col + 1
            qubit_id = list(name_dict.values())[0]
            _add_dual_axis(fig, ds, qubit_id, config.dual_axis, row, col, grid.n_cols)
    
    return fig


def _add_heatmap_trace(
    fig: go.Figure,
    ds: xr.Dataset,
    qubit_id: str,
    trace_config: Any,  # HeatmapTraceConfig
    row: int,
    col: int
):
    """Add a heatmap trace to the figure."""
    ds_qubit = ds.sel(qubit=qubit_id)
    
    # Get data arrays
    x_data = ds_qubit[trace_config.x_source].values
    y_data = ds_qubit[trace_config.y_source].values
    z_data = ds_qubit[trace_config.z_source].values
    
    # Ensure proper shape for heatmap
    if z_data.ndim == 1:
        z_data = z_data[np.newaxis, :]
    
    # Calculate robust z-limits
    zmin = np.nanpercentile(z_data.flatten(), trace_config.zmin_percentile)
    zmax = np.nanpercentile(z_data.flatten(), trace_config.zmax_percentile)
    
    # Build hover customdata if specified
    customdata = None
    if trace_config.custom_data_sources:
        custom_arrays = []
        for src in trace_config.custom_data_sources:
            if src in ds_qubit:
                custom_arrays.append(ds_qubit[src].values)
        if custom_arrays:
            customdata = np.stack(custom_arrays, axis=-1)
    
    fig.add_trace(
        go.Heatmap(
            x=x_data,
            y=y_data, 
            z=z_data,
            customdata=customdata,
            colorscale=trace_config.colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            colorbar=dict(
                title=trace_config.colorbar.title if hasattr(trace_config, 'colorbar') else "|IQ|",
                thickness=trace_config.colorbar.thickness if hasattr(trace_config, 'colorbar') else 14
            ),
            hovertemplate=trace_config.hover_template,
            name=trace_config.name
        ),
        row=row,
        col=col
    )


def _add_1d_chevron_traces(
    fig: go.Figure,
    ds: xr.Dataset,
    qubit_id: str,
    config: Any,  # ChevronConfig
    row: int,
    col: int
):
    """Add 1D traces for chevron plots."""
    ds_qubit = ds.sel(qubit=qubit_id)
    
    # Handle single pulse case
    if "nb_of_pulses" in ds_qubit.dims and ds_qubit.sizes["nb_of_pulses"] == 1:
        ds_qubit = ds_qubit.isel(nb_of_pulses=0)
    
    for trace_config in config.traces:
        if trace_config.plot_type in ["scatter", "line"]:
            x_data = ds_qubit[trace_config.x_source].values
            y_data = ds_qubit[trace_config.y_source].values
            
            # Build customdata
            customdata = None
            if trace_config.custom_data_sources:
                custom_arrays = [ds_qubit[src].values for src in trace_config.custom_data_sources if src in ds_qubit]
                if custom_arrays:
                    customdata = np.stack(custom_arrays, axis=-1)
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=trace_config.mode,
                    name=trace_config.name,
                    customdata=customdata,
                    hovertemplate=trace_config.hover_template,
                    line=trace_config.style,
                    showlegend=False
                ),
                row=row,
                col=col
            )


def _add_2d_chevron_traces(
    fig: go.Figure,
    ds: xr.Dataset,
    qubit_id: str,
    config: Any,  # ChevronConfig
    row: int, 
    col: int
):
    """Add 2D heatmap traces for chevron plots."""
    for trace_config in config.traces:
        if trace_config.plot_type == "heatmap":
            _add_heatmap_trace(fig, ds, qubit_id, trace_config, row, col)


def _add_overlays(
    fig: go.Figure,
    ds_fit: xr.Dataset,
    qubit_id: str,
    overlays: List[Any],
    row: int,
    col: int
):
    """Add overlay traces (lines, markers) based on fit results."""
    if qubit_id not in ds_fit.qubit.values:
        return
        
    fit_qubit = ds_fit.sel(qubit=qubit_id)
    
    # Check if overlay condition is met
    for overlay in overlays:
        condition_met = True
        if hasattr(overlay, 'condition_source') and hasattr(fit_qubit, overlay.condition_source):
            condition_value = getattr(fit_qubit, overlay.condition_source).values
            condition_met = condition_value == overlay.condition_value
        
        if not condition_met:
            continue
            
        if overlay.type == "line":
            _add_line_overlay(fig, fit_qubit, overlay, row, col)
        elif overlay.type == "marker":
            _add_marker_overlay(fig, fit_qubit, overlay, row, col)


def _add_line_overlay(fig: go.Figure, fit_qubit: xr.Dataset, overlay: Any, row: int, col: int):
    """Add a line overlay (vertical or horizontal)."""
    if not hasattr(fit_qubit, overlay.position_source):
        return
        
    position = float(getattr(fit_qubit, overlay.position_source).values)
    
    if overlay.orientation == "vertical":
        # Need to determine y-range from the existing traces
        fig.add_trace(
            go.Scatter(
                x=[position, position],
                y=[0, 1],  # Will be updated by plotly to match data range
                mode="lines",
                line=overlay.line_style,
                showlegend=False,
                hoverinfo="skip"
            ),
            row=row,
            col=col
        )
    elif overlay.orientation == "horizontal":
        fig.add_trace(
            go.Scatter(
                x=[0, 1],  # Will be updated by plotly to match data range  
                y=[position, position],
                mode="lines",
                line=overlay.line_style,
                showlegend=False,
                hoverinfo="skip"
            ),
            row=row,
            col=col
        )


def _add_marker_overlay(fig: go.Figure, fit_qubit: xr.Dataset, overlay: Any, row: int, col: int):
    """Add a marker overlay."""
    if not (hasattr(fit_qubit, overlay.x_source) and hasattr(fit_qubit, overlay.y_source)):
        return
        
    x_pos = float(getattr(fit_qubit, overlay.x_source).values)
    y_pos = float(getattr(fit_qubit, overlay.y_source).values)
    
    fig.add_trace(
        go.Scatter(
            x=[x_pos],
            y=[y_pos],
            mode="markers",
            marker=overlay.marker_style,
            showlegend=False,
            hoverinfo="skip"
        ),
        row=row,
        col=col
    )


def _add_dual_axis(
    fig: go.Figure,
    ds: xr.Dataset,
    qubit_id: str,
    dual_config: Any,  # DualAxisConfig
    row: int,
    col: int,
    n_cols: int
):
    """Add a dual axis (top x-axis) to a subplot."""
    ds_qubit = ds.sel(qubit=qubit_id)
    
    if dual_config.top_axis_source not in ds_qubit:
        return
        
    top_axis_data = ds_qubit[dual_config.top_axis_source].values
    
    subplot_index = (row - 1) * n_cols + col
    main_xaxis = f"x{subplot_index}"
    top_xaxis_layout = f"xaxis{subplot_index + dual_config.overlay_offset}"
    
    if len(top_axis_data) > 1:
        tick_text = [dual_config.top_axis_format.format(v) for v in top_axis_data]
        
        fig.layout[top_xaxis_layout] = dict(
            overlaying=main_xaxis,
            side="top",
            title=dual_config.top_axis_title,
            showgrid=False,
            tickmode="array",
            tickvals=list(top_axis_data),
            ticktext=tick_text,
            range=[float(np.min(top_axis_data)), float(np.max(top_axis_data))]
        )
    else:
        fig.layout[top_xaxis_layout] = dict(
            overlaying=main_xaxis,
            side="top", 
            title=dual_config.top_axis_title,
            showgrid=False,
            tickmode="auto"
        )


def _position_colorbars(
    fig: go.Figure,
    heatmap_info: List[tuple],
    n_cols: int,
    colorbar_config: Optional[Any] = None
):
    """Position colorbars for heatmap subplots."""
    if not colorbar_config:
        return
        
    for hm_idx, row, col in heatmap_info:
        axis_num = (row - 1) * n_cols + col
        xaxis_key = f"xaxis{axis_num}"
        yaxis_key = f"yaxis{axis_num}"
        
        if xaxis_key in fig.layout and yaxis_key in fig.layout:
            x_dom = fig.layout[xaxis_key].domain
            y_dom = fig.layout[yaxis_key].domain
            
            x0_cb = x_dom[1] + colorbar_config.x_offset
            y0 = y_dom[0]
            y1 = y_dom[1]
            bar_len = (y1 - y0) * colorbar_config.height_ratio
            bar_center_y = (y0 + y1) / 2
            
            if hm_idx < len(fig.data):
                fig.data[hm_idx].colorbar.update({
                    "x": x0_cb,
                    "y": bar_center_y,
                    "len": bar_len,
                    "thickness": colorbar_config.thickness,
                    "xanchor": "left",
                    "yanchor": "middle",
                    "ticks": colorbar_config.ticks,
                    "ticklabelposition": colorbar_config.ticklabelposition
                })