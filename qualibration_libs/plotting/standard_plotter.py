from typing import Dict, List, Literal, Optional

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
from qualibration_libs.plotting.grids import PlotlyQubitGrid, plotly_grid_iter
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
        subplot_titles=[f"Qubit {list(nd.values())[0]}" for nd in grid.name_dicts],
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    for i, name_dict in plotly_grid_iter(grid):
        row = i // grid.n_cols + 1
        col = i % grid.n_cols + 1
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