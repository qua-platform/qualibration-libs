from typing import List

import numpy as np
import plotly.graph_objects as go

from .configs import PlotStyling

styling = PlotStyling()

def add_plotly_top_axis(
    fig: go.Figure,
    row: int,
    col: int,
    n_cols: int,
    x_values: np.ndarray,
    top_axis_tick_labels: np.ndarray,
    top_axis_title: str,
    tick_format: str = "{:.2f}",
):
    """
    Adds a secondary "relabeling" x-axis to the top of a plotly subplot.
    This axis does not have its own scale, it just adds labels to the ticks of the main x-axis.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object.
    row : int
        The subplot row (1-indexed).
    col : int
        The subplot col (1-indexed).
    n_cols : int
        The total number of columns in the grid.
    x_values : np.ndarray
        The values from the main x-axis to use for tick positioning.
    top_axis_tick_labels : np.ndarray
        The labels for the top x-axis ticks.
    top_axis_title : str
        The title for the top x-axis.
    tick_format : str, optional
        The format string for the tick labels, by default "{:.2f}".
    """
    subplot_index = (row - 1) * n_cols + col
    if subplot_index == 1:
        main_xaxis_name = "x"
    else:
        main_xaxis_name = f"x{subplot_index}"

    top_xaxis_name = f"xaxis{subplot_index + styling.plotly_axis_offset}"

    fig.update_layout(
        {
            top_xaxis_name: dict(
                overlaying=main_xaxis_name,
                side="top",
                title=top_axis_title,
                showgrid=False,
                tickmode="array",
                tickvals=list(x_values),
                ticktext=[tick_format.format(v) for v in top_axis_tick_labels],
            )
        }
    )
