from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Mapping, Any

import numpy as np
import plotly.graph_objects as go
from .config import PlotTheme


class Overlay:
    """Adds itself to a subplot (row, col) on a Plotly Figure."""

    def add_to(self, fig, *, row: int, col: int, theme, **style):
        raise NotImplementedError


@dataclass
class LineOverlay(Overlay):
    x: np.ndarray
    y: np.ndarray
    name: Optional[str] = None
    dash: Optional[str] = "dash"
    width: Optional[float] = None
    show_legend: bool = True

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, **style):
        fig.add_trace(
            go.Scatter(
                x=self.x,
                y=self.y,
                name=self.name,
                mode="lines",
                line={
                    "dash": self.dash,
                    "width": self.width or theme.line_width,
                    **style.get("line", {}),
                },
                showlegend=self.show_legend,
            ),
            row=row,
            col=col,
        )


@dataclass
class RefLine(Overlay):
    """Add vertical and/or horizontal reference lines to a plot.

    RefLine uses Plotly's add_vline and add_hline methods to draw reference
    lines that span the full extent of a subplot. Useful for marking thresholds,
    target values, or other important reference points.

    Attributes:
        x (float | None): X-coordinate for a vertical reference line. If None,
            no vertical line is drawn.
        y (float | None): Y-coordinate for a horizontal reference line. If None,
            no horizontal line is drawn.
        name (str | None): Optional name for the reference line (currently unused).
        dash (str): Line dash style. Options include "solid", "dot", "dash",
            "longdash", "dashdot", "longdashdot". Default is "dot".
        width (float | None): Line width in pixels. If None, uses theme.line_width.
        color (str | None): Line color. Can be a named color (e.g., "red", "blue"),
            hex color (e.g., "#FF0000"), or RGB/RGBA string. If None, uses the
            theme's default color.

    Examples:
        >>> # Add a vertical reference line at x=5
        >>> RefLine(x=5.0)

        >>> # Add a horizontal reference line at y=0.5 with custom styling
        >>> RefLine(y=0.5, dash="dash", width=2, color="red")

        >>> # Add both vertical and horizontal reference lines
        >>> RefLine(x=3.0, y=0.8, color="#00FF00")
    """
    x: Optional[float] = None
    y: Optional[float] = None
    name: Optional[str] = None
    dash: str = "dot"
    width: Optional[float] = None
    color: Optional[str] = None

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme: PlotTheme, **style):
        """Add the reference line(s) to the specified subplot.

        Constructs line configuration from instance attributes and theme settings,
        then uses Plotly's add_vline/add_hline to draw lines that automatically
        span the full extent of the subplot.

        Args:
            fig (go.Figure): The Plotly figure to add the reference line to.
            row (int): Subplot row index (1-indexed).
            col (int): Subplot column index (1-indexed).
            theme (PlotTheme): Theme object providing default styling values.
            **style: Additional style overrides. Can include a "line" dict with
                custom line properties (color, dash, width, etc.).

        Note:
            If both self.x and self.y are set, both vertical and horizontal
            lines will be drawn, creating a crosshair effect.
        """
        line_config = {
            "width": self.width or theme.line_width,
            "dash": self.dash,
            **style.get("line", {}),
        }
        if self.color is not None:
            line_config["color"] = self.color
        if self.x is not None:
            fig.add_vline(x=self.x, row=row, col=col, line=line_config)
        if self.y is not None:
            fig.add_hline(y=self.y, row=row, col=col, line=line_config)


@dataclass
class ScatterOverlay(Overlay):
    x: np.ndarray
    y: np.ndarray
    name: Optional[str] = None
    marker_size: Optional[float] = None

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, **style):
        fig.add_trace(
            go.Scatter(
                x=self.x,
                y=self.y,
                name=self.name,
                mode="markers",
                marker={
                    "size": self.marker_size or theme.marker_size,
                    **style.get("marker", {}),
                },
            ),
            row=row,
            col=col,
        )


@dataclass
class TextBoxOverlay(Overlay):
    text: str
    anchor: str = "top right"

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, **style):
        x = 1.0 if "right" in self.anchor else 0.0
        y = 1.0 if "top" in self.anchor else 0.0
        fig.add_annotation(
            text=self.text,
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right" if "right" in self.anchor else "left",
            yanchor="top" if "top" in self.anchor else "bottom",
            row=row,
            col=col,
        )


@dataclass
class FitOverlay(Overlay):
    y_fit: Optional[np.ndarray] = None
    params: Optional[Mapping[str, Any]] = None
    formatter: Optional[Callable[[Mapping[str, Any]], str]] = None
    name: str = "fit"
    dash: str = "dash"
    width: Optional[float] = None

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, x=None, **style):
        if self.y_fit is not None and x is not None:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=self.y_fit,
                    name=self.name,
                    mode="lines",
                    line={
                        "dash": self.dash,
                        "width": self.width or theme.line_width,
                        **style.get("line", {}),
                    },
                ),
                row=row,
                col=col,
            )
        if self.params is not None and self.formatter is not None:
            text = self.formatter(self.params)
            TextBoxOverlay(text=text).add_to(fig, row=row, col=col, theme=theme)
