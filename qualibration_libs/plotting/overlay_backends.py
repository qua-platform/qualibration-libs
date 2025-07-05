"""Backend adapters for the overlay abstraction system.

This module provides concrete implementations of the PlotBackend protocol
for different plotting libraries (Plotly and Matplotlib).
"""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .overlays import PlotBackend
from .configs.constants import PlotConstants


class PlotlyBackend:
    """Plotly backend adapter for overlay rendering."""
    
    def __init__(self, figure: go.Figure):
        """Initialize with a Plotly figure.
        
        Args:
            figure: The Plotly figure to add overlays to
        """
        self.figure = figure
    
    def add_vertical_line(
        self,
        x: float,
        y_range: Tuple[float, float],
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a vertical line to the Plotly figure.
        
        Args:
            x: X position of the line
            y_range: Tuple of (y_min, y_max) for the line
            style: Dictionary with style properties (color, width, dash)
            subplot_position: Grid position (row, col) for the subplot
        """
        row, col = subplot_position
        
        self.figure.add_trace(
            go.Scatter(
                x=[x, x],
                y=list(y_range),
                mode="lines",
                line=dict(
                    color=style.get("color", "#000000"),
                    width=style.get("width", 2),
                    dash=style.get("dash", "solid")
                ),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=row,
            col=col
        )
    
    def add_horizontal_line(
        self,
        y: float,
        x_range: Tuple[float, float],
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a horizontal line to the Plotly figure.
        
        Args:
            y: Y position of the line
            x_range: Tuple of (x_min, x_max) for the line
            style: Dictionary with style properties (color, width, dash)
            subplot_position: Grid position (row, col) for the subplot
        """
        row, col = subplot_position
        
        self.figure.add_trace(
            go.Scatter(
                x=list(x_range),
                y=[y, y],
                mode="lines",
                line=dict(
                    color=style.get("color", "#000000"),
                    width=style.get("width", 2),
                    dash=style.get("dash", "solid")
                ),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=row,
            col=col
        )
    
    def add_marker(
        self,
        x: float,
        y: float,
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a marker to the Plotly figure.
        
        Args:
            x: X position of the marker
            y: Y position of the marker
            style: Dictionary with style properties (symbol, color, size)
            subplot_position: Grid position (row, col) for the subplot
        """
        row, col = subplot_position
        
        self.figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    symbol=style.get("symbol", "circle"),
                    color=style.get("color", "#000000"),
                    size=style.get("size", 10)
                ),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=row,
            col=col
        )


class MatplotlibBackend:
    """Matplotlib backend adapter for overlay rendering."""
    
    def __init__(self, axes: plt.Axes):
        """Initialize with matplotlib axes.
        
        Args:
            axes: The matplotlib axes to add overlays to
        """
        self.axes = axes
        self._line_style_map = {
            "solid": "-",
            "dash": "--",
            "dashdot": "-.",
            "dot": ":"
        }
        self._marker_style_map = {
            "x": "x",
            "circle": "o",
            "square": "s",
            "diamond": "D",
            "cross": "+",
            "triangle-up": "^",
            "triangle-down": "v",
            "star": "*"
        }
    
    def add_vertical_line(
        self,
        x: float,
        y_range: Tuple[float, float],
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a vertical line to the matplotlib axes.
        
        Args:
            x: X position of the line
            y_range: Ignored - matplotlib axvline spans full height
            style: Dictionary with style properties (color, width, dash)
            subplot_position: Ignored - axes already selected
        """
        color = style.get("color", "#000000")
        linewidth = style.get("width", 2)
        linestyle = self._line_style_map.get(style.get("dash", "solid"), "-")
        
        self.axes.axvline(
            x=x,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )
    
    def add_horizontal_line(
        self,
        y: float,
        x_range: Tuple[float, float],
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a horizontal line to the matplotlib axes.
        
        Args:
            y: Y position of the line
            x_range: Ignored - matplotlib axhline spans full width
            style: Dictionary with style properties (color, width, dash)
            subplot_position: Ignored - axes already selected
        """
        color = style.get("color", "#000000")
        linewidth = style.get("width", 2)
        linestyle = self._line_style_map.get(style.get("dash", "solid"), "-")
        
        self.axes.axhline(
            y=y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )
    
    def add_marker(
        self,
        x: float,
        y: float,
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a marker to the matplotlib axes.
        
        Args:
            x: X position of the marker
            y: Y position of the marker
            style: Dictionary with style properties (symbol, color, size)
            subplot_position: Ignored - axes already selected
        """
        color = style.get("color", "#000000")
        size = style.get("size", 10)
        symbol = style.get("symbol", "o")
        marker = self._marker_style_map.get(symbol, "o")
        
        self.axes.plot(
            x, y,
            marker=marker,
            color=color,
            markersize=size,
            linestyle='None'
        )


class MultiAxesMatplotlibBackend:
    """Matplotlib backend that handles multiple axes in a grid layout."""
    
    def __init__(self, figure: plt.Figure, axes_grid: list):
        """Initialize with matplotlib figure and axes grid.
        
        Args:
            figure: The matplotlib figure
            axes_grid: 2D list of axes objects
        """
        self.figure = figure
        self.axes_grid = axes_grid
        self.single_backend_cache = {}
    
    def _get_axes(self, subplot_position: Tuple[int, int]) -> plt.Axes:
        """Get the axes for a specific subplot position.
        
        Args:
            subplot_position: Grid position (row, col) - 1-indexed
            
        Returns:
            The matplotlib axes at that position
        """
        row, col = subplot_position
        # Convert from 1-indexed to 0-indexed
        return self.axes_grid[row - 1][col - 1]
    
    def add_vertical_line(
        self,
        x: float,
        y_range: Tuple[float, float],
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a vertical line to the specified subplot."""
        ax = self._get_axes(subplot_position)
        backend = self._get_single_backend(ax)
        backend.add_vertical_line(x, y_range, style, subplot_position)
    
    def add_horizontal_line(
        self,
        y: float,
        x_range: Tuple[float, float],
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a horizontal line to the specified subplot."""
        ax = self._get_axes(subplot_position)
        backend = self._get_single_backend(ax)
        backend.add_horizontal_line(y, x_range, style, subplot_position)
    
    def add_marker(
        self,
        x: float,
        y: float,
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a marker to the specified subplot."""
        ax = self._get_axes(subplot_position)
        backend = self._get_single_backend(ax)
        backend.add_marker(x, y, style, subplot_position)
    
    def _get_single_backend(self, ax: plt.Axes) -> MatplotlibBackend:
        """Get or create a single-axes backend for the given axes."""
        if ax not in self.single_backend_cache:
            self.single_backend_cache[ax] = MatplotlibBackend(ax)
        return self.single_backend_cache[ax]