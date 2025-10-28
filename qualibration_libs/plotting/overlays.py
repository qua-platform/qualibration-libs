from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import plotly.graph_objects as go

from .config import PlotTheme


class Overlay:
    """Base class for plot overlays in the qualibrate plotting system.

    Overlays are additional visual elements (lines, markers, annotations) that can be
    added on top of the main data plot. They are commonly used to display fit curves,
    reference lines, theoretical predictions, or key measurement points.

    This is an abstract base class that defines the interface for all overlay types.
    Subclasses must implement the `add_to()` method to define how they are rendered
    on a Plotly figure.

    Usage in QualibrationFigure.plot
    ---------------------------------
    Overlays are passed to `QualibrationFigure.plot()` via the `overlays` parameter,
    which accepts three formats:

    1. **Sequence of overlays** - Same overlays applied to all qubits:

        >>> overlays = [
        ...     RefLine(x=5.0, name="Target"),
        ...     RefLine(y=0.5, name="Threshold")
        ... ]
        >>> fig = QualibrationFigure.plot(dataset, x='freq', overlays=overlays)

    2. **Dictionary of overlays** - Qubit-specific overlays:

        >>> overlays = {
        ...     'q0': [RefLine(x=5.0), RefLine(y=0.5)],
        ...     'q1': [RefLine(x=5.2), RefLine(y=0.6)]
        ... }
        >>> fig = QualibrationFigure.plot(dataset, x='freq', overlays=overlays)

    3. **Callable (most common)** - Function that generates overlays per qubit:

        >>> def create_overlays(qubit_name, qubit_data):
        ...     # Extract fit parameters specific to this qubit
        ...     fit_params = extract_fit_parameters(qubit_name)
        ...
        ...     # Generate fitted curve
        ...     fitted_curve = calculate_fit_curve(**fit_params)
        ...
        ...     return [
        ...         FitOverlay(y_fit=fitted_curve, name="Fit"),
        ...         RefLine(x=fit_params['optimal_x'], name="Optimal")
        ...     ]
        >>>
        >>> fig = QualibrationFigure.plot(
        ...     dataset,
        ...     x='frequency',
        ...     data_var='response',
        ...     overlays=create_overlays
        ... )

    The callable approach is most flexible as it allows accessing qubit-specific data
    and fit results to dynamically generate appropriate overlays for each subplot.

    Implementing Custom Overlays
    -----------------------------
    To create a custom overlay type, subclass Overlay and implement `add_to()`:

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class CustomOverlay(Overlay):
        ...     value: float
        ...     color: str = "blue"
        ...
        ...     def add_to(self, fig, *, row, col, theme, **style):
        ...         # Add your custom visualization to fig at (row, col)
        ...         fig.add_hline(y=self.value, row=row, col=col,
        ...                       line=dict(color=self.color))

    Notes
    -----
    - The `add_to()` method is called by QualibrationFigure for each subplot
    - Overlays receive the figure, subplot position (row, col), and theme settings
    - Style overrides from the main plot can be accessed via `**style`
    - Multiple overlays can be combined in a list for complex visualizations

    See Also
    --------
    FitOverlay : Overlay for fitted curves with residual support
    RefLine : Overlay for reference lines (vertical or horizontal)
    LineOverlay : Overlay for arbitrary line plots
    ScatterOverlay : Overlay for scatter points
    QualibrationFigure.plot : Main plotting function that uses overlays
    """

    def add_to(self, fig, *, row: int, col: int, theme, **style):
        raise NotImplementedError


@dataclass
class LineOverlay(Overlay):
    """Add line overlays to plots for arbitrary curves or trajectories.

    LineOverlay draws a continuous line through specified (x, y) points. Unlike
    FitOverlay which is specifically designed for fitted curves with residual support,
    LineOverlay is a general-purpose tool for adding any line-based visualization to
    plots (e.g., theoretical curves, trajectories, boundaries, manually defined paths).

    The line is drawn with mode="lines" (no markers) and can be styled with various
    dash patterns and widths.

    Attributes:
        x (np.ndarray): Array of x-coordinates for the line points.
        y (np.ndarray): Array of y-coordinates for the line points. Must have the
            same length as x.
        name (str | None): Name for the line in the legend. If None, the line will
            not appear in the legend.
        dash (str | None): Line dash style. Options include "solid", "dot", "dash",
            "longdash", "dashdot", "longdashdot". Default is "dash". If None, uses
            Plotly's default.
        width (float | None): Line width in pixels. If None, uses theme.line_width.
        show_legend (bool): Whether to show this line in the legend. Default is True.
            Set to False to hide from legend even if name is provided.

    Examples:
        Basic line overlay:

        >>> # Add a theoretical curve to the plot
        >>> x_theory = np.linspace(0, 10, 100)
        >>> y_theory = calculate_theoretical_response(x_theory)
        >>> LineOverlay(x=x_theory, y=y_theory, name="Theory")

        Custom styling:

        >>> # Add a boundary line with custom appearance
        >>> x_boundary = np.array([0, 5, 10])
        >>> y_boundary = np.array([1.0, 1.5, 1.0])
        >>> LineOverlay(
        ...     x=x_boundary,
        ...     y=y_boundary,
        ...     name="Boundary",
        ...     dash="solid",
        ...     width=2.0
        ... )

        Usage in QualibrationFigure.plot:

        >>> def create_overlays(qubit_name, qubit_data):
        ...     # Generate theoretical prediction
        ...     x_vals = qubit_data['frequency'].values
        ...     y_theory = calculate_theoretical_curve(x_vals)
        ...
        ...     return [
        ...         LineOverlay(
        ...             x=x_vals,
        ...             y=y_theory,
        ...             name="Theory",
        ...             dash="dashdot",
        ...             width=1.5
        ...         )
        ...     ]
        >>>
        >>> fig = QualibrationFigure.plot(
        ...     dataset,
        ...     x='frequency',
        ...     data_var='response',
        ...     overlays=create_overlays
        ... )

    Notes:
        - Lines are drawn with mode="lines" (no markers on points)
        - Additional line styling can be passed via the `**style` parameter in
          `add_to()` using a "line" dict (e.g., {"line": {"color": "red"}})
        - Style overrides take precedence over class attributes
        - For fitted curves that should contribute to residual calculations, use
          FitOverlay instead
        - For simple straight reference lines spanning the plot, use RefLine instead

    See Also:
        FitOverlay : For fitted curve overlays with residual support
        RefLine : For reference lines spanning the plot
        ScatterOverlay : For scatter point overlays
    """

    x: np.ndarray
    y: np.ndarray
    name: str | None = None
    dash: str | None = "dash"
    width: float | None = None
    show_legend: bool = True

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, **style):
        line_cfg = {
            "dash": self.dash,
            "width": self.width or theme.line_width,
            **style.get("line", {}),
        }
        # Allow a direct color override via style or class-level line dict
        if "color" in style:
            line_cfg["color"] = style["color"]
        fig.add_trace(
            go.Scatter(
                x=self.x,
                y=self.y,
                name=self.name,
                mode="lines",
                line=line_cfg,
                legendgroup=style.get("legendgroup"),
                showlegend=style.get("showlegend", self.show_legend),
            ),
            row=row,
            col=col,
        )


@dataclass
class RefLine(Overlay):
    """Reference line overlay for vertical and/or horizontal lines.

    Overview:
    - Axis-aware lines via Plotly `add_vline` / `add_hline` that span the
      subplot's axis limits. Lines respect zoom/autorange and explicit axis ranges.
    - Useful for marking thresholds, target values, crosshairs, or baseline levels.

    Attributes:
    - x (float | None): X-coordinate for a vertical line. If None, no vertical line.
    - y (float | None): Y-coordinate for a horizontal line. If None, no horizontal line.
    - name (str | None): Optional name (currently unused in legend).
    - dash (str): Line dash style (e.g., "solid", "dot", "dash", "longdash"). Default "dot".
    - width (float | None): Line width in px. If None, uses `theme.line_width`.
    - color (str | None): Line color string. Accepts any Plotly-compatible color
      (e.g., "#FF0000", "red", "rgb(255,0,0)"). If None, theme/default color is used.

    Styling and precedence:
    - Base style: `{dash, width or theme.line_width}`.
    - If `color` is provided on the overlay, it is included.
    - Plot-level overrides passed via `style['line']` take precedence over all
      overlay attributes (including `color`).

    Subplots:
    - `row` and `col` route the line to the correct subplot created with
      `plotly.subplots.make_subplots`.

    Examples:
    >>> # Vertical reference line at x=5 with a hex color
    >>> RefLine(x=5.0, color="#FF5733")
    >>>
    >>> # Override color/width at plot-time (override wins over overlay settings)
    >>> fig = make_subplots(rows=1, cols=1)
    >>> RefLine(x=5.0, color="#FF5733").add_to(fig, row=1, col=1, theme=theme,
    ...     line={"color": "blue", "width": 3})
    >>>
    >>> # Draw both vertical and horizontal lines (crosshair)
    >>> RefLine(x=3.0, y=0.8, dash="dash")
    """

    x: float | None = None
    y: float | None = None
    name: str | None = None
    dash: str = "dot"
    width: float | None = None
    color: str | None = None

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, **style):
        """Add reference lines to the specified subplot.

        Styling resolution order:
        1) Start with base `{dash, width or theme.line_width}`
        2) Include `color` from overlay if provided
        3) Override with `style['line']` if present

        Behavior:
        - If `x` is set, draws `add_vline(x=...)` in the given (row, col).
        - If `y` is set, draws `add_hline(y=...)` in the given (row, col).
        - If both are set, draws a crosshair.

        Returns:
        - None

        Notes:
        - Invalid `row`/`col` values or missing subplot grids will raise Plotly errors.
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
    """Add scatter points as an overlay to plots.

    ScatterOverlay adds individual data points to a plot as markers without connecting
    lines. This is useful for highlighting specific points of interest, adding
    additional measurement data, or overlaying computed values on top of existing plots.

    Unlike the main plot data, scatter overlays are explicitly positioned and can have
    custom styling independent of the theme.

    Attributes:
        x (np.ndarray): Array of x-coordinates for the scatter points.
        y (np.ndarray): Array of y-coordinates for the scatter points. Must have the
            same length as x.
        name (str | None): Name for the scatter points in the legend. If None, the
            scatter points will not appear in the legend.
        marker_size (float | None): Size of the marker points in pixels. If None,
            uses theme.marker_size.
        marker_symbol (str | None): Symbol/shape for the markers. Common options include
            "circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down",
            "star", "hexagon", etc. If None, uses Plotly's default (circle). Can be
            overridden by passing {"marker": {"symbol": "..."}} in style overrides.

    Examples:
        Basic scatter overlay:

        >>> # Highlight specific measurement points
        >>> x_points = np.array([1.5, 2.3, 4.1])
        >>> y_points = np.array([0.5, 0.8, 0.3])
        >>> ScatterOverlay(x=x_points, y=y_points, name="Key Points")

        Mark optimal values from analysis:

        >>> # Mark the optimal point found by optimization
        >>> optimal_x = np.array([optimal_amplitude])
        >>> optimal_y = np.array([optimal_response])
        >>> ScatterOverlay(
        ...     x=optimal_x,
        ...     y=optimal_y,
        ...     name="Optimal Point",
        ...     marker_size=12,
        ...     marker_symbol="star"
        ... )

        Usage in QualibrationFigure.plot:

        >>> def create_overlays(qubit_name, qubit_data):
        ...     # Extract key measurement points
        ...     key_points_x, key_points_y = extract_key_points(qubit_data)
        ...
        ...     return [
        ...         ScatterOverlay(
        ...             x=key_points_x,
        ...             y=key_points_y,
        ...             name="Measurements",
        ...             marker_size=8,
        ...             marker_symbol="diamond"
        ...         )
        ...     ]
        >>>
        >>> fig = QualibrationFigure.plot(
        ...     dataset,
        ...     x='frequency',
        ...     data_var='response',
        ...     overlays=create_overlays
        ... )

    Notes:
        - Scatter points are drawn with mode="markers" (no connecting lines)
        - Additional marker styling can be passed via the `**style` parameter in
          `add_to()` using a "marker" dict (e.g., {"marker": {"color": "red", "symbol": "x"}})
        - Style overrides take precedence over class attributes (including marker_symbol)
        - If you need lines connecting points, use LineOverlay instead
        - For single reference points, consider using RefLine with both x and y set

    See Also:
        LineOverlay : For overlays with connected lines
        RefLine : For reference lines spanning the plot
        FitOverlay : For fitted curve overlays
    """

    x: np.ndarray
    y: np.ndarray
    name: str | None = None
    marker_size: float | None = None
    marker_symbol: str | None = None

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, **style):
        marker_config = {
            "size": self.marker_size or theme.marker_size,
        }
        if self.marker_symbol is not None:
            marker_config["symbol"] = self.marker_symbol

        # Apply style overrides on top (allowing them to override marker_symbol)
        marker_config.update(style.get("marker", {}))

        fig.add_trace(
            go.Scatter(
                x=self.x,
                y=self.y,
                name=self.name,
                mode="markers",
                marker=marker_config,
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
    """Add fitted curve overlays to 1D plots with optional parameter display.

    FitOverlay is designed to visualize fitted models alongside raw data in 1D plots.
    It draws the fitted curve as a line trace and optionally displays fit parameters
    as a text box. This overlay is commonly used in calibration plots to show the
    results of curve fitting (e.g., oscillations, exponential decays, resonances).

    The overlay can provide fit data for residual calculations. If `y_fit` is provided,
    the QualibrationFigure will automatically use it to compute residuals when
    `residuals=True` is set.

    Attributes:
        y_fit (np.ndarray | None): Array of fitted y-values corresponding to the x-axis
            of the plot. If None, no fit line is drawn. This should have the same length
            as the x-axis data.
        params (Mapping[str, Any] | None): Dictionary of fit parameters (e.g., amplitude,
            frequency, phase). If both params and formatter are provided, a text box with
            formatted parameters will be added to the plot. If None, no text is displayed.
        formatter (Callable[[Mapping[str, Any]], str] | None): Function that takes the
            params dictionary and returns a formatted string for display. Only used if
            params is also provided. The function should return a string suitable for
            display in a plot annotation.
        name (str): Name for the fit line in the legend. Default is "fit".
        dash (str): Line dash style. Options include "solid", "dot", "dash", "longdash",
            "dashdot", "longdashdot". Default is "dash".
        width (float | None): Line width in pixels. If None, uses theme.line_width.

    Examples:
        Basic fit overlay without parameters:

        >>> # Compute fitted curve from your fitting function
        >>> fitted_curve = calculate_fit_curve(**parameters)
        >>> FitOverlay(y_fit=fitted_curve, name="Fit")

        Fit overlay with parameter display:

        >>> def format_params(p):
        ...     return f"A = {p['amplitude']:.3f}\\nf = {p['frequency']:.3f} MHz"
        >>>
        >>> fit_params = {'amplitude': 0.523, 'frequency': 5.234}
        >>> fitted_curve = calculate_fit_curve(**fit_params)
        >>> FitOverlay(
        ...     y_fit=fitted_curve,
        ...     params=fit_params,
        ...     formatter=format_params,
        ...     name="Oscillation Fit"
        ... )

        Custom styling:

        >>> fitted_curve = calculate_fit_curve(**parameters)
        >>> FitOverlay(
        ...     y_fit=fitted_curve,
        ...     name="Model",
        ...     dash="solid",
        ...     width=2.0
        ... )

        Usage in QualibrationFigure.plot:

        >>> def create_fit_overlay(qubit_name, qubit_data):
        ...     # Extract fit parameters from your fit dataset
        ...     fit_params = extract_fit_parameters(qubit_name)
        ...
        ...     # Compute fitted curve using your fitting function
        ...     fitted_curve = calculate_fit_curve(**fit_params)
        ...
        ...     return [FitOverlay(y_fit=fitted_curve, name="Fit")]
        >>>
        >>> fig = QualibrationFigure.plot(
        ...     dataset,
        ...     x='amp_prefactor',
        ...     data_var='I',
        ...     overlays=create_fit_overlay
        ... )

    Notes:
        - The x-values for the fit are provided by the plotting system via the `x`
          parameter in `add_to()`, not stored in the overlay itself
        - If both `params` and `formatter` are provided, a TextBoxOverlay is automatically
          created and positioned in the top-right corner of the subplot
        - The fit line is drawn as a continuous line (mode="lines") to distinguish it
          from raw data points
        - When used with `residuals=True` in QualibrationFigure.plot, the `y_fit` data
          is used to compute residuals as (data - fit)

    See Also:
        LineOverlay : For arbitrary line overlays without fit semantics
        TextBoxOverlay : For standalone text annotations
        QualibrationFigure.plot : Main plotting function that uses overlays
    """

    y_fit: np.ndarray | None = None
    params: Mapping[str, Any] | None = None
    formatter: Callable[[Mapping[str, Any]], str] | None = None
    name: str = "fit"
    dash: str = "dash"
    width: float | None = None

    def add_to(self, fig: go.Figure, *, row: int, col: int, theme, x=None, **style):
        if self.y_fit is not None and x is not None:
            line_cfg = {
                "dash": self.dash,
                "width": self.width or theme.line_width,
                **style.get("line", {}),
            }
            # Allow a direct color override via style
            if "color" in style:
                line_cfg["color"] = style["color"]
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=self.y_fit,
                    name=self.name,
                    mode="lines",
                    line=line_cfg,
                    legendgroup=style.get("legendgroup"),
                    showlegend=style.get("showlegend", True),
                ),
                row=row,
                col=col,
            )
        if self.params is not None and self.formatter is not None:
            text = self.formatter(self.params)
            TextBoxOverlay(text=text).add_to(fig, row=row, col=col, theme=theme)