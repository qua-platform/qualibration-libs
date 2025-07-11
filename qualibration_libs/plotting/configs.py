"""
Enhanced plotting configuration dataclasses for standardized quantum calibration plotting.

This module extends the basic PlotConfig system to support complex plotting scenarios
including heatmaps, dual axes, overlays, and specialized quantum measurement visualizations.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# =============================================================================
# Core Configuration Models
# =============================================================================


class LayoutConfig(BaseModel):
    title: str
    x_axis_title: str
    y_axis_title: str
    legend: Dict = Field(default_factory=dict)


class TraceConfig(BaseModel):
    plot_type: Literal["scatter", "heatmap", "line"]
    x_source: str
    y_source: str
    z_source: Optional[str] = None  # For heatmaps
    name: str
    mode: Optional[str] = "lines+markers"
    hover_template: Optional[str] = None
    custom_data_sources: List[str] = Field(default_factory=list)
    visible: bool = True


class PlotConfig(BaseModel):
    layout: LayoutConfig
    traces: List[TraceConfig]
    fit_traces: List[TraceConfig] = Field(default_factory=list)


# =============================================================================
# Specialized and Component Models
# =============================================================================


class ColorbarConfig(BaseModel):
    """Configuration for plot colorbars."""

    title: str = "|IQ|"
    x_offset: float = 0.03
    width: float = 0.02
    height_ratio: float = 0.90
    thickness: int = 14
    position: Literal["right", "bottom"] = "right"
    ticks: Literal["inside", "outside"] = "outside"
    ticklabelposition: Literal["inside", "outside"] = "outside"


class HeatmapTraceConfig(TraceConfig):
    """Extended trace configuration for heatmap plots."""

    plot_type: Literal["heatmap"] = "heatmap"
    colorscale: str = "Viridis"
    colorbar: Optional[ColorbarConfig] = Field(default_factory=ColorbarConfig)
    zmin_percentile: float = 2.0  # Robust color scaling
    zmax_percentile: float = 98.0


class DualAxisConfig(BaseModel):
    """Configuration for dual x-axis support (e.g., frequency + detuning)."""

    enabled: bool = True
    top_axis_title: str
    top_axis_source: str  # Data source for top axis values
    top_axis_format: str = "{:.2f}"  # Format string for tick labels
    overlay_offset: int = 100  # Plotly axis offset for overlays


# =============================================================================
# Overlay Configuration Models
# =============================================================================


class OverlayConfig(BaseModel):
    """Configuration for plot overlays (fit lines, markers, etc.)."""

    type: Literal["line", "marker", "annotation"]
    condition_source: str  # Data source to check for overlay condition (e.g., "outcome")
    condition_value: Any = "successful"  # Value that enables the overlay
    style: Dict[str, Any] = Field(default_factory=dict)


class LineOverlayConfig(OverlayConfig):
    """Configuration for line overlays (vertical/horizontal lines)."""

    type: Literal["line"] = "line"
    orientation: Literal["vertical", "horizontal"]
    position_source: str  # Data source for line position
    line_style: Dict[str, Any] = Field(
        default_factory=lambda: {"color": "#FF0000", "width": 2.5, "dash": "dash"}
    )


class MarkerOverlayConfig(OverlayConfig):
    """Configuration for marker overlays (sweet spots, optimal points)."""

    type: Literal["marker"] = "marker"
    x_source: str
    y_source: str
    marker_style: Dict[str, Any] = Field(
        default_factory=lambda: {"symbol": "x", "color": "#FF00FF", "size": 15}
    )


# =============================================================================
# Main Plot Configuration Models
# =============================================================================


class SpectroscopyConfig(PlotConfig):
    """Configuration for 1D spectroscopy plots (node 02a style)."""

    dual_axis: Optional[DualAxisConfig] = None


class HeatmapConfig(PlotConfig):
    """Configuration for 2D heatmap plots (nodes 02b, 02c style)."""

    traces: List[HeatmapTraceConfig]
    overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]] = Field(
        default_factory=list
    )
    subplot_spacing: Dict[str, float] = Field(
        default_factory=lambda: {"horizontal": 0.15, "vertical": 0.12}
    )
    dual_axis: Optional[DualAxisConfig] = None


class ChevronConfig(PlotConfig):
    """Configuration for chevron-style plots (node 04b power Rabi style)."""

    traces: List[Union[TraceConfig, HeatmapTraceConfig]]
    overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]] = Field(
        default_factory=list
    )
    dual_axis: Optional[DualAxisConfig] = None
    subplot_spacing: Dict[str, float] = Field(
        default_factory=lambda: {"horizontal": 0.1, "vertical": 0.2}
    )


# =============================================================================
# Utility Models
# =============================================================================


class PlotDimensions(BaseModel):
    """Standard plot dimensions and styling."""

    fig_width: int = 1500
    fig_height: int = 900
    subplot_width: int = 400
    subplot_height: int = 400
    min_width: int = 1000
    margin: Dict[str, int] = Field(
        default_factory=lambda: {"l": 60, "r": 60, "t": 80, "b": 60}
    )


# =============================================================================
# Standard Configuration Instances
# =============================================================================

STANDARD_SPECTROSCOPY_CONFIG = SpectroscopyConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy",
        x_axis_title="RF frequency [GHz]",
        y_axis_title="R = √(I² + Q²) [mV]",
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="full_freq_GHz",
            y_source="IQ_abs_mV",
            name="Raw Data",
            style={"color": "#1f77b4"},
        )
    ],
    fit_traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="full_freq_GHz",
            y_source="fitted_data_mV",
            name="Fit",
            mode="lines",
            style={"color": "#FF0000", "dash": "dash"},
        )
    ],
    dual_axis=DualAxisConfig(
        top_axis_title="Detuning [MHz]", top_axis_source="detuning_MHz"
    ),
)

STANDARD_FLUX_HEATMAP_CONFIG = HeatmapConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy: Flux vs Frequency",
        x_axis_title="Flux bias [V]",
        y_axis_title="RF frequency [GHz]",
    ),
    traces=[
        HeatmapTraceConfig(
            plot_type="heatmap",
            x_source="flux_bias",
            y_source="freq_GHz",
            z_source="IQ_abs",
            name="Raw Data",
        )
    ],
    overlays=[
        LineOverlayConfig(
            orientation="vertical",
            condition_source="outcome",
            position_source="idle_offset",
            line_style={"color": "#FF0000", "width": 2.5, "dash": "dash"},
        ),
        LineOverlayConfig(
            orientation="vertical",
            condition_source="outcome",
            position_source="flux_min",
            line_style={"color": "#800080", "width": 2.5, "dash": "dash"},
        ),
        MarkerOverlayConfig(
            condition_source="outcome",
            x_source="idle_offset",
            y_source="sweet_spot_frequency_GHz",
            marker_style={"symbol": "x", "color": "#FF00FF", "size": 15},
        ),
    ],
    dual_axis=DualAxisConfig(
        top_axis_title="Current [A]",
        top_axis_source="attenuated_current",
        top_axis_format="{:.6f}",
    ),
)

STANDARD_AMPLITUDE_HEATMAP_CONFIG = HeatmapConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy: Power vs Frequency",
        x_axis_title="Frequency [GHz]",
        y_axis_title="Power [dBm]",
    ),
    traces=[
        HeatmapTraceConfig(
            plot_type="heatmap",
            x_source="freq_GHz",
            y_source="power_dbm",
            z_source="IQ_abs_norm",
            name="Raw Data",
        )
    ],
    overlays=[
        LineOverlayConfig(
            orientation="vertical",
            condition_source="outcome",
            position_source="res_freq_GHz",
            line_style={"color": "#FF0000", "width": 2, "dash": "dash"},
        ),
        LineOverlayConfig(
            orientation="horizontal",
            condition_source="outcome",
            position_source="optimal_power",
            line_style={"color": "#FF00FF", "width": 2},
        ),
    ],
)

STANDARD_CHEVRON_CONFIG = ChevronConfig(
    layout=LayoutConfig(
        title="Power Rabi",
        x_axis_title="Pulse amplitude [mV]",
        y_axis_title="Rotated I quadrature [mV]",
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="amp_mV",
            y_source="I_mV",
            name="Raw Data",
            style={"color": "#1f77b4"},
            hover_template="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata[0]:.3f}<br>%{y:.3f} mV<extra></extra>",
            custom_data_sources=["amp_prefactor"],
        )
    ],
    fit_traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="amp_mV",
            y_source="fitted_data_mV",
            name="Fit",
            mode="lines",
            style={"color": "#FF0000", "width": 2},
        )
    ],
    dual_axis=DualAxisConfig(
        top_axis_title="Amplitude prefactor", top_axis_source="amp_prefactor"
    ),
)


POWER_RABI_CONFIG = PlotConfig(
    layout=LayoutConfig(
        title="Power Rabi",
        x_axis_title="Pulse amplitude [mV]",
        y_axis_title="Signal [mV]"  # Will be updated based on data type
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",  # Will be changed to heatmap for 2D data
            x_source="amp_mV",
            y_source="I_mV",  # Will be determined dynamically
            name="Raw Data",
            mode="lines+markers",
            style={"color": "#1f77b4"}
        )
    ],
    fit_traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="amp_mV",
            y_source="fitted_data_mV", 
            name="Fit",
            mode="lines",
            style={"color": "#FF0000", "width": 2}
        )
    ]
)


class PlotStyling(BaseModel):
    """A collection of all styling configurations for plots."""

    # General Colors
    raw_data_color: str = "#1f77b4"
    fit_color: str = "#FF0000"
    idle_offset_color: str = "#FF0000"
    min_offset_color: str = "#800080"
    sweet_spot_color: str = "#FF00FF"
    optimal_power_color: str = "#FF00FF"
    resonator_freq_color: str = "#FF0000"
    heatmap_colorscale: str = "Viridis"

    # Matplotlib Specific Styling
    matplotlib_fig_width: int = 15
    matplotlib_fig_height: int = 9
    matplotlib_fit_linewidth: float = 2.0
    matplotlib_idle_offset_linewidth: float = 2.5
    matplotlib_idle_offset_linestyle: str = "dashed"
    matplotlib_sweet_spot_marker: str = "*"
    matplotlib_sweet_spot_markersize: int = 18
    matplotlib_raw_data_alpha: float = 0.5
    matplotlib_optimal_power_linestyle: str = "-"
    matplotlib_resonator_freq_linestyle: str = "--"

    # Plotly Specific Styling
    plotly_fig_width: int = 1500
    plotly_fig_height: int = 900
    plotly_subplot_width: int = 400
    plotly_subplot_height: int = 400
    plotly_min_width: int = 1000
    plotly_margin: Dict[str, int] = Field(
        default_factory=lambda: {"l": 60, "r": 60, "t": 80, "b": 60}
    )
    plotly_horizontal_spacing: float = 0.15
    plotly_vertical_spacing: float = 0.12
    plotly_annotation_font_size: int = 16
    plotly_axis_offset: int = 100

    # Plotly Line and Marker Styles
    plotly_fit_linewidth: float = 2.5
    plotly_fit_linestyle: str = "dash"
    plotly_sweet_spot_marker_symbol: str = "x"
    plotly_sweet_spot_marker_size: int = 15
    plotly_good_fit_marker_size: int = 8
    plotly_resonator_freq_linewidth: float = 2.0
    plotly_resonator_freq_linestyle: str = "dash"
    plotly_optimal_power_linewidth: float = 2.0

    # Plotly Colorbar
    plotly_colorbar_thickness: int = 14
    plotly_colorbar_x_offset: float = 0.03
    plotly_colorbar_width: float = 0.02
    plotly_colorbar_height_ratio: float = 0.90
    colorbar_config: ColorbarConfig = Field(default_factory=ColorbarConfig)


# Configuration for the phase vs. frequency plot
PHASE_VS_FREQ_CONFIG = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy (Phase)",
        x_axis_title="RF Frequency [GHz]",
        y_axis_title="Phase [rad]",
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            name="Raw Phase",
            x_source="full_freq_ghz",
            y_source="phase",
            custom_data_sources=["detuning_mhz"],
            hover_template="<b>Freq</b>: %{x:.4f} GHz<br>" +
                           "<b>Detuning</b>: %{customdata[0]:.2f} MHz<br>" +
                           "<b>Phase</b>: %{y:.3f} rad<extra></extra>",
            style={"color": "#1f77b4"},
        )
    ],
)

# Configuration for the amplitude vs. frequency plot, including the fit
AMPLITUDE_VS_FREQ_CONFIG = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy (Amplitude + Fit)",
        x_axis_title="RF Frequency [GHz]",
        y_axis_title="R = sqrt(I² + Q²) [mV]",
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            name="Raw Amplitude",
            x_source="full_freq_ghz",
            y_source="iq_abs_mv",
            custom_data_sources=["detuning_mhz"],
            hover_template="<b>Freq</b>: %{x:.4f} GHz<br>" +
                           "<b>Detuning</b>: %{customdata[0]:.2f} MHz<br>" +
                           "<b>Amplitude</b>: %{y:.3f} mV<extra></extra>",
            style={"color": "#1f77b4"},
        )
    ],
    fit_traces=[
        TraceConfig(
            plot_type="line",
            name="Fit",
            x_source="full_freq_ghz",
            y_source="fitted_curve_mv",
            mode="lines",
            style={"color": "#FF0000", "dash": "dash"},
            hover_template="<b>Fit</b><extra></extra>"
        )
    ]
) 

# Configuration for amplitude vs power heatmap plot
AMPLITUDE_VS_POWER_CONFIG = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy vs Power",
        x_axis_title="Frequency (GHz)", 
        y_axis_title="Power (dBm)"
    ),
    traces=[
        TraceConfig(
            plot_type="heatmap",
            x_source="freq_GHz",
            y_source="power_dbm",
            z_source="IQ_abs_norm",
            name="Raw Data",
            style={"colorscale": "Viridis"}
        )
    ],
    fit_traces=[
        # Fit overlays can be added here if needed
        # TraceConfig for vertical line at resonance frequency
        # TraceConfig for horizontal line at optimal power
    ]
)

# Configuration for flux vs frequency heatmap plot
FLUX_VS_FREQUENCY_CONFIG = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy vs Flux",
        x_axis_title="Flux bias [V]",
        y_axis_title="RF frequency [GHz]"
    ),
    traces=[
        TraceConfig(
            plot_type="heatmap",
            x_source="flux_bias",
            y_source="freq_GHz", 
            z_source="IQ_abs",
            name="Raw Data",
            style={"colorscale": "Viridis"}
        )
    ],
    fit_traces=[
        # Fit overlays can be added here if needed
        # TraceConfig for vertical lines at idle_offset, flux_min
        # TraceConfig for marker at sweet spot
    ]
)