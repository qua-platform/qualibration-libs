"""
Enhanced plotting configuration dataclasses for standardized quantum calibration plotting.

This module extends the basic PlotConfig system to support complex plotting scenarios
including heatmaps, dual axes, overlays, and specialized quantum measurement visualizations.
"""

from typing import Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field

# Re-export existing basic configs for backward compatibility
from .standard_plotter import TraceConfig, LayoutConfig, PlotConfig


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


class DualAxisConfig(BaseModel):
    """Configuration for dual x-axis support (e.g., frequency + detuning)."""
    enabled: bool = True
    top_axis_title: str
    top_axis_source: str  # Data source for top axis values
    top_axis_format: str = "{:.2f}"  # Format string for tick labels
    overlay_offset: int = 100  # Plotly axis offset for overlays


class HeatmapTraceConfig(TraceConfig):
    """Extended trace configuration for heatmap plots."""
    plot_type: Literal["heatmap"] = "heatmap"
    colorscale: str = "Viridis"
    colorbar: Optional[ColorbarConfig] = Field(default_factory=ColorbarConfig)
    zmin_percentile: float = 2.0  # Robust color scaling
    zmax_percentile: float = 98.0
    
    
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
    line_style: Dict[str, Any] = Field(default_factory=lambda: {
        "color": "#FF0000", "width": 2.5, "dash": "dash"
    })


class MarkerOverlayConfig(OverlayConfig):
    """Configuration for marker overlays (sweet spots, optimal points)."""
    type: Literal["marker"] = "marker"
    x_source: str
    y_source: str
    marker_style: Dict[str, Any] = Field(default_factory=lambda: {
        "symbol": "x", "color": "#FF00FF", "size": 15
    })


class SpectroscopyConfig(PlotConfig):
    """Configuration for 1D spectroscopy plots (node 02a style)."""
    dual_axis: Optional[DualAxisConfig] = None
    

class HeatmapConfig(PlotConfig):
    """Configuration for 2D heatmap plots (nodes 02b, 02c style)."""
    traces: List[HeatmapTraceConfig]
    overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]] = Field(default_factory=list)
    subplot_spacing: Dict[str, float] = Field(default_factory=lambda: {
        "horizontal": 0.15, "vertical": 0.12
    })
    dual_axis: Optional[DualAxisConfig] = None
    

class ChevronConfig(PlotConfig):
    """Configuration for chevron-style plots (node 04b power Rabi style)."""
    traces: List[Union[TraceConfig, HeatmapTraceConfig]]
    overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]] = Field(default_factory=list)
    dual_axis: Optional[DualAxisConfig] = None
    subplot_spacing: Dict[str, float] = Field(default_factory=lambda: {
        "horizontal": 0.1, "vertical": 0.2
    })


class PlotDimensions(BaseModel):
    """Standard plot dimensions and styling."""
    fig_width: int = 1500
    fig_height: int = 900
    subplot_width: int = 400
    subplot_height: int = 400
    min_width: int = 1000
    margin: Dict[str, int] = Field(default_factory=lambda: {
        "l": 60, "r": 60, "t": 80, "b": 60
    })


# Standard configurations for common plot types
STANDARD_SPECTROSCOPY_CONFIG = SpectroscopyConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy",
        x_axis_title="RF frequency [GHz]", 
        y_axis_title="R = √(I² + Q²) [mV]"
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="full_freq_GHz",
            y_source="IQ_abs_mV", 
            name="Raw Data",
            style={"color": "#1f77b4"}
        )
    ],
    fit_traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="full_freq_GHz",
            y_source="fitted_data_mV",
            name="Fit",
            mode="lines",
            style={"color": "#FF0000", "dash": "dash"}
        )
    ],
    dual_axis=DualAxisConfig(
        top_axis_title="Detuning [MHz]",
        top_axis_source="detuning_MHz"
    )
)

STANDARD_FLUX_HEATMAP_CONFIG = HeatmapConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy: Flux vs Frequency",
        x_axis_title="Flux bias [V]",
        y_axis_title="RF frequency [GHz]"
    ),
    traces=[
        HeatmapTraceConfig(
            plot_type="heatmap",
            x_source="flux_bias",
            y_source="freq_GHz", 
            z_source="IQ_abs",
            name="Raw Data"
        )
    ],
    overlays=[
        LineOverlayConfig(
            orientation="vertical",
            condition_source="outcome",
            position_source="idle_offset",
            line_style={"color": "#FF0000", "width": 2.5, "dash": "dash"}
        ),
        LineOverlayConfig(
            orientation="vertical", 
            condition_source="outcome",
            position_source="flux_min",
            line_style={"color": "#800080", "width": 2.5, "dash": "dash"}
        ),
        MarkerOverlayConfig(
            condition_source="outcome",
            x_source="idle_offset",
            y_source="sweet_spot_frequency_GHz",
            marker_style={"symbol": "x", "color": "#FF00FF", "size": 15}
        )
    ],
    dual_axis=DualAxisConfig(
        top_axis_title="Current [A]",
        top_axis_source="attenuated_current",
        top_axis_format="{:.6f}"
    )
)

STANDARD_AMPLITUDE_HEATMAP_CONFIG = HeatmapConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy: Power vs Frequency", 
        x_axis_title="Frequency [GHz]",
        y_axis_title="Power [dBm]"
    ),
    traces=[
        HeatmapTraceConfig(
            plot_type="heatmap",
            x_source="freq_GHz",
            y_source="power_dbm",
            z_source="IQ_abs_norm", 
            name="Raw Data"
        )
    ],
    overlays=[
        LineOverlayConfig(
            orientation="vertical",
            condition_source="outcome", 
            position_source="res_freq_GHz",
            line_style={"color": "#FF0000", "width": 2, "dash": "dash"}
        ),
        LineOverlayConfig(
            orientation="horizontal",
            condition_source="outcome",
            position_source="optimal_power",
            line_style={"color": "#FF00FF", "width": 2}
        )
    ]
)

STANDARD_POWER_RABI_CONFIG = ChevronConfig(
    layout=LayoutConfig(
        title="Power Rabi",
        x_axis_title="Pulse amplitude [mV]",
        y_axis_title="Rotated I quadrature [mV]"
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="amp_mV", 
            y_source="I_mV",
            name="Raw Data",
            style={"color": "#1f77b4"},
            hover_template="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata[0]:.3f}<br>%{y:.3f} mV<extra></extra>",
            custom_data_sources=["amp_prefactor"]
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
    ],
    dual_axis=DualAxisConfig(
        top_axis_title="Amplitude prefactor",
        top_axis_source="amp_prefactor"
    )
)