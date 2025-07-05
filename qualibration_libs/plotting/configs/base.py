"""
Enhanced base configuration classes for the unified plotting framework.

This module provides the foundational configuration classes that support
both simple plotting scenarios and complex adaptive plotting requirements.
"""

from typing import Dict, List, Literal, Optional, Union, Any, Callable
from pydantic import BaseModel, Field
from .visual_standards import (
    Colors, LineStyles, AxisLabels, HoverTemplates, 
    get_raw_data_style, get_fit_line_style, get_optimal_marker_style
)


class TraceConfig(BaseModel):
    """Configuration for individual plot traces (lines, scatter, heatmap)."""
    
    plot_type: Literal["scatter", "heatmap", "line"]
    x_source: str                           # Data source for x-axis
    y_source: str                           # Data source for y-axis  
    z_source: Optional[str] = None          # Data source for z-axis (heatmaps)
    name: str                               # Display name for trace
    mode: Optional[str] = "lines+markers"   # Plotly mode
    style: Dict[str, Any] = Field(default_factory=get_raw_data_style)
    hover_template: Optional[str] = None
    custom_data_sources: List[str] = Field(default_factory=list)
    visible: bool = True
    
    # Conditional visibility
    condition_source: Optional[str] = None   # Data source for condition check
    condition_value: Any = "successful"      # Value that enables the trace


class LayoutConfig(BaseModel):
    """Configuration for plot layout (titles, axes, styling)."""
    
    title: str
    x_axis_title: str
    y_axis_title: str
    legend: Dict[str, Any] = Field(default_factory=dict)
    
    # Subplot-specific settings
    subplot_spacing: Optional[Dict[str, float]] = None
    margins: Optional[Dict[str, int]] = None


class DualAxisConfig(BaseModel):
    """Configuration for dual x-axis support (e.g., frequency + detuning)."""
    
    enabled: bool = True
    top_axis_title: str
    top_axis_source: str                    # Data source for top axis values
    top_axis_format: str = "{:.2f}"        # Format string for tick labels
    overlay_offset: int = 100               # Plotly axis offset for overlays


class OverlayConfig(BaseModel):
    """Base configuration for plot overlays (fit lines, markers, etc.)."""
    
    type: Literal["line", "marker", "annotation"]
    condition_source: str                   # Data source to check for overlay condition
    condition_value: Any = "successful"     # Value that enables the overlay
    style: Dict[str, Any] = Field(default_factory=dict)


class LineOverlayConfig(OverlayConfig):
    """Configuration for line overlays (vertical/horizontal lines)."""
    
    type: Literal["line"] = "line"
    orientation: Literal["vertical", "horizontal"]
    position_source: str                    # Data source for line position
    line_style: Dict[str, Any] = Field(default_factory=lambda: {
        "color": Colors.FIT_LINE,
        "width": LineStyles.OVERLAY_LINE_WIDTH,
        "dash": LineStyles.OVERLAY_LINE_DASH
    })


class MarkerOverlayConfig(OverlayConfig):
    """Configuration for marker overlays (sweet spots, optimal points)."""
    
    type: Literal["marker"] = "marker"
    x_source: str
    y_source: str
    marker_style: Dict[str, Any] = Field(default_factory=get_optimal_marker_style)


class PlotConfig(BaseModel):
    """Base configuration for all plot types."""
    
    layout: LayoutConfig
    traces: List[TraceConfig]
    fit_traces: List[TraceConfig] = Field(default_factory=list)
    overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]] = Field(default_factory=list)
    dual_axis: Optional[DualAxisConfig] = None
    
    # Plot type identification
    plot_family: str = "base"               # Used for engine routing


class ColorbarConfig(BaseModel):
    """Configuration for heatmap colorbars."""
    
    title: str = "|IQ|"
    x_offset: float = 0.03                  # Distance from subplot
    width: float = 0.02                     # Width as fraction of figure
    height_ratio: float = 0.90              # Height as fraction of subplot
    thickness: int = 14                     # Thickness in pixels
    position: Literal["right", "bottom"] = "right"
    ticks: Literal["inside", "outside"] = "outside"
    ticklabelposition: Literal["inside", "outside"] = "outside"
    
    # Z-axis scaling
    zmin_percentile: float = 2.0            # Robust color scaling
    zmax_percentile: float = 98.0


class HeatmapTraceConfig(TraceConfig):
    """Extended trace configuration for heatmap plots."""
    
    plot_type: Literal["heatmap"] = "heatmap"
    colorscale: str = Colors.HEATMAP_COLORSCALE
    colorbar: Optional[ColorbarConfig] = Field(default_factory=ColorbarConfig)
    zmin_percentile: float = 2.0            # Robust color scaling
    zmax_percentile: float = 98.0


class AdaptiveTraceConfig(TraceConfig):
    """Trace configuration that adapts based on data dimensionality."""
    
    # 1D configuration
    line_mode: str = "lines+markers"
    line_style: Dict[str, Any] = Field(default_factory=get_raw_data_style)
    
    # 2D configuration
    heatmap_colorscale: str = Colors.HEATMAP_COLORSCALE
    heatmap_colorbar: Optional[ColorbarConfig] = Field(default_factory=ColorbarConfig)
    
    # Adaptation logic
    adaptation_dimension: str               # Dimension to check for 1D vs 2D
    adaptation_threshold: int = 1           # Threshold for switching to 2D


class DimensionalityDetector:
    """Utility class for detecting plot dimensionality."""
    
    @staticmethod
    def detect_power_rabi_dimensionality(ds_raw) -> str:
        """Detect if Power Rabi data is 1D or 2D."""
        if "nb_of_pulses" in ds_raw.dims and ds_raw.sizes.get("nb_of_pulses", 1) > 1:
            return "2D"
        return "1D"
    
    @staticmethod
    def get_recommended_config(experiment_type: str, ds_raw) -> str:
        """Get recommended configuration type based on experiment and data."""
        if experiment_type == "power_rabi":
            dimensionality = DimensionalityDetector.detect_power_rabi_dimensionality(ds_raw)
            return f"power_rabi_{dimensionality.lower()}"
        
        # Standard cases
        experiment_config_map = {
            "resonator_spectroscopy": "spectroscopy",
            "resonator_spectroscopy_vs_amplitude": "heatmap",
            "resonator_spectroscopy_vs_flux": "heatmap",
        }
        
        return experiment_config_map.get(experiment_type, "base")


# ===== FACTORY FUNCTIONS =====

def create_standard_trace(
    x_source: str,
    y_source: str,
    name: str,
    plot_type: str = "scatter",
    style_override: Optional[Dict[str, Any]] = None
) -> TraceConfig:
    """Create a standard trace configuration with visual standards."""
    
    style = get_raw_data_style()
    if style_override:
        style.update(style_override)
    
    return TraceConfig(
        plot_type=plot_type,
        x_source=x_source,
        y_source=y_source,
        name=name,
        style=style
    )


def create_fit_trace(
    x_source: str,
    y_source: str,
    name: str = "Fit",
    style_override: Optional[Dict[str, Any]] = None
) -> TraceConfig:
    """Create a fit trace configuration with standard fit styling."""
    
    style = get_fit_line_style()
    if style_override:
        style.update(style_override)
    
    return TraceConfig(
        plot_type="line",
        x_source=x_source,
        y_source=y_source,
        name=name,
        mode="lines",
        style=style,
        condition_source="outcome",  # Only show if fit was successful
        condition_value="successful"
    )


def create_heatmap_trace(
    x_source: str,
    y_source: str,
    z_source: str,
    name: str,
    colorbar_title: str = "|IQ|"
) -> HeatmapTraceConfig:
    """Create a heatmap trace configuration with standard styling."""
    
    colorbar = ColorbarConfig(title=colorbar_title)
    
    return HeatmapTraceConfig(
        x_source=x_source,
        y_source=y_source,
        z_source=z_source,
        name=name,
        colorbar=colorbar
    )


def create_optimal_marker(
    x_source: str,
    y_source: str,
    condition_source: str = "outcome"
) -> MarkerOverlayConfig:
    """Create an optimal point marker overlay."""
    
    return MarkerOverlayConfig(
        x_source=x_source,
        y_source=y_source,
        condition_source=condition_source
    )


def create_vertical_line(
    position_source: str,
    condition_source: str = "outcome"
) -> LineOverlayConfig:
    """Create a vertical line overlay."""
    
    return LineOverlayConfig(
        orientation="vertical",
        position_source=position_source,
        condition_source=condition_source
    )


def create_dual_axis(
    top_axis_title: str,
    top_axis_source: str,
    format_string: str = "{:.2f}"
) -> DualAxisConfig:
    """Create a dual axis configuration."""
    
    return DualAxisConfig(
        top_axis_title=top_axis_title,
        top_axis_source=top_axis_source,
        top_axis_format=format_string
    )