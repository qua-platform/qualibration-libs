"""
Fluent configuration builder for creating plot configurations.

This module provides a developer-friendly fluent interface for creating
plot configurations, eliminating the verbose and error-prone manual
configuration creation process.
"""

from typing import Dict, Any, List, Optional, Union

from .base import (
    LayoutConfig, TraceConfig, HeatmapTraceConfig, ColorbarConfig,
    LineOverlayConfig, MarkerOverlayConfig, DualAxisConfig,
    create_standard_trace, create_fit_trace, create_heatmap_trace,
    create_optimal_marker, create_vertical_line, create_dual_axis
)
from .spectroscopy import SpectroscopyConfig
from .heatmap import HeatmapConfig
from .visual_standards import (
    AxisLabels, Colors, LineStyles,
    get_raw_data_style, get_fit_line_style, get_heatmap_style
)


class PlotConfigurationBuilder:
    """Fluent interface for creating plot configurations.
    
    This builder provides a chainable API for constructing plot configurations,
    making it easier to create complex configurations with less boilerplate code
    and better IDE support.
    
    Example:
        >>> config = (PlotConfigurationBuilder()
        ...     .title("My Experiment")
        ...     .x_axis("Frequency [GHz]")
        ...     .y_axis("Signal [mV]")
        ...     .add_raw_trace("freq", "signal")
        ...     .add_fit_trace("freq", "fitted_signal")
        ...     .build())
    """
    
    def __init__(self):
        """Initialize the builder with default values."""
        self._layout = LayoutConfig(
            title="",
            x_axis_title="",
            y_axis_title=""
        )
        self._traces: List[TraceConfig] = []
        self._fit_traces: List[TraceConfig] = []
        self._overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]] = []
        self._dual_axis: Optional[DualAxisConfig] = None
        self._experiment_type: Optional[str] = None
        self._plot_family: str = "base"
        
    def title(self, title: str) -> 'PlotConfigurationBuilder':
        """Set the plot title.
        
        Args:
            title: The main title for the plot
            
        Returns:
            Self for chaining
        """
        self._layout.title = title
        return self
    
    def x_axis(self, title: str) -> 'PlotConfigurationBuilder':
        """Set the X-axis title.
        
        Args:
            title: The X-axis label (e.g., "Frequency [GHz]")
            
        Returns:
            Self for chaining
        """
        self._layout.x_axis_title = title
        return self
    
    def y_axis(self, title: str) -> 'PlotConfigurationBuilder':
        """Set the Y-axis title.
        
        Args:
            title: The Y-axis label (e.g., "Signal [mV]")
            
        Returns:
            Self for chaining
        """
        self._layout.y_axis_title = title
        return self
    
    def subplot_spacing(
        self, 
        horizontal: Optional[float] = None, 
        vertical: Optional[float] = None
    ) -> 'PlotConfigurationBuilder':
        """Set subplot spacing for multi-qubit plots.
        
        Args:
            horizontal: Horizontal spacing between subplots (0-1)
            vertical: Vertical spacing between subplots (0-1)
            
        Returns:
            Self for chaining
        """
        if self._layout.subplot_spacing is None:
            self._layout.subplot_spacing = {}
        
        if horizontal is not None:
            self._layout.subplot_spacing["horizontal"] = horizontal
        if vertical is not None:
            self._layout.subplot_spacing["vertical"] = vertical
            
        return self
    
    def margins(
        self,
        left: Optional[int] = None,
        right: Optional[int] = None,
        top: Optional[int] = None,
        bottom: Optional[int] = None
    ) -> 'PlotConfigurationBuilder':
        """Set plot margins.
        
        Args:
            left: Left margin in pixels
            right: Right margin in pixels
            top: Top margin in pixels
            bottom: Bottom margin in pixels
            
        Returns:
            Self for chaining
        """
        if self._layout.margins is None:
            self._layout.margins = {}
            
        if left is not None:
            self._layout.margins["l"] = left
        if right is not None:
            self._layout.margins["r"] = right
        if top is not None:
            self._layout.margins["t"] = top
        if bottom is not None:
            self._layout.margins["b"] = bottom
            
        return self
    
    def legend(self, **kwargs) -> 'PlotConfigurationBuilder':
        """Configure legend properties.
        
        Args:
            **kwargs: Legend configuration options (position, orientation, etc.)
            
        Returns:
            Self for chaining
        """
        self._layout.legend.update(kwargs)
        return self
    
    def add_raw_trace(
        self,
        x_source: str,
        y_source: str,
        name: str = "Raw Data",
        style_override: Optional[Dict[str, Any]] = None,
        hover_template: Optional[str] = None,
        custom_data_sources: Optional[List[str]] = None
    ) -> 'PlotConfigurationBuilder':
        """Add a raw data trace with standard styling.
        
        Args:
            x_source: Data source for X-axis values
            y_source: Data source for Y-axis values
            name: Display name for the trace
            style_override: Optional style overrides
            hover_template: Custom hover template
            custom_data_sources: Additional data sources for hover info
            
        Returns:
            Self for chaining
        """
        trace = create_standard_trace(x_source, y_source, name, style_override=style_override)
        
        if hover_template:
            trace.hover_template = hover_template
        if custom_data_sources:
            trace.custom_data_sources = custom_data_sources
            
        self._traces.append(trace)
        return self
    
    def add_fit_trace(
        self,
        x_source: str,
        y_source: str,
        name: str = "Fit",
        style_override: Optional[Dict[str, Any]] = None,
        condition_source: str = "outcome",
        condition_value: Any = "successful"
    ) -> 'PlotConfigurationBuilder':
        """Add a fit trace with standard fit styling.
        
        Args:
            x_source: Data source for X-axis values
            y_source: Data source for Y-axis values (usually fitted data)
            name: Display name for the trace
            style_override: Optional style overrides
            condition_source: Data source to check for visibility condition
            condition_value: Value that enables the trace
            
        Returns:
            Self for chaining
        """
        trace = create_fit_trace(x_source, y_source, name, style_override)
        trace.condition_source = condition_source
        trace.condition_value = condition_value
        
        self._fit_traces.append(trace)
        return self
    
    def add_heatmap_trace(
        self,
        x_source: str,
        y_source: str,
        z_source: str,
        name: str = "Heatmap",
        colorbar_title: str = "|IQ|",
        colorscale: str = Colors.HEATMAP_COLORSCALE,
        zmin_percentile: float = 2.0,
        zmax_percentile: float = 98.0
    ) -> 'PlotConfigurationBuilder':
        """Add a heatmap trace with standard styling.
        
        Args:
            x_source: Data source for X-axis values
            y_source: Data source for Y-axis values
            z_source: Data source for Z-axis values (color intensity)
            name: Display name for the trace
            colorbar_title: Title for the colorbar
            colorscale: Colorscale name (e.g., "Viridis", "RdBu")
            zmin_percentile: Lower percentile for color scaling
            zmax_percentile: Upper percentile for color scaling
            
        Returns:
            Self for chaining
        """
        trace = create_heatmap_trace(x_source, y_source, z_source, name, colorbar_title)
        trace.colorscale = colorscale
        trace.zmin_percentile = zmin_percentile
        trace.zmax_percentile = zmax_percentile
        
        self._traces.append(trace)
        return self
    
    def add_custom_trace(
        self,
        trace_config: TraceConfig
    ) -> 'PlotConfigurationBuilder':
        """Add a custom trace configuration.
        
        Args:
            trace_config: Pre-configured trace object
            
        Returns:
            Self for chaining
        """
        if hasattr(trace_config, 'condition_source') and trace_config.condition_source:
            self._fit_traces.append(trace_config)
        else:
            self._traces.append(trace_config)
        return self
    
    def add_optimal_marker(
        self,
        x_source: str,
        y_source: str,
        condition_source: str = "outcome",
        condition_value: Any = "successful",
        marker_style: Optional[Dict[str, Any]] = None
    ) -> 'PlotConfigurationBuilder':
        """Add an optimal point marker overlay.
        
        Args:
            x_source: Data source for X position
            y_source: Data source for Y position
            condition_source: Data source to check for overlay condition
            condition_value: Value that enables the overlay
            marker_style: Optional style overrides
            
        Returns:
            Self for chaining
        """
        marker = create_optimal_marker(x_source, y_source, condition_source)
        marker.condition_value = condition_value
        
        if marker_style:
            marker.marker_style.update(marker_style)
            
        self._overlays.append(marker)
        return self
    
    def add_vertical_line(
        self,
        position_source: str,
        condition_source: str = "outcome",
        condition_value: Any = "successful",
        line_style: Optional[Dict[str, Any]] = None
    ) -> 'PlotConfigurationBuilder':
        """Add a vertical line overlay.
        
        Args:
            position_source: Data source for line X position
            condition_source: Data source to check for overlay condition
            condition_value: Value that enables the overlay
            line_style: Optional style overrides
            
        Returns:
            Self for chaining
        """
        line = create_vertical_line(position_source, condition_source)
        line.condition_value = condition_value
        
        if line_style:
            line.line_style.update(line_style)
            
        self._overlays.append(line)
        return self
    
    def add_horizontal_line(
        self,
        position_source: str,
        condition_source: str = "outcome",
        condition_value: Any = "successful",
        line_style: Optional[Dict[str, Any]] = None
    ) -> 'PlotConfigurationBuilder':
        """Add a horizontal line overlay.
        
        Args:
            position_source: Data source for line Y position
            condition_source: Data source to check for overlay condition
            condition_value: Value that enables the overlay
            line_style: Optional style overrides
            
        Returns:
            Self for chaining
        """
        line = LineOverlayConfig(
            orientation="horizontal",
            position_source=position_source,
            condition_source=condition_source,
            condition_value=condition_value
        )
        
        if line_style:
            line.line_style.update(line_style)
            
        self._overlays.append(line)
        return self
    
    def add_dual_axis(
        self,
        title: str,
        source: str,
        format_string: str = "{:.3f}",
        overlay_offset: int = 100
    ) -> 'PlotConfigurationBuilder':
        """Add dual X-axis configuration.
        
        Args:
            title: Title for the secondary (top) axis
            source: Data source for secondary axis values
            format_string: Format string for tick labels
            overlay_offset: Plotly axis offset for overlays
            
        Returns:
            Self for chaining
        """
        self._dual_axis = create_dual_axis(title, source, format_string)
        self._dual_axis.overlay_offset = overlay_offset
        return self
    
    def for_experiment_type(self, experiment_type: str) -> 'PlotConfigurationBuilder':
        """Set the experiment type for specialized handling.
        
        Args:
            experiment_type: Type of experiment (e.g., "power_rabi", "flux_spectroscopy")
            
        Returns:
            Self for chaining
        """
        self._experiment_type = experiment_type
        
        # Determine plot family based on experiment type
        heatmap_experiments = [
            "flux_spectroscopy", "amplitude_spectroscopy", 
            "power_rabi_2d", "resonator_spectroscopy_vs_flux",
            "resonator_spectroscopy_vs_amplitude"
        ]
        
        if experiment_type in heatmap_experiments:
            self._plot_family = "heatmap"
        else:
            self._plot_family = "spectroscopy"
            
        return self
    
    def plot_family(self, family: str) -> 'PlotConfigurationBuilder':
        """Explicitly set the plot family.
        
        Args:
            family: Plot family ("spectroscopy", "heatmap", "base")
            
        Returns:
            Self for chaining
        """
        self._plot_family = family
        return self
    
    def with_standard_axes(
        self,
        experiment_type: str
    ) -> 'PlotConfigurationBuilder':
        """Apply standard axis labels for common experiment types.
        
        Args:
            experiment_type: Type of experiment
            
        Returns:
            Self for chaining
        """
        axis_map = {
            "resonator_spectroscopy": (AxisLabels.RF_FREQUENCY_GHZ, AxisLabels.IQ_AMPLITUDE_MV),
            "power_rabi": (AxisLabels.PULSE_AMPLITUDE_MV, AxisLabels.I_MV),
            "flux_spectroscopy": (AxisLabels.FLUX_BIAS_V, AxisLabels.RF_FREQUENCY_GHZ),
            "amplitude_spectroscopy": (AxisLabels.DETUNING_MHZ, AxisLabels.POWER_DBM),
            "ramsey": (AxisLabels.IDLE_TIME_NS, AxisLabels.STATE),
            "t1": (AxisLabels.WAIT_TIME_NS, AxisLabels.STATE),
        }
        
        if experiment_type in axis_map:
            x_label, y_label = axis_map[experiment_type]
            self.x_axis(x_label).y_axis(y_label)
            
        return self
    
    def validate(self) -> bool:
        """Validate the configuration before building.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self._layout.title:
            raise ValueError("Plot title is required")
            
        if not self._layout.x_axis_title:
            raise ValueError("X-axis title is required")
            
        if not self._layout.y_axis_title:
            raise ValueError("Y-axis title is required")
            
        if not self._traces and not self._fit_traces:
            raise ValueError("At least one trace is required")
            
        # Check for heatmap consistency
        has_heatmap = any(isinstance(t, HeatmapTraceConfig) or t.plot_type == "heatmap" 
                         for t in self._traces)
        has_line = any(t.plot_type in ["line", "scatter"] for t in self._traces)
        
        if has_heatmap and has_line:
            raise ValueError("Cannot mix heatmap and line/scatter traces in the same plot")
            
        return True
    
    def build(self) -> Union[SpectroscopyConfig, HeatmapConfig]:
        """Build the final configuration object.
        
        Returns:
            Configured plot object (SpectroscopyConfig or HeatmapConfig)
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        self.validate()
        
        # Determine configuration type based on traces
        is_heatmap = any(
            isinstance(t, HeatmapTraceConfig) or t.plot_type == "heatmap" 
            for t in self._traces
        )
        
        # Build appropriate config type
        if is_heatmap or self._plot_family == "heatmap":
            config = HeatmapConfig(
                layout=self._layout,
                traces=self._traces,
                fit_traces=self._fit_traces,
                overlays=self._overlays,
                plot_family=self._plot_family
            )
        else:
            config = SpectroscopyConfig(
                layout=self._layout,
                traces=self._traces,
                fit_traces=self._fit_traces,
                overlays=self._overlays,
                dual_axis=self._dual_axis,
                plot_family=self._plot_family
            )
            
        return config
    
    def copy(self) -> 'PlotConfigurationBuilder':
        """Create a copy of the current builder state.
        
        Useful for creating variations of a configuration.
        
        Returns:
            New builder instance with copied state
        """
        import copy
        
        new_builder = PlotConfigurationBuilder()
        new_builder._layout = copy.deepcopy(self._layout)
        new_builder._traces = copy.deepcopy(self._traces)
        new_builder._fit_traces = copy.deepcopy(self._fit_traces)
        new_builder._overlays = copy.deepcopy(self._overlays)
        new_builder._dual_axis = copy.deepcopy(self._dual_axis)
        new_builder._experiment_type = self._experiment_type
        new_builder._plot_family = self._plot_family
        
        return new_builder


# Convenience functions for common patterns

def quick_spectroscopy_config(
    x_source: str,
    y_source: str,
    title: str = "Spectroscopy",
    x_label: str = "Frequency [GHz]",
    y_label: str = "Signal [mV]",
    include_fit: bool = True
) -> SpectroscopyConfig:
    """Create a simple spectroscopy configuration quickly.
    
    Args:
        x_source: X-axis data source
        y_source: Y-axis data source
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        include_fit: Whether to include fit trace
        
    Returns:
        Configured SpectroscopyConfig
    """
    builder = (PlotConfigurationBuilder()
        .title(title)
        .x_axis(x_label)
        .y_axis(y_label)
        .add_raw_trace(x_source, y_source))
    
    if include_fit:
        builder.add_fit_trace(x_source, f"fitted_{y_source}")
        
    return builder.build()


def quick_heatmap_config(
    x_source: str,
    y_source: str,
    z_source: str,
    title: str = "Heatmap",
    x_label: str = "X Parameter",
    y_label: str = "Y Parameter",
    colorbar_title: str = "|IQ|"
) -> HeatmapConfig:
    """Create a simple heatmap configuration quickly.
    
    Args:
        x_source: X-axis data source
        y_source: Y-axis data source
        z_source: Z-axis (color) data source
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        colorbar_title: Colorbar title
        
    Returns:
        Configured HeatmapConfig
    """
    return (PlotConfigurationBuilder()
        .title(title)
        .x_axis(x_label)
        .y_axis(y_label)
        .add_heatmap_trace(x_source, y_source, z_source, colorbar_title=colorbar_title)
        .build())