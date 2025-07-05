"""
Unified plotting configuration system.

This module provides a comprehensive configuration system for quantum calibration
plotting that supports simple plots, complex heatmaps, and adaptive plotting
based on data characteristics.
"""

# Base configuration classes
from .base import (
    TraceConfig,
    LayoutConfig, 
    DualAxisConfig,
    OverlayConfig,
    LineOverlayConfig,
    MarkerOverlayConfig,
    PlotConfig,
    ColorbarConfig,
    HeatmapTraceConfig,
    AdaptiveTraceConfig,
    DimensionalityDetector,
    create_standard_trace,
    create_fit_trace,
    create_heatmap_trace,
    create_optimal_marker,
    create_vertical_line,
    create_dual_axis
)

# Visual standards
from .visual_standards import (
    Colors,
    LineStyles,
    FigureDimensions,
    SubplotSpacing,
    Margins,
    ColorbarConfig as VizColorbarConfig,
    AxisLabels,
    NumberFormatting,
    HoverTemplates,
    DualAxisConfig as VizDualAxisConfig,
    Typography,
    LegendConfig,
    UnitConversions,
    get_standard_plotly_style,
    get_standard_matplotlib_size,
    get_raw_data_style,
    get_fit_line_style,
    get_optimal_marker_style,
    get_heatmap_style
)

# Specialized configurations
from .spectroscopy import (
    SpectroscopyConfig,
    create_resonator_spectroscopy_config,
    create_phase_spectroscopy_config,
    create_power_rabi_1d_config,
    create_power_rabi_state_config,
    get_spectroscopy_config,
    SPECTROSCOPY_CONFIGS
)

from .heatmap import (
    HeatmapConfig,
    create_flux_spectroscopy_config,
    create_amplitude_spectroscopy_config,
    create_power_rabi_2d_config,
    create_power_rabi_2d_state_config,
    get_heatmap_config,
    HEATMAP_CONFIGS
)

from .adaptive import (
    AdaptiveConfig,
    create_adaptive_config,
    create_power_rabi_adaptive,
    create_resonator_spectroscopy_adaptive,
    get_adaptive_config,
    describe_adaptive_selection
)

# Fluent configuration builder
from .builder import (
    PlotConfigurationBuilder,
    quick_spectroscopy_config,
    quick_heatmap_config
)

# Configuration templates
from .templates import (
    ConfigurationTemplates,
    get_template,
    customize_template,
    TemplateSets
)

__all__ = [
    # Base classes
    "TraceConfig",
    "LayoutConfig",
    "DualAxisConfig", 
    "OverlayConfig",
    "LineOverlayConfig",
    "MarkerOverlayConfig",
    "PlotConfig",
    "ColorbarConfig",
    "HeatmapTraceConfig",
    "AdaptiveTraceConfig",
    "DimensionalityDetector",
    
    # Factory functions
    "create_standard_trace",
    "create_fit_trace", 
    "create_heatmap_trace",
    "create_optimal_marker",
    "create_vertical_line",
    "create_dual_axis",
    
    # Visual standards
    "Colors",
    "LineStyles",
    "FigureDimensions",
    "SubplotSpacing",
    "Margins",
    "AxisLabels",
    "NumberFormatting",
    "HoverTemplates",
    "Typography",
    "LegendConfig",
    "UnitConversions",
    "get_standard_plotly_style",
    "get_standard_matplotlib_size",
    "get_raw_data_style",
    "get_fit_line_style",
    "get_optimal_marker_style",
    "get_heatmap_style",
    
    # Specialized configurations
    "SpectroscopyConfig",
    "HeatmapConfig",
    "create_resonator_spectroscopy_config",
    "create_phase_spectroscopy_config",
    "create_power_rabi_1d_config",
    "create_power_rabi_state_config",
    "create_flux_spectroscopy_config",
    "create_amplitude_spectroscopy_config",
    "create_power_rabi_2d_config",
    "create_power_rabi_2d_state_config",
    "get_spectroscopy_config",
    "get_heatmap_config",
    "SPECTROSCOPY_CONFIGS",
    "HEATMAP_CONFIGS",
    
    # Adaptive configurations
    "AdaptiveConfig",
    "create_adaptive_config",
    "create_power_rabi_adaptive",
    "create_resonator_spectroscopy_adaptive",
    "get_adaptive_config",
    "describe_adaptive_selection",
    
    # Fluent configuration builder
    "PlotConfigurationBuilder",
    "quick_spectroscopy_config",
    "quick_heatmap_config",
    
    # Configuration templates
    "ConfigurationTemplates",
    "get_template",
    "customize_template",
    "TemplateSets",
]