"""
Enhanced standard plotting interface for the unified quantum calibration framework.

This module provides the main API for creating plots using the enhanced configuration
system and adaptive engines. It serves as the primary entry point for all plotting
operations while maintaining backward compatibility.
"""

from typing import Any, Dict, List, Optional, Union

import xarray as xr
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# Import configurations for backward compatibility
from .configs import (AdaptiveConfig, DualAxisConfig, HeatmapConfig,
                      LayoutConfig, PlotConfig, SpectroscopyConfig,
                      TraceConfig, create_adaptive_config, get_adaptive_config)
# Import the enhanced engines
from .engines import AdaptiveEngine, MatplotlibEngine, PlotlyEngine
from .engines.adaptive_engine import (create_adaptive_figures,
                                      create_adaptive_matplotlib_figure,
                                      create_adaptive_plotly_figure)

# Legacy imports for backward compatibility
try:
    # Keep old class definitions for backward compatibility
    from .configs.base import LayoutConfig as LegacyLayoutConfig
    from .configs.base import PlotConfig as LegacyPlotConfig
    from .configs.base import TraceConfig as LegacyTraceConfig
except ImportError:
    # Fallback to new definitions
    LegacyTraceConfig = TraceConfig
    LegacyLayoutConfig = LayoutConfig
    LegacyPlotConfig = PlotConfig


# ===== ENHANCED API =====

def create_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    experiment_type: str,
    ds_fit: Optional[xr.Dataset] = None,
    config_override: Optional[Union[PlotConfig, AdaptiveConfig]] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """
    Create both Plotly and Matplotlib figures using adaptive configuration.
    
    This is the main entry point for the enhanced plotting system. It automatically
    detects the appropriate plot type based on experiment type and data characteristics.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        experiment_type: Type of experiment ("power_rabi", "resonator_spectroscopy", etc.)
        ds_fit: Optional fitted dataset
        config_override: Optional configuration override
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing both figure types:
        {
            "plotly": PlotlyFigure,
            "matplotlib": MatplotlibFigure,
            "interactive": PlotlyFigure,  # Alias
            "static": MatplotlibFigure,   # Alias
        }
        
    Examples:
        # Power Rabi (automatically detects 1D vs 2D)
        figures = create_figures(ds_raw, qubits, "power_rabi", ds_fit)
        
        # Resonator spectroscopy
        figures = create_figures(ds_raw, qubits, "resonator_spectroscopy", ds_fit)
        
        # Show interactive plot
        figures["plotly"].show()
    """
    return create_adaptive_figures(ds_raw, qubits, experiment_type, ds_fit, **kwargs)


def create_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: Union[List[PlotConfig], str],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> PlotlyFigure:
    """
    Create a Plotly figure using configuration or experiment type.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        plot_configs: Either list of PlotConfig objects or experiment type string
        ds_fit: Optional fitted dataset
        **kwargs: Additional configuration parameters
        
    Returns:
        Plotly figure object
        
    Examples:
        # Using experiment type (recommended)
        fig = create_plotly_figure(ds_raw, qubits, "power_rabi", ds_fit)
        
        # Using explicit configuration
        config = SpectroscopyConfig(...)
        fig = create_plotly_figure(ds_raw, qubits, [config], ds_fit)
    """
    if isinstance(plot_configs, str):
        # Use adaptive engine with experiment type
        return create_adaptive_plotly_figure(ds_raw, qubits, plot_configs, ds_fit, **kwargs)
    else:
        # Use explicit configuration
        engine = PlotlyEngine()
        return engine.create_figure(ds_raw, qubits, plot_configs, ds_fit)


def create_matplotlib_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: Union[List[PlotConfig], str],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> MatplotlibFigure:
    """
    Create a Matplotlib figure using configuration or experiment type.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        plot_configs: Either list of PlotConfig objects or experiment type string
        ds_fit: Optional fitted dataset
        **kwargs: Additional configuration parameters
        
    Returns:
        Matplotlib figure object
        
    Examples:
        # Using experiment type (recommended)
        fig = create_matplotlib_figure(ds_raw, qubits, "power_rabi", ds_fit)
        
        # Using explicit configuration
        config = HeatmapConfig(...)
        fig = create_matplotlib_figure(ds_raw, qubits, [config], ds_fit)
    """
    if isinstance(plot_configs, str):
        # Use adaptive engine with experiment type
        return create_adaptive_matplotlib_figure(ds_raw, qubits, plot_configs, ds_fit, **kwargs)
    else:
        # Use explicit configuration
        engine = MatplotlibEngine()
        return engine.create_figure(ds_raw, qubits, plot_configs, ds_fit)


# ===== SPECIALIZED PLOTTING FUNCTIONS =====

def create_power_rabi_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """
    Create Power Rabi figures with automatic 1D/2D detection.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        ds_fit: Optional fitted dataset
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing both figure types
    """
    return create_figures(ds_raw, qubits, "power_rabi", ds_fit, **kwargs)


def create_resonator_spectroscopy_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """
    Create resonator spectroscopy figures.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        ds_fit: Optional fitted dataset
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing both figure types
    """
    return create_figures(ds_raw, qubits, "resonator_spectroscopy", ds_fit, **kwargs)


def create_flux_spectroscopy_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """
    Create resonator spectroscopy vs flux figures.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        ds_fit: Optional fitted dataset
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing both figure types
    """
    return create_figures(ds_raw, qubits, "resonator_spectroscopy_vs_flux", ds_fit, **kwargs)


def create_amplitude_spectroscopy_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """
    Create resonator spectroscopy vs amplitude figures.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        ds_fit: Optional fitted dataset
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing both figure types
    """
    return create_figures(ds_raw, qubits, "resonator_spectroscopy_vs_amplitude", ds_fit, **kwargs)


# ===== UTILITY FUNCTIONS =====

def describe_configuration(
    ds_raw: xr.Dataset,
    experiment_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Get detailed information about configuration selection for debugging.
    
    Args:
        ds_raw: Raw experimental dataset
        experiment_type: Type of experiment
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with configuration details
    """
    engine = AdaptiveEngine()
    return engine.describe_configuration(ds_raw, experiment_type, **kwargs)


def validate_plotting_inputs(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None
) -> Dict[str, Any]:
    """
    Validate inputs for plotting and provide diagnostic information.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        ds_fit: Optional fitted dataset
        
    Returns:
        Dictionary with validation results and suggestions
    """
    from .engines.data_validators import DataValidator
    
    validator = DataValidator()
    results = {
        "raw_data_valid": True,
        "fit_data_valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    try:
        validator.validate_dataset(ds_raw, qubits, is_fit_data=False)
    except Exception as e:
        results["raw_data_valid"] = False
        results["errors"].append(f"Raw data validation failed: {str(e)}")
    
    if ds_fit is not None:
        try:
            validator.validate_dataset(ds_fit, qubits, is_fit_data=True)
        except Exception as e:
            results["fit_data_valid"] = False
            results["errors"].append(f"Fit data validation failed: {str(e)}")
    
    # Add data summaries
    results["raw_data_summary"] = validator.get_dataset_summary(ds_raw)
    if ds_fit is not None:
        results["fit_data_summary"] = validator.get_dataset_summary(ds_fit)
    
    return results


# ===== BACKWARD COMPATIBILITY =====

# Keep the old function signatures for backward compatibility
def create_specialized_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_config: Any,
    ds_fit: Optional[xr.Dataset] = None,
    ds_prepared: Optional[xr.Dataset] = None
) -> PlotlyFigure:
    """
    Legacy function for backward compatibility.
    
    This function is deprecated. Use create_plotly_figure() instead.
    """
    import warnings
    warnings.warn(
        "create_specialized_plotly_figure is deprecated. Use create_plotly_figure() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Try to map old config to new system
    if hasattr(plot_config, 'plot_family'):
        experiment_type = plot_config.plot_family
    else:
        experiment_type = "generic"
    
    return create_plotly_figure(ds_raw, qubits, experiment_type, ds_fit)


# Export the main functions
__all__ = [
    # Main API
    "create_figures",
    "create_plotly_figure", 
    "create_matplotlib_figure",
    
    # Specialized functions
    "create_power_rabi_figures",
    "create_resonator_spectroscopy_figures",
    "create_flux_spectroscopy_figures",
    "create_amplitude_spectroscopy_figures",
    
    # Utility functions
    "describe_configuration",
    "validate_plotting_inputs",
    
    # Configuration classes (for explicit configuration)
    "PlotConfig",
    "SpectroscopyConfig", 
    "HeatmapConfig",
    "AdaptiveConfig",
    "TraceConfig",
    "LayoutConfig",
    "DualAxisConfig",
    
    # Factory functions
    "get_adaptive_config",
    "create_adaptive_config",
    
    # Legacy compatibility
    "create_specialized_plotly_figure",
]