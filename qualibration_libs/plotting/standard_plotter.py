"""
Enhanced standard plotting interface for the unified quantum calibration framework.

This module provides the main API for creating plots using the enhanced configuration
system and adaptive engines. It serves as the primary entry point for all plotting
operations while maintaining backward compatibility.
"""

from typing import Any, Dict, List, Optional, Union
import logging

import xarray as xr
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# Import configurations for backward compatibility
from .configs import (AdaptiveConfig, DualAxisConfig, HeatmapConfig,
                      LayoutConfig, PlotConfig, SpectroscopyConfig,
                      TraceConfig, create_adaptive_config, get_adaptive_config)
# Import exceptions
from .exceptions import ValidationError, ConfigurationError

# Engine imports moved to function level to avoid circular dependencies

logger = logging.getLogger(__name__)

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
    """Create both Plotly and Matplotlib figures using adaptive configuration.
    
    This is the main entry point for the enhanced plotting system. It automatically
    detects the appropriate plot type based on experiment type and data characteristics.
    
    Args:
        ds_raw: Raw experimental dataset containing measurement data.
        qubits: List of qubit objects to plot.
        experiment_type: Type of experiment (e.g., "power_rabi", "resonator_spectroscopy",
            "flux_spectroscopy", "amplitude_spectroscopy", "ramsey", "t1").
        ds_fit: Optional fitted dataset containing analysis results.
        config_override: Optional configuration to override adaptive defaults.
        **kwargs: Additional configuration parameters passed to the adaptive engine.
        
    Returns:
        Dictionary containing both figure types with keys:
            - "plotly": Interactive Plotly figure
            - "matplotlib": Static Matplotlib figure
            - "interactive": Alias for "plotly"
            - "static": Alias for "matplotlib"
        
    Raises:
        ValidationError: If input data validation fails.
        ConfigurationError: If configuration is invalid or experiment type unknown.
        EngineError: If figure creation fails in either engine.
        
    Examples:
        >>> # Power Rabi (automatically detects 1D vs 2D)
        >>> figures = create_figures(ds_raw, qubits, "power_rabi", ds_fit)
        >>> 
        >>> # Resonator spectroscopy
        >>> figures = create_figures(ds_raw, qubits, "resonator_spectroscopy", ds_fit)
        >>> 
        >>> # Show interactive plot
        >>> figures["plotly"].show()
        >>> 
        >>> # Save static plot
        >>> figures["matplotlib"].savefig("plot.png", dpi=300)
    """
    from .engines.adaptive_engine import create_adaptive_figures
    return create_adaptive_figures(ds_raw, qubits, experiment_type, ds_fit, **kwargs)


def create_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: Union[List[PlotConfig], str],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> PlotlyFigure:
    """Create a Plotly figure using configuration or experiment type.
    
    Supports both adaptive configuration (using experiment type string) and
    explicit configuration objects for maximum flexibility.
    
    Args:
        ds_raw: Raw experimental dataset containing measurement data.
        qubits: List of qubit objects to plot.
        plot_configs: Either a list of PlotConfig objects for explicit configuration,
            or an experiment type string for adaptive configuration.
        ds_fit: Optional fitted dataset containing analysis results.
        **kwargs: Additional configuration parameters. When using experiment type,
            these are passed to the adaptive engine.
        
    Returns:
        Interactive Plotly figure object ready for display or further customization.
        
    Raises:
        ValidationError: If input data validation fails.
        ConfigurationError: If configuration is invalid or experiment type unknown.
        EngineError: If figure creation fails.
        TypeError: If plot_configs is neither string nor list of PlotConfig.
        
    Examples:
        >>> # Using experiment type (recommended for standard plots)
        >>> fig = create_plotly_figure(ds_raw, qubits, "power_rabi", ds_fit)
        >>> fig.show()
        >>> 
        >>> # Using explicit configuration (for custom plots)
        >>> from qualibration_libs.plotting.configs import SpectroscopyConfig
        >>> config = SpectroscopyConfig(
        ...     layout=LayoutConfig(title="Custom Plot"),
        ...     traces=[TraceConfig(x_source="freq", y_source="signal")]
        ... )
        >>> fig = create_plotly_figure(ds_raw, qubits, [config], ds_fit)
    """
    if isinstance(plot_configs, str):
        # Use adaptive engine with experiment type
        from .engines.adaptive_engine import create_adaptive_plotly_figure
        return create_adaptive_plotly_figure(ds_raw, qubits, plot_configs, ds_fit, **kwargs)
    else:
        # Use explicit configuration
        from .engines import PlotlyEngine
        engine = PlotlyEngine()
        return engine.create_figure(ds_raw, qubits, plot_configs, ds_fit)


def create_matplotlib_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: Union[List[PlotConfig], str],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> MatplotlibFigure:
    """Create a Matplotlib figure using configuration or experiment type.
    
    Produces publication-quality static figures suitable for papers and reports.
    Supports both adaptive configuration and explicit configuration objects.
    
    Args:
        ds_raw: Raw experimental dataset containing measurement data.
        qubits: List of qubit objects to plot.
        plot_configs: Either a list of PlotConfig objects for explicit configuration,
            or an experiment type string for adaptive configuration.
        ds_fit: Optional fitted dataset containing analysis results.
        **kwargs: Additional configuration parameters. When using experiment type,
            these are passed to the adaptive engine.
        
    Returns:
        Matplotlib figure object ready for display or saving.
        
    Raises:
        ValidationError: If input data validation fails.
        ConfigurationError: If configuration is invalid or experiment type unknown.
        EngineError: If figure creation fails.
        TypeError: If plot_configs is neither string nor list of PlotConfig.
        
    Examples:
        >>> # Using experiment type (recommended for standard plots)
        >>> fig = create_matplotlib_figure(ds_raw, qubits, "power_rabi", ds_fit)
        >>> fig.savefig("power_rabi.pdf", dpi=300, bbox_inches='tight')
        >>> 
        >>> # Using explicit configuration (for custom plots)
        >>> from qualibration_libs.plotting.configs import HeatmapConfig
        >>> config = HeatmapConfig(
        ...     layout=LayoutConfig(title="2D Scan"),
        ...     heatmap_traces=[HeatmapTraceConfig(x_source="x", y_source="y", z_source="z")]
        ... )
        >>> fig = create_matplotlib_figure(ds_raw, qubits, [config], ds_fit)
        >>> plt.show()
    """
    if isinstance(plot_configs, str):
        # Use adaptive engine with experiment type
        from .engines.adaptive_engine import create_adaptive_matplotlib_figure
        return create_adaptive_matplotlib_figure(ds_raw, qubits, plot_configs, ds_fit, **kwargs)
    else:
        # Use explicit configuration
        from .engines import MatplotlibEngine
        engine = MatplotlibEngine()
        return engine.create_figure(ds_raw, qubits, plot_configs, ds_fit)


# ===== SPECIALIZED PLOTTING FUNCTIONS =====

def create_power_rabi_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """Create Power Rabi figures with automatic 1D/2D detection.
    
    Automatically detects whether the Power Rabi experiment is 1D (amplitude sweep)
    or 2D (amplitude vs pulses chevron) and creates appropriate visualizations.
    
    Args:
        ds_raw: Raw experimental dataset containing Power Rabi measurement data.
        qubits: List of qubit objects to plot.
        ds_fit: Optional fitted dataset containing Rabi oscillation fit results.
        **kwargs: Additional configuration parameters passed to create_figures.
        
    Returns:
        Dictionary containing both figure types. See create_figures for details.
        
    Raises:
        See create_figures for possible exceptions.
        
    Examples:
        >>> figures = create_power_rabi_figures(ds_raw, qubits, ds_fit)
        >>> figures["plotly"].show()  # Interactive visualization
    """
    return create_figures(ds_raw, qubits, "power_rabi", ds_fit, **kwargs)


def create_resonator_spectroscopy_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """Create resonator spectroscopy figures.
    
    Creates visualizations for resonator spectroscopy experiments showing
    resonance frequency identification with optional Lorentzian fits.
    
    Args:
        ds_raw: Raw experimental dataset containing resonator spectroscopy data.
        qubits: List of qubit objects to plot.
        ds_fit: Optional fitted dataset containing resonance fit results.
        **kwargs: Additional configuration parameters passed to create_figures.
        
    Returns:
        Dictionary containing both figure types. See create_figures for details.
        
    Raises:
        See create_figures for possible exceptions.
        
    Examples:
        >>> figures = create_resonator_spectroscopy_figures(ds_raw, qubits, ds_fit)
        >>> figures["matplotlib"].savefig("resonator_spec.pdf")
    """
    return create_figures(ds_raw, qubits, "resonator_spectroscopy", ds_fit, **kwargs)


def create_flux_spectroscopy_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """Create resonator spectroscopy vs flux figures.
    
    Creates 2D heatmap visualizations showing resonator frequency dependence
    on flux bias, useful for identifying flux sweet spots and periodicities.
    
    Args:
        ds_raw: Raw experimental dataset containing flux spectroscopy data.
        qubits: List of qubit objects to plot.
        ds_fit: Optional fitted dataset containing extracted resonance frequencies.
        **kwargs: Additional configuration parameters passed to create_figures.
        
    Returns:
        Dictionary containing both figure types. See create_figures for details.
        
    Raises:
        See create_figures for possible exceptions.
        
    Examples:
        >>> figures = create_flux_spectroscopy_figures(ds_raw, qubits, ds_fit)
        >>> # Flux sweet spots visible as vertical features in the heatmap
    """
    return create_figures(ds_raw, qubits, "resonator_spectroscopy_vs_flux", ds_fit, **kwargs)


def create_amplitude_spectroscopy_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """Create resonator spectroscopy vs amplitude figures.
    
    Creates 2D heatmap visualizations showing resonator response dependence
    on drive amplitude, useful for identifying power-dependent effects.
    
    Args:
        ds_raw: Raw experimental dataset containing amplitude spectroscopy data.
        qubits: List of qubit objects to plot.
        ds_fit: Optional fitted dataset containing extracted parameters.
        **kwargs: Additional configuration parameters passed to create_figures.
        
    Returns:
        Dictionary containing both figure types. See create_figures for details.
        
    Raises:
        See create_figures for possible exceptions.
        
    Examples:
        >>> figures = create_amplitude_spectroscopy_figures(ds_raw, qubits, ds_fit)
        >>> # Look for bifurcation or nonlinear effects at high powers
    """
    return create_figures(ds_raw, qubits, "resonator_spectroscopy_vs_amplitude", ds_fit, **kwargs)


# ===== UTILITY FUNCTIONS =====

def describe_configuration(
    ds_raw: xr.Dataset,
    experiment_type: str,
    **kwargs
) -> Dict[str, Any]:
    """Get detailed information about configuration selection for debugging.
    
    Useful for understanding how the adaptive engine interprets your data
    and selects appropriate plotting configurations.
    
    Args:
        ds_raw: Raw experimental dataset to analyze.
        experiment_type: Type of experiment to describe configuration for.
        **kwargs: Additional configuration parameters to consider.
        
    Returns:
        Dictionary containing:
            - "detected_type": What experiment type was detected
            - "plot_type": Selected plot type (1D, 2D, etc.)
            - "data_shape": Shape information about the data
            - "config_details": Details about selected configuration
            - "available_params": Parameters that can be customized
        
    Examples:
        >>> info = describe_configuration(ds_raw, "power_rabi")
        >>> print(info["detected_type"])  # Shows actual detection result
        >>> print(info["available_params"])  # Shows customization options
    """
    from .engines import AdaptiveEngine
    engine = AdaptiveEngine()
    return engine.describe_configuration(ds_raw, experiment_type, **kwargs)


def validate_plotting_inputs(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None
) -> Dict[str, Any]:
    """Validate inputs for plotting and provide diagnostic information.
    
    Performs comprehensive validation of datasets before plotting to catch
    common issues early and provide helpful error messages.
    
    Args:
        ds_raw: Raw experimental dataset to validate.
        qubits: List of qubit objects to validate against dataset.
        ds_fit: Optional fitted dataset to validate.
        
    Returns:
        Dictionary containing:
            - "raw_data_valid": Boolean indicating if raw data is valid
            - "fit_data_valid": Boolean indicating if fit data is valid
            - "errors": List of error messages
            - "warnings": List of warning messages
            - "suggestions": List of suggestions to fix issues
            - "raw_data_summary": Summary of raw data structure
            - "fit_data_summary": Summary of fit data structure (if provided)
        
    Examples:
        >>> results = validate_plotting_inputs(ds_raw, qubits, ds_fit)
        >>> if not results["raw_data_valid"]:
        ...     print("Errors:", results["errors"])
        ...     print("Suggestions:", results["suggestions"])
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
    except ValidationError as e:
        results["raw_data_valid"] = False
        results["errors"].append(f"Raw data validation failed: {str(e)}")
        if hasattr(e, 'suggestions'):
            results["suggestions"].extend(e.suggestions)
    except Exception as e:
        results["raw_data_valid"] = False
        results["errors"].append(f"Unexpected error during raw data validation: {str(e)}")
        logger.error("Unexpected validation error", exc_info=True)
    
    if ds_fit is not None:
        try:
            validator.validate_dataset(ds_fit, qubits, is_fit_data=True)
        except ValidationError as e:
            results["fit_data_valid"] = False
            results["errors"].append(f"Fit data validation failed: {str(e)}")
            if hasattr(e, 'suggestions'):
                results["suggestions"].extend(e.suggestions)
        except Exception as e:
            results["fit_data_valid"] = False
            results["errors"].append(f"Unexpected error during fit data validation: {str(e)}")
            logger.error("Unexpected validation error", exc_info=True)
    
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