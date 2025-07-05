"""
Adaptive plotting engine that automatically selects appropriate plot types.

This engine analyzes input data and configuration to determine the best
rendering approach, particularly for experiments like Power Rabi that can
be either 1D or 2D depending on the data characteristics.
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import xarray as xr
from plotly.graph_objects import Figure as PlotlyFigure
from matplotlib.figure import Figure as MatplotlibFigure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from ..configs import (
    PlotConfig, SpectroscopyConfig, HeatmapConfig, AdaptiveConfig,
    DimensionalityDetector, get_adaptive_config, describe_adaptive_selection
)
from .plotly_engine import PlotlyEngine
from .matplotlib_engine import MatplotlibEngine
from .data_validators import DataValidator


class AdaptiveEngine:
    """
    Adaptive plotting engine that automatically selects plot types.
    
    This engine provides the main interface for adaptive plotting, analyzing
    data characteristics and routing to appropriate specialized engines.
    """
    
    def __init__(self):
        self.plotly_engine = PlotlyEngine()
        self.matplotlib_engine = MatplotlibEngine()
        self.data_validator = DataValidator()
    
    def create_figures(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        experiment_type: str,
        ds_fit: Optional[xr.Dataset] = None,
        config_override: Optional[Union[PlotConfig, AdaptiveConfig]] = None,
        **kwargs
    ) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
        """
        Create both Plotly and Matplotlib figures using adaptive configuration.
        
        Args:
            ds_raw: Raw experimental dataset
            qubits: List of qubits to plot
            experiment_type: Type of experiment (e.g., "power_rabi")
            ds_fit: Optional fitted dataset
            config_override: Optional configuration override
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary containing both figure types
        """
        # Validate input data
        self.data_validator.validate_dataset(ds_raw, qubits)
        if ds_fit is not None:
            self.data_validator.validate_dataset(ds_fit, qubits, is_fit_data=True)
        
        # Get appropriate configuration
        if config_override is not None:
            if isinstance(config_override, AdaptiveConfig):
                config = config_override.get_config(ds_raw, **kwargs)
            else:
                config = config_override
        else:
            config = get_adaptive_config(experiment_type, ds_raw, **kwargs)
        
        # Log configuration selection for debugging
        selection_info = describe_adaptive_selection(experiment_type, ds_raw, **kwargs)
        print(f"Adaptive plotting selected: {selection_info}")
        
        # Route to appropriate engines
        plotly_fig = self._create_plotly_figure(ds_raw, qubits, config, ds_fit)
        matplotlib_fig = self._create_matplotlib_figure(ds_raw, qubits, config, ds_fit)
        
        return {
            "plotly": plotly_fig,
            "matplotlib": matplotlib_fig,
            "interactive": plotly_fig,  # Alias for interactive
            "static": matplotlib_fig,   # Alias for static
        }
    
    def create_plotly_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        experiment_type: str,
        ds_fit: Optional[xr.Dataset] = None,
        config_override: Optional[Union[PlotConfig, AdaptiveConfig]] = None,
        **kwargs
    ) -> PlotlyFigure:
        """Create Plotly figure using adaptive configuration."""
        
        figures = self.create_figures(
            ds_raw, qubits, experiment_type, ds_fit, config_override, **kwargs
        )
        return figures["plotly"]
    
    def create_matplotlib_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        experiment_type: str,
        ds_fit: Optional[xr.Dataset] = None,
        config_override: Optional[Union[PlotConfig, AdaptiveConfig]] = None,
        **kwargs
    ) -> MatplotlibFigure:
        """Create Matplotlib figure using adaptive configuration."""
        
        figures = self.create_figures(
            ds_raw, qubits, experiment_type, ds_fit, config_override, **kwargs
        )
        return figures["matplotlib"]
    
    def _create_plotly_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: Union[SpectroscopyConfig, HeatmapConfig],
        ds_fit: Optional[xr.Dataset] = None
    ) -> PlotlyFigure:
        """Create Plotly figure using appropriate engine."""
        
        if isinstance(config, SpectroscopyConfig):
            return self.plotly_engine.create_spectroscopy_figure(ds_raw, qubits, config, ds_fit)
        elif isinstance(config, HeatmapConfig):
            return self.plotly_engine.create_heatmap_figure(ds_raw, qubits, config, ds_fit)
        else:
            # Fallback to generic figure creation
            return self.plotly_engine.create_figure(ds_raw, qubits, [config], ds_fit)
    
    def _create_matplotlib_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: Union[SpectroscopyConfig, HeatmapConfig],
        ds_fit: Optional[xr.Dataset] = None
    ) -> MatplotlibFigure:
        """Create Matplotlib figure using appropriate engine."""
        
        if isinstance(config, SpectroscopyConfig):
            return self.matplotlib_engine.create_spectroscopy_figure(ds_raw, qubits, config, ds_fit)
        elif isinstance(config, HeatmapConfig):
            return self.matplotlib_engine.create_heatmap_figure(ds_raw, qubits, config, ds_fit)
        else:
            # Fallback to generic figure creation
            return self.matplotlib_engine.create_figure(ds_raw, qubits, [config], ds_fit)
    
    def describe_configuration(
        self,
        ds_raw: xr.Dataset,
        experiment_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get detailed information about configuration selection.
        
        Args:
            ds_raw: Raw experimental dataset
            experiment_type: Type of experiment
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary with configuration details
        """
        config = get_adaptive_config(experiment_type, ds_raw, **kwargs)
        selection_info = describe_adaptive_selection(experiment_type, ds_raw, **kwargs)
        
        # Analyze data characteristics
        data_info = self._analyze_data_characteristics(ds_raw, experiment_type)
        
        return {
            "experiment_type": experiment_type,
            "selection_description": selection_info,
            "config_type": type(config).__name__,
            "plot_family": getattr(config, 'plot_family', 'unknown'),
            "data_characteristics": data_info,
            "dimensionality": self._detect_dimensionality(ds_raw, experiment_type),
            "measurement_type": self._detect_measurement_type(ds_raw, experiment_type),
        }
    
    def _analyze_data_characteristics(self, ds_raw: xr.Dataset, experiment_type: str) -> Dict[str, Any]:
        """Analyze data characteristics for debugging and optimization."""
        
        characteristics = {
            "dimensions": dict(ds_raw.dims),
            "data_variables": list(ds_raw.data_vars.keys()),
            "coordinates": list(ds_raw.coords.keys()),
            "total_data_points": ds_raw.sizes.get('qubit', 1) * np.prod([
                size for dim, size in ds_raw.dims.items() if dim != 'qubit'
            ]),
        }
        
        # Experiment-specific analysis
        if experiment_type == "power_rabi":
            characteristics.update({
                "has_pulse_dimension": "nb_of_pulses" in ds_raw.dims,
                "pulse_levels": ds_raw.sizes.get("nb_of_pulses", 1),
                "amplitude_levels": ds_raw.sizes.get("amp_prefactor", 0),
            })
        
        return characteristics
    
    def _detect_dimensionality(self, ds_raw: xr.Dataset, experiment_type: str) -> str:
        """Detect plot dimensionality for debugging."""
        
        if experiment_type == "power_rabi":
            return DimensionalityDetector.detect_power_rabi_dimensionality(ds_raw)
        
        # For other experiments, analyze number of sweep dimensions
        sweep_dims = [dim for dim in ds_raw.dims if dim != 'qubit']
        
        if len(sweep_dims) <= 1:
            return "1D"
        else:
            return "2D"
    
    def _detect_measurement_type(self, ds_raw: xr.Dataset, experiment_type: str) -> str:
        """Detect measurement type for debugging."""
        
        if experiment_type == "power_rabi":
            # Check for state vs IQ measurement
            state_vars = [var for var in ds_raw.data_vars if var.startswith('state')]
            i_vars = [var for var in ds_raw.data_vars if var.startswith('I')]
            q_vars = [var for var in ds_raw.data_vars if var.startswith('Q')]
            
            if state_vars:
                return "state_discrimination"
            elif i_vars and q_vars:
                return "iq_measurement"
            else:
                return "unknown"
        
        # For resonator spectroscopy experiments
        if "resonator" in experiment_type:
            return "iq_measurement"
        
        return "unknown"


# ===== CONVENIENCE FUNCTIONS =====

def create_adaptive_figures(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    experiment_type: str,
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> Dict[str, Union[PlotlyFigure, MatplotlibFigure]]:
    """
    Convenience function to create adaptive figures.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        experiment_type: Type of experiment
        ds_fit: Optional fitted dataset
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing both figure types
    """
    engine = AdaptiveEngine()
    return engine.create_figures(ds_raw, qubits, experiment_type, ds_fit, **kwargs)


def create_adaptive_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    experiment_type: str,
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> PlotlyFigure:
    """Convenience function to create adaptive Plotly figure."""
    engine = AdaptiveEngine()
    return engine.create_plotly_figure(ds_raw, qubits, experiment_type, ds_fit, **kwargs)


def create_adaptive_matplotlib_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    experiment_type: str,
    ds_fit: Optional[xr.Dataset] = None,
    **kwargs
) -> MatplotlibFigure:
    """Convenience function to create adaptive Matplotlib figure."""
    engine = AdaptiveEngine()
    return engine.create_matplotlib_figure(ds_raw, qubits, experiment_type, ds_fit, **kwargs)