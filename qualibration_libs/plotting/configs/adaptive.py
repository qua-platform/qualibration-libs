"""
Adaptive configurations for experiments that can produce different plot types.

This module provides configurations that automatically adapt based on data
characteristics, such as Power Rabi experiments that can be 1D or 2D.
"""

from typing import Union, Any
import xarray as xr
from .base import PlotConfig, DimensionalityDetector
from .spectroscopy import SpectroscopyConfig, get_spectroscopy_config
from .heatmap import HeatmapConfig, get_heatmap_config


class AdaptiveConfig:
    """
    Adaptive configuration that selects appropriate plot type based on data.
    
    This class analyzes the input data and automatically chooses between
    different configuration types (e.g., 1D vs 2D for Power Rabi).
    """
    
    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type
        self._config_cache = {}
    
    def get_config(self, ds_raw: xr.Dataset, **kwargs) -> Union[SpectroscopyConfig, HeatmapConfig]:
        """
        Get the appropriate configuration based on data characteristics.
        
        Args:
            ds_raw: Raw dataset to analyze
            **kwargs: Additional parameters for configuration selection
            
        Returns:
            Appropriate configuration object
        """
        config_key = self._get_config_key(ds_raw, **kwargs)
        
        if config_key not in self._config_cache:
            self._config_cache[config_key] = self._create_config(config_key, **kwargs)
        
        return self._config_cache[config_key]
    
    def _get_config_key(self, ds_raw: xr.Dataset, **kwargs) -> str:
        """Determine the configuration key based on data analysis."""
        
        if self.experiment_type == "power_rabi":
            return self._get_power_rabi_config_key(ds_raw, **kwargs)
        
        # For other experiments, use static mapping
        experiment_config_map = {
            "resonator_spectroscopy": "resonator_spectroscopy_amplitude",
            "resonator_spectroscopy_vs_amplitude": "amplitude_spectroscopy",
            "resonator_spectroscopy_vs_flux": "flux_spectroscopy",
        }
        
        return experiment_config_map.get(self.experiment_type, "base")
    
    def _get_power_rabi_config_key(self, ds_raw: xr.Dataset, **kwargs) -> str:
        """Determine Power Rabi configuration based on dimensionality and data type."""
        
        # Detect 1D vs 2D
        dimensionality = DimensionalityDetector.detect_power_rabi_dimensionality(ds_raw)
        
        # Detect state vs IQ measurement
        measurement_type = self._detect_power_rabi_measurement_type(ds_raw)
        
        # Build configuration key
        if dimensionality == "1D":
            if measurement_type == "state":
                return "power_rabi_1d_state"
            else:
                return "power_rabi_1d"
        else:  # 2D
            if measurement_type == "state":
                return "power_rabi_2d_state"
            else:
                return "power_rabi_2d"
    
    def _detect_power_rabi_measurement_type(self, ds_raw: xr.Dataset) -> str:
        """Detect if Power Rabi uses state discrimination or IQ measurement."""
        
        # Check for state variables
        qubit_dims = [dim for dim in ds_raw.dims if dim.startswith('qubit')]
        if not qubit_dims:
            return "iq"  # Default fallback
        
        # Sample a qubit to check data variables
        sample_qubit = ds_raw[qubit_dims[0]].values[0] if len(ds_raw[qubit_dims[0]]) > 0 else None
        if sample_qubit is None:
            return "iq"
        
        # Check for state variables (state1, state2, etc.)
        state_vars = [var for var in ds_raw.data_vars if var.startswith('state')]
        if state_vars:
            return "state"
        
        # Check for IQ variables (I1, Q1, I2, Q2, etc.)
        i_vars = [var for var in ds_raw.data_vars if var.startswith('I')]
        q_vars = [var for var in ds_raw.data_vars if var.startswith('Q')]
        
        if i_vars and q_vars:
            return "iq"
        
        # Default to IQ if unclear
        return "iq"
    
    def _create_config(self, config_key: str, **kwargs) -> Union[SpectroscopyConfig, HeatmapConfig]:
        """Create the appropriate configuration object."""
        
        # 1D configurations (SpectroscopyConfig)
        spectroscopy_configs = {
            "resonator_spectroscopy_amplitude",
            "power_rabi_1d",
            "power_rabi_1d_state"
        }
        
        # 2D configurations (HeatmapConfig)
        heatmap_configs = {
            "amplitude_spectroscopy",
            "flux_spectroscopy", 
            "power_rabi_2d",
            "power_rabi_2d_state"
        }
        
        if config_key in spectroscopy_configs:
            return get_spectroscopy_config(config_key)
        elif config_key in heatmap_configs:
            return get_heatmap_config(config_key)
        else:
            raise ValueError(f"Unknown configuration key: {config_key}")
    
    def describe_selection(self, ds_raw: xr.Dataset, **kwargs) -> str:
        """Get a human-readable description of the configuration selection."""
        
        config_key = self._get_config_key(ds_raw, **kwargs)
        config = self.get_config(ds_raw, **kwargs)
        
        description_map = {
            "power_rabi_1d": "1D Power Rabi (single pulse, IQ measurement)",
            "power_rabi_1d_state": "1D Power Rabi (single pulse, state discrimination)",
            "power_rabi_2d": "2D Power Rabi chevron (multi-pulse, IQ measurement)",
            "power_rabi_2d_state": "2D Power Rabi chevron (multi-pulse, state discrimination)",
            "resonator_spectroscopy_amplitude": "1D Resonator spectroscopy (amplitude)",
            "amplitude_spectroscopy": "2D Resonator spectroscopy vs power",
            "flux_spectroscopy": "2D Resonator spectroscopy vs flux",
        }
        
        base_description = description_map.get(config_key, f"Configuration: {config_key}")
        
        if self.experiment_type == "power_rabi":
            # Add dimensionality details for Power Rabi
            dimensionality = DimensionalityDetector.detect_power_rabi_dimensionality(ds_raw)
            measurement_type = self._detect_power_rabi_measurement_type(ds_raw)
            
            pulse_info = ""
            if "nb_of_pulses" in ds_raw.dims:
                n_pulses = ds_raw.sizes.get("nb_of_pulses", 1)
                pulse_info = f" ({n_pulses} pulse levels)"
            
            return f"{base_description}{pulse_info}"
        
        return base_description


# ===== FACTORY FUNCTIONS =====

def create_adaptive_config(experiment_type: str) -> AdaptiveConfig:
    """Create an adaptive configuration for the given experiment type."""
    return AdaptiveConfig(experiment_type)


def create_power_rabi_adaptive() -> AdaptiveConfig:
    """Create adaptive configuration specifically for Power Rabi experiments."""
    return AdaptiveConfig("power_rabi")


def create_resonator_spectroscopy_adaptive() -> AdaptiveConfig:
    """Create adaptive configuration for resonator spectroscopy experiments."""
    return AdaptiveConfig("resonator_spectroscopy")


# ===== CONVENIENCE FUNCTIONS =====

def get_adaptive_config(
    experiment_type: str,
    ds_raw: xr.Dataset,
    **kwargs
) -> Union[SpectroscopyConfig, HeatmapConfig]:
    """
    Get the appropriate configuration for an experiment and dataset.
    
    Args:
        experiment_type: Type of experiment (e.g., "power_rabi")
        ds_raw: Raw dataset to analyze
        **kwargs: Additional configuration parameters
        
    Returns:
        Appropriate configuration object
    """
    adaptive = create_adaptive_config(experiment_type)
    return adaptive.get_config(ds_raw, **kwargs)


def describe_adaptive_selection(
    experiment_type: str,
    ds_raw: xr.Dataset,
    **kwargs
) -> str:
    """
    Get a description of the configuration selection for debugging.
    
    Args:
        experiment_type: Type of experiment
        ds_raw: Raw dataset to analyze
        **kwargs: Additional configuration parameters
        
    Returns:
        Human-readable description of the selection
    """
    adaptive = create_adaptive_config(experiment_type)
    return adaptive.describe_selection(ds_raw, **kwargs)