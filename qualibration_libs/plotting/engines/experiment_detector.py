"""Centralized experiment type detection for plotting engines.

This module provides a unified way to detect experiment types from dataset structure,
eliminating duplicate detection logic across different plotting engines.
"""

from typing import Dict, List, Optional, Callable, Any
import xarray as xr
from abc import ABC, abstractmethod


class ExperimentDetector:
    """Main detector that uses a chain of detectors to identify experiment types."""
    
    def __init__(self):
        """Initialize with all available detectors."""
        self._detectors: List[BaseExperimentDetector] = [
            PowerRabiDetector(),
            FluxSpectroscopyDetector(),
            AmplitudeSpectroscopyDetector(),
            ResonatorSpectroscopyDetector(),
            TwoToneSpectroscopyDetector(),
            RamseyDetector(),
            T1Detector(),
        ]
    
    def detect_experiment_type(self, ds_raw: xr.Dataset) -> str:
        """Detect the experiment type from dataset structure.
        
        Args:
            ds_raw: Raw measurement dataset
            
        Returns:
            String identifier for experiment type (e.g., 'power_rabi', 'flux_spectroscopy')
            Returns 'unknown' if no detector matches
        """
        for detector in self._detectors:
            if detector.matches(ds_raw):
                return detector.experiment_type
        
        return "unknown"
    
    def get_experiment_properties(self, ds_raw: xr.Dataset) -> Dict[str, Any]:
        """Get detailed properties about the detected experiment.
        
        Args:
            ds_raw: Raw measurement dataset
            
        Returns:
            Dictionary with experiment properties including:
            - type: Experiment type string
            - dimensions: Number of sweep dimensions (1D, 2D, etc.)
            - sweep_params: List of sweep parameter names
            - has_fit_data: Whether fit data is expected
        """
        exp_type = self.detect_experiment_type(ds_raw)
        
        for detector in self._detectors:
            if detector.experiment_type == exp_type:
                return detector.get_properties(ds_raw)
        
        return {
            "type": "unknown",
            "dimensions": self._count_dimensions(ds_raw),
            "sweep_params": [],
            "has_fit_data": False
        }
    
    def _count_dimensions(self, ds_raw: xr.Dataset) -> int:
        """Count the number of sweep dimensions in the dataset."""
        # Exclude 'qubit' dimension as it's not a sweep parameter
        sweep_dims = [dim for dim in ds_raw.dims if dim != 'qubit']
        return len(sweep_dims)
    
    def register_detector(self, detector: 'BaseExperimentDetector'):
        """Register a new experiment detector.
        
        This allows extending the detection system with custom experiment types.
        
        Args:
            detector: New detector instance to add
        """
        self._detectors.append(detector)


class BaseExperimentDetector(ABC):
    """Abstract base class for experiment type detectors."""
    
    @property
    @abstractmethod
    def experiment_type(self) -> str:
        """Return the experiment type identifier."""
        pass
    
    @abstractmethod
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """Check if this detector matches the dataset.
        
        Args:
            ds_raw: Raw measurement dataset
            
        Returns:
            True if this detector matches the dataset structure
        """
        pass
    
    def get_properties(self, ds_raw: xr.Dataset) -> Dict[str, Any]:
        """Get detailed properties about the experiment.
        
        Args:
            ds_raw: Raw measurement dataset
            
        Returns:
            Dictionary with experiment properties
        """
        return {
            "type": self.experiment_type,
            "dimensions": self._count_dimensions(ds_raw),
            "sweep_params": self._get_sweep_params(ds_raw),
            "has_fit_data": True  # Most experiments have fit data
        }
    
    def _count_dimensions(self, ds_raw: xr.Dataset) -> int:
        """Count the number of sweep dimensions."""
        sweep_dims = [dim for dim in ds_raw.dims if dim != 'qubit']
        return len(sweep_dims)
    
    def _get_sweep_params(self, ds_raw: xr.Dataset) -> List[str]:
        """Get list of sweep parameter names."""
        return [dim for dim in ds_raw.dims if dim != 'qubit']


class PowerRabiDetector(BaseExperimentDetector):
    """Detector for Power Rabi experiments."""
    
    @property
    def experiment_type(self) -> str:
        return "power_rabi"
    
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """Power Rabi experiments have amplitude and pulse-related coordinates."""
        power_rabi_indicators = ["amp_prefactor", "full_amp", "nb_of_pulses"]
        
        return all(
            coord in ds_raw.coords or coord in ds_raw.dims 
            for coord in power_rabi_indicators
        )
    
    def get_properties(self, ds_raw: xr.Dataset) -> Dict[str, Any]:
        """Get Power Rabi specific properties."""
        props = super().get_properties(ds_raw)
        
        # Check if it's 1D or 2D Power Rabi
        if "nb_of_pulses" in ds_raw.dims and len(ds_raw.dims["nb_of_pulses"]) > 1:
            props["subtype"] = "2D_power_rabi"
            props["dimensions"] = 2
        else:
            props["subtype"] = "1D_power_rabi"
            props["dimensions"] = 1
        
        return props


class FluxSpectroscopyDetector(BaseExperimentDetector):
    """Detector for Flux Spectroscopy experiments."""
    
    @property
    def experiment_type(self) -> str:
        return "flux_spectroscopy"
    
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """Flux spectroscopy has flux coordinates but no power coordinates."""
        flux_indicators = ["flux_bias", "attenuated_current", "current"]
        power_indicators = ["power", "power_dbm"]
        
        has_flux = any(coord in ds_raw.coords for coord in flux_indicators)
        has_power = any(coord in ds_raw.coords for coord in power_indicators)
        
        # Must have flux and NOT have power
        return has_flux and not has_power
    
    def get_properties(self, ds_raw: xr.Dataset) -> Dict[str, Any]:
        """Get Flux Spectroscopy specific properties."""
        props = super().get_properties(ds_raw)
        
        # Determine if it's resonator or qubit spectroscopy
        if "detuning" in ds_raw.coords or "detuning" in ds_raw.dims:
            props["subtype"] = "resonator_flux_spectroscopy"
        else:
            props["subtype"] = "qubit_flux_spectroscopy"
        
        return props


class AmplitudeSpectroscopyDetector(BaseExperimentDetector):
    """Detector for Amplitude/Power Spectroscopy experiments."""
    
    @property
    def experiment_type(self) -> str:
        return "amplitude_spectroscopy"
    
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """Amplitude spectroscopy has power coordinates and frequency coordinates."""
        power_indicators = ["power", "power_dbm", "amplitude"]
        freq_indicators = ["detuning", "frequency", "freq"]
        
        has_power = any(coord in ds_raw.coords for coord in power_indicators)
        has_freq = any(coord in ds_raw.coords for coord in freq_indicators)
        
        # Must have both power and frequency
        return has_power and has_freq


class ResonatorSpectroscopyDetector(BaseExperimentDetector):
    """Detector for basic Resonator Spectroscopy experiments."""
    
    @property
    def experiment_type(self) -> str:
        return "resonator_spectroscopy"
    
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """Basic resonator spectroscopy has detuning but no other sweep parameters."""
        has_detuning = "detuning" in ds_raw.coords or "detuning" in ds_raw.dims
        
        # Check it's not a 2D spectroscopy
        other_sweep_params = ["flux_bias", "power", "amplitude", "drive_frequency"]
        has_other_sweeps = any(param in ds_raw.coords for param in other_sweep_params)
        
        return has_detuning and not has_other_sweeps
    
    def get_properties(self, ds_raw: xr.Dataset) -> Dict[str, Any]:
        """Get Resonator Spectroscopy specific properties."""
        props = super().get_properties(ds_raw)
        props["dimensions"] = 1  # Always 1D
        return props


class TwoToneSpectroscopyDetector(BaseExperimentDetector):
    """Detector for Two-Tone Spectroscopy experiments."""
    
    @property
    def experiment_type(self) -> str:
        return "two_tone_spectroscopy"
    
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """Two-tone has both readout and drive frequency parameters."""
        readout_indicators = ["readout_freq", "ro_freq", "readout_detuning"]
        drive_indicators = ["drive_freq", "qubit_freq", "drive_detuning"]
        
        has_readout = any(coord in ds_raw.coords for coord in readout_indicators)
        has_drive = any(coord in ds_raw.coords for coord in drive_indicators)
        
        return has_readout and has_drive


class RamseyDetector(BaseExperimentDetector):
    """Detector for Ramsey experiments."""
    
    @property
    def experiment_type(self) -> str:
        return "ramsey"
    
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """Ramsey experiments have idle time parameter."""
        ramsey_indicators = ["idle_time", "tau", "delay", "wait_time"]
        
        # Check for Ramsey-specific parameters
        has_time_param = any(coord in ds_raw.coords for coord in ramsey_indicators)
        
        # Also check for phase-related data that's common in Ramsey
        has_phase = "phase" in ds_raw.data_vars
        
        return has_time_param and has_phase


class T1Detector(BaseExperimentDetector):
    """Detector for T1 (relaxation) experiments."""
    
    @property  
    def experiment_type(self) -> str:
        return "t1"
    
    def matches(self, ds_raw: xr.Dataset) -> bool:
        """T1 experiments have wait/delay time but no phase data."""
        time_indicators = ["wait_time", "delay", "tau", "time"]
        
        has_time = any(coord in ds_raw.coords for coord in time_indicators)
        
        # T1 doesn't have phase data (unlike Ramsey)
        has_phase = "phase" in ds_raw.data_vars
        
        return has_time and not has_phase