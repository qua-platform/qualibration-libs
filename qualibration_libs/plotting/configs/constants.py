"""
Constants for the plotting module.

This module centralizes all hardcoded values, strings, and magic numbers
used throughout the plotting codebase to improve maintainability and
reduce duplication.
"""

from enum import Enum
from typing import Final


class CoordinateNames:
    """Standard coordinate and variable names used in datasets."""
    
    # Core coordinates
    QUBIT: Final[str] = "qubit"
    OUTCOME: Final[str] = "outcome"
    SUCCESSFUL: Final[str] = "successful"
    
    # Frequency-related
    DETUNING: Final[str] = "detuning"
    DETUNING_MHZ: Final[str] = "detuning_MHz"
    FREQ_GHZ: Final[str] = "freq_GHz"
    FULL_FREQ: Final[str] = "full_freq"
    FULL_FREQ_GHZ: Final[str] = "full_freq_GHz"
    FREQUENCY: Final[str] = "frequency"
    
    # Power/amplitude-related
    POWER: Final[str] = "power"
    POWER_DBM: Final[str] = "power_dbm"
    AMP_MV: Final[str] = "amp_mV"
    AMP_PREFACTOR: Final[str] = "amp_prefactor"
    AMPLITUDE: Final[str] = "amplitude"
    FULL_AMP: Final[str] = "full_amp"
    
    # Flux-related
    FLUX_BIAS: Final[str] = "flux_bias"
    CURRENT: Final[str] = "current"
    ATTENUATED_CURRENT: Final[str] = "attenuated_current"
    
    # IQ data
    I: Final[str] = "I"
    Q: Final[str] = "Q"
    IQ_ABS: Final[str] = "IQ_abs"
    IQ_ABS_MV: Final[str] = "IQ_abs_mV"
    IQ_ABS_NORM: Final[str] = "IQ_abs_norm"
    PHASE: Final[str] = "phase"
    PHASE_DEG: Final[str] = "phase_deg"
    
    # State-related
    STATE: Final[str] = "state"
    NB_OF_PULSES: Final[str] = "nb_of_pulses"
    
    # Fit-related
    FIT_VALS: Final[str] = "fit_vals"
    FITTED_DATA_MV: Final[str] = "fitted_data_mV"
    X180_AMPLITUDE: Final[str] = "x180_amplitude"
    X90_AMPLITUDE: Final[str] = "x90_amplitude"


class ExperimentTypes(str, Enum):
    """Supported experiment types."""
    
    POWER_RABI = "power_rabi"
    FLUX_SPECTROSCOPY = "flux_spectroscopy" 
    RESONATOR_SPECTROSCOPY = "resonator_spectroscopy"
    AMPLITUDE_SPECTROSCOPY = "amplitude_spectroscopy"
    RESONATOR_SPECTROSCOPY_VS_FLUX = "resonator_spectroscopy_vs_flux"
    RESONATOR_SPECTROSCOPY_VS_AMPLITUDE = "resonator_spectroscopy_vs_amplitude"
    RESONATOR_SPECTROSCOPY_VS_POWER = "resonator_spectroscopy_vs_power"
    TWO_TONE_SPECTROSCOPY = "two_tone_spectroscopy"
    RAMSEY = "ramsey"
    T1 = "t1"
    UNKNOWN = "unknown"
    AUTO = "auto"


class PlotConstants:
    """Numerical constants used in plotting."""
    
    # Percentiles for robust scaling
    DEFAULT_MIN_PERCENTILE: Final[float] = 2.0
    DEFAULT_MAX_PERCENTILE: Final[float] = 98.0
    
    # Unit conversions
    GHZ_PER_HZ: Final[float] = 1e-9
    MHZ_PER_HZ: Final[float] = 1e-6
    MV_PER_V: Final[float] = 1e3
    
    # Layout spacing
    DEFAULT_SUBPLOT_SPACING: Final[float] = 0.05
    TIGHT_SUBPLOT_SPACING: Final[float] = 0.03
    WIDE_SUBPLOT_SPACING: Final[float] = 0.1
    
    # Figure dimensions
    DEFAULT_FIGURE_WIDTH: Final[int] = 800
    DEFAULT_FIGURE_HEIGHT: Final[int] = 600
    
    # Alpha values
    RAW_DATA_ALPHA: Final[float] = 0.8
    FIT_LINE_ALPHA: Final[float] = 1.0
    OVERLAY_ALPHA: Final[float] = 0.7
    
    # Marker sizes
    DEFAULT_MARKER_SIZE: Final[int] = 6
    OVERLAY_MARKER_SIZE: Final[int] = 10
    
    # Line widths
    DEFAULT_LINE_WIDTH: Final[int] = 2
    FIT_LINE_WIDTH: Final[int] = 3
    OVERLAY_LINE_WIDTH: Final[int] = 2


class PlotModes:
    """Plot display modes."""
    
    LINES: Final[str] = "lines"
    MARKERS: Final[str] = "markers"
    LINES_MARKERS: Final[str] = "lines+markers"
    NONE: Final[str] = "none"


class ColorScales:
    """Standard color scales for heatmaps."""
    
    VIRIDIS: Final[str] = "Viridis"
    PLASMA: Final[str] = "Plasma"
    INFERNO: Final[str] = "Inferno"
    MAGMA: Final[str] = "Magma"
    CIVIDIS: Final[str] = "Cividis"
    TURBO: Final[str] = "Turbo"
    RD_BU: Final[str] = "RdBu"


# Note: AxisLabels are defined in visual_standards.py
# This module focuses on coordinate names and other constants