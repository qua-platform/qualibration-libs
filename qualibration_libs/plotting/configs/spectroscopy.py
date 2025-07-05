"""
Specialized configurations for spectroscopy-style plots (1D line plots with fits).

This module provides configurations for experiments that produce 1D line plots
with fit overlays, such as resonator spectroscopy measurements.
"""

from typing import List, Optional

from pydantic import Field

from .base import (DualAxisConfig, LayoutConfig, PlotConfig, TraceConfig,
                   create_dual_axis, create_fit_trace, create_standard_trace)
from .visual_standards import AxisLabels, HoverTemplates


class SpectroscopyConfig(PlotConfig):
    """Configuration for 1D spectroscopy plots with fit overlays."""
    
    plot_family: str = "spectroscopy"
    dual_axis: Optional[DualAxisConfig] = None


# ===== STANDARD CONFIGURATIONS =====

def create_resonator_spectroscopy_config() -> SpectroscopyConfig:
    """Create standard configuration for resonator spectroscopy (02a)."""
    
    layout = LayoutConfig(
        title="Resonator Spectroscopy",
        x_axis_title=AxisLabels.RF_FREQUENCY_GHZ,
        y_axis_title=AxisLabels.IQ_AMPLITUDE_MV
    )
    
    # Raw amplitude trace
    amplitude_trace = create_standard_trace(
        x_source="full_freq_GHz",
        y_source="IQ_abs_mV",
        name="Raw Data"
    )
    amplitude_trace.hover_template = HoverTemplates.RESONATOR_SPECTROSCOPY
    amplitude_trace.custom_data_sources = ["detuning_MHz"]
    
    # Fit trace
    fit_trace = create_fit_trace(
        x_source="full_freq_GHz",
        y_source="fitted_data_mV",
        name="Lorentzian Fit"
    )
    
    # Dual axis for detuning
    dual_axis = create_dual_axis(
        top_axis_title=AxisLabels.DETUNING_MHZ,
        top_axis_source="detuning_MHz"
    )
    
    return SpectroscopyConfig(
        layout=layout,
        traces=[amplitude_trace],
        fit_traces=[fit_trace],
        dual_axis=dual_axis
    )


def create_phase_spectroscopy_config() -> SpectroscopyConfig:
    """Create configuration for phase spectroscopy plots."""
    
    layout = LayoutConfig(
        title="Resonator Spectroscopy (Phase)",
        x_axis_title=AxisLabels.RF_FREQUENCY_GHZ,
        y_axis_title="Phase [rad]"
    )
    
    # Raw phase trace
    phase_trace = create_standard_trace(
        x_source="full_freq_GHz",
        y_source="phase",
        name="Raw Phase"
    )
    phase_trace.hover_template = (
        "<b>Freq</b>: %{x:.4f} GHz<br>"
        "<b>Detuning</b>: %{customdata[0]:.2f} MHz<br>"
        "<b>Phase</b>: %{y:.3f} rad<extra></extra>"
    )
    phase_trace.custom_data_sources = ["detuning_MHz"]
    
    # Dual axis for detuning
    dual_axis = create_dual_axis(
        top_axis_title=AxisLabels.DETUNING_MHZ,
        top_axis_source="detuning_MHz"
    )
    
    return SpectroscopyConfig(
        layout=layout,
        traces=[phase_trace],
        dual_axis=dual_axis
    )


def create_power_rabi_1d_config() -> SpectroscopyConfig:
    """Create configuration for 1D Power Rabi plots (single pulse)."""
    
    layout = LayoutConfig(
        title="Power Rabi",
        x_axis_title=AxisLabels.PULSE_AMPLITUDE_MV,
        y_axis_title=AxisLabels.ROTATED_I_MV
    )
    
    # Raw Rabi trace
    rabi_trace = create_standard_trace(
        x_source="amp_mV",
        y_source="I_mV",
        name="Raw Data"
    )
    rabi_trace.hover_template = HoverTemplates.POWER_RABI
    rabi_trace.custom_data_sources = ["amp_prefactor"]
    
    # Sinusoidal fit trace
    fit_trace = create_fit_trace(
        x_source="amp_mV",
        y_source="fitted_data_mV",
        name="Sinusoidal Fit"
    )
    
    # Dual axis for amplitude prefactor
    dual_axis = create_dual_axis(
        top_axis_title=AxisLabels.AMPLITUDE_PREFACTOR,
        top_axis_source="amp_prefactor",
        format_string="{:.3f}"
    )
    
    return SpectroscopyConfig(
        layout=layout,
        traces=[rabi_trace],
        fit_traces=[fit_trace],
        dual_axis=dual_axis
    )


def create_power_rabi_state_config() -> SpectroscopyConfig:
    """Create configuration for Power Rabi with state discrimination."""
    
    layout = LayoutConfig(
        title="Power Rabi",
        x_axis_title=AxisLabels.PULSE_AMPLITUDE_MV,
        y_axis_title=AxisLabels.QUBIT_STATE
    )
    
    # State trace
    state_trace = create_standard_trace(
        x_source="amp_mV",
        y_source="state",
        name="Raw Data"
    )
    state_trace.hover_template = (
        "Amplitude: %{x:.3f} mV<br>"
        "Prefactor: %{customdata[0]:.3f}<br>"
        "State: %{y}<extra></extra>"
    )
    state_trace.custom_data_sources = ["amp_prefactor"]
    
    # Fit trace for state
    fit_trace = create_fit_trace(
        x_source="amp_mV",
        y_source="fitted_state",
        name="Fit"
    )
    
    # Dual axis for amplitude prefactor
    dual_axis = create_dual_axis(
        top_axis_title=AxisLabels.AMPLITUDE_PREFACTOR,
        top_axis_source="amp_prefactor",
        format_string="{:.3f}"
    )
    
    return SpectroscopyConfig(
        layout=layout,
        traces=[state_trace],
        fit_traces=[fit_trace],
        dual_axis=dual_axis
    )


# ===== CONFIGURATION REGISTRY =====

SPECTROSCOPY_CONFIGS = {
    "resonator_spectroscopy_amplitude": create_resonator_spectroscopy_config,
    "resonator_spectroscopy_phase": create_phase_spectroscopy_config,
    "power_rabi_1d": create_power_rabi_1d_config,
    "power_rabi_1d_state": create_power_rabi_state_config,
}


def get_spectroscopy_config(config_name: str) -> SpectroscopyConfig:
    """Get a predefined spectroscopy configuration."""
    if config_name not in SPECTROSCOPY_CONFIGS:
        available = ", ".join(SPECTROSCOPY_CONFIGS.keys())
        raise ValueError(f"Unknown spectroscopy config '{config_name}'. Available: {available}")
    
    return SPECTROSCOPY_CONFIGS[config_name]()