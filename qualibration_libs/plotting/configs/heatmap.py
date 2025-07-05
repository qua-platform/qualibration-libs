"""
Specialized configurations for heatmap-style plots (2D colormaps with overlays).

This module provides configurations for experiments that produce 2D heatmap plots
with fit overlays and markers, such as flux vs frequency or power vs frequency sweeps.
"""

from typing import Dict, List, Optional, Union

from pydantic import Field

from .base import (ColorbarConfig, DualAxisConfig, HeatmapTraceConfig,
                   LayoutConfig, LineOverlayConfig, MarkerOverlayConfig,
                   PlotConfig, create_dual_axis, create_heatmap_trace,
                   create_optimal_marker, create_vertical_line)
from .visual_standards import AxisLabels, SubplotSpacing


class HeatmapConfig(PlotConfig):
    """Configuration for 2D heatmap plots with overlays."""
    
    plot_family: str = "heatmap"
    traces: List[HeatmapTraceConfig]
    overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]] = Field(default_factory=list)
    subplot_spacing: Dict[str, float] = Field(default_factory=lambda: {
        "horizontal": SubplotSpacing.HEATMAP_HORIZONTAL,
        "vertical": SubplotSpacing.HEATMAP_VERTICAL
    })
    dual_axis: Optional[DualAxisConfig] = None


# ===== STANDARD CONFIGURATIONS =====

def create_flux_spectroscopy_config() -> HeatmapConfig:
    """Create configuration for resonator spectroscopy vs flux (02c)."""
    
    layout = LayoutConfig(
        title="Resonator Spectroscopy: Flux vs Frequency",
        x_axis_title=AxisLabels.FLUX_BIAS_V,
        y_axis_title=AxisLabels.RF_FREQUENCY_GHZ
    )
    
    # Main heatmap trace
    heatmap_trace = create_heatmap_trace(
        x_source="flux_bias",
        y_source="freq_GHz",
        z_source="IQ_abs",
        name="Raw Data",
        colorbar_title="|IQ|"
    )
    heatmap_trace.hover_template = (
        "Flux [V]: %{x:.3f}<br>"
        "Freq [GHz]: %{y:.3f}<br>"
        "Current [A]: %{customdata:.6f}<br>"
        "|IQ|: %{z:.3f}<extra>Qubit {qubit_id}</extra>"
    )
    heatmap_trace.custom_data_sources = ["attenuated_current"]
    
    # Overlays for successful fits
    overlays = [
        # Sweet spot (idle offset) line
        create_vertical_line(
            position_source="idle_offset",
            condition_source="outcome"
        ),
        
        # Flux minimum line
        LineOverlayConfig(
            orientation="vertical",
            position_source="flux_min",
            condition_source="outcome",
            line_style={
                "color": "#800080",  # Purple for flux minimum
                "width": 2.5,
                "dash": "dash"
            }
        ),
        
        # Sweet spot marker
        create_optimal_marker(
            x_source="idle_offset",
            y_source="sweet_spot_frequency_GHz",
            condition_source="outcome"
        )
    ]
    
    # Dual axis for current
    dual_axis = create_dual_axis(
        top_axis_title=AxisLabels.CURRENT_A,
        top_axis_source="attenuated_current",
        format_string="{:.6f}"
    )
    
    return HeatmapConfig(
        layout=layout,
        traces=[heatmap_trace],
        overlays=overlays,
        dual_axis=dual_axis
    )


def create_amplitude_spectroscopy_config() -> HeatmapConfig:
    """Create configuration for resonator spectroscopy vs amplitude (02b)."""
    
    layout = LayoutConfig(
        title="Resonator Spectroscopy: Power vs Frequency",
        x_axis_title=AxisLabels.FREQUENCY_GHZ,
        y_axis_title=AxisLabels.POWER_DBM
    )
    
    # Main heatmap trace
    heatmap_trace = create_heatmap_trace(
        x_source="freq_GHz",
        y_source="power_dbm",
        z_source="IQ_abs_norm",
        name="Raw Data",
        colorbar_title="|IQ| (normalized)"
    )
    heatmap_trace.hover_template = (
        "Freq [GHz]: %{x:.3f}<br>"
        "Power [dBm]: %{y:.2f}<br>"
        "Detuning [MHz]: %{customdata:.2f}<br>"
        "|IQ|: %{z:.3f}<extra>Qubit {qubit_id}</extra>"
    )
    heatmap_trace.custom_data_sources = ["detuning_MHz"]
    
    # Overlays for successful fits
    overlays = [
        # Resonator frequency line
        create_vertical_line(
            position_source="res_freq_GHz",
            condition_source="outcome"
        ),
        
        # Optimal power line
        LineOverlayConfig(
            orientation="horizontal",
            position_source="optimal_power",
            condition_source="outcome",
            line_style={
                "color": "#FF00FF",  # Magenta for optimal power
                "width": 2.0,
                "dash": "solid"
            }
        )
    ]
    
    return HeatmapConfig(
        layout=layout,
        traces=[heatmap_trace],
        overlays=overlays
    )


def create_power_rabi_2d_config() -> HeatmapConfig:
    """Create configuration for 2D Power Rabi (chevron) plots."""
    
    layout = LayoutConfig(
        title="Power Rabi",
        x_axis_title=AxisLabels.PULSE_AMPLITUDE_MV,
        y_axis_title="Number of pulses"
    )
    
    # Main chevron heatmap
    heatmap_trace = create_heatmap_trace(
        x_source="amp_mV",
        y_source="nb_of_pulses",
        z_source="I_mV",
        name="Chevron Pattern",
        colorbar_title="I [mV]"
    )
    heatmap_trace.hover_template = (
        "Amplitude: %{x:.3f} mV<br>"
        "Pulses: %{y}<br>"
        "Prefactor: %{customdata:.3f}<br>"
        "I: %{z:.3f} mV<extra>Qubit {qubit_id}</extra>"
    )
    heatmap_trace.custom_data_sources = ["amp_prefactor"]
    
    # Optimal amplitude marker
    overlays = [
        LineOverlayConfig(
            orientation="vertical",
            condition_source="outcome",
            condition_value="successful",
            position_source="opt_amp_prefactor",
            line_style={"color": "#FF0000", "width": 2.5, "dash": "dash"}
        )
    ]
    
    # Dual axis for amplitude prefactor
    dual_axis = create_dual_axis(
        top_axis_title=AxisLabels.AMPLITUDE_PREFACTOR,
        top_axis_source="amp_prefactor",
        format_string="{:.3f}"
    )
    
    return HeatmapConfig(
        layout=layout,
        traces=[heatmap_trace],
        overlays=overlays,
        dual_axis=dual_axis,
        subplot_spacing={
            "horizontal": SubplotSpacing.STANDARD_HORIZONTAL,
            "vertical": SubplotSpacing.STANDARD_VERTICAL
        }
    )


def create_power_rabi_2d_state_config() -> HeatmapConfig:
    """Create configuration for 2D Power Rabi with state discrimination."""
    
    layout = LayoutConfig(
        title="Power Rabi",
        x_axis_title=AxisLabels.PULSE_AMPLITUDE_MV,
        y_axis_title="Number of pulses"
    )
    
    # State chevron heatmap
    heatmap_trace = create_heatmap_trace(
        x_source="amp_mV",
        y_source="nb_of_pulses",
        z_source="state",
        name="State Chevron",
        colorbar_title="State"
    )
    heatmap_trace.hover_template = (
        "Amplitude: %{x:.3f} mV<br>"
        "Pulses: %{y}<br>"
        "Prefactor: %{customdata:.3f}<br>"
        "State: %{z}<extra>Qubit {qubit_id}</extra>"
    )
    heatmap_trace.custom_data_sources = ["amp_prefactor"]
    
    # Use Viridis colorscale like original (state data uses same as I/Q data)
    heatmap_trace.colorscale = "Viridis"
    
    # Optimal amplitude marker
    overlays = [
        create_vertical_line(
            position_source="opt_amp_mV",
            condition_source="outcome"
        )
    ]
    
    # Dual axis for amplitude prefactor
    dual_axis = create_dual_axis(
        top_axis_title=AxisLabels.AMPLITUDE_PREFACTOR,
        top_axis_source="amp_prefactor",
        format_string="{:.3f}"
    )
    
    return HeatmapConfig(
        layout=layout,
        traces=[heatmap_trace],
        overlays=overlays,
        dual_axis=dual_axis,
        subplot_spacing={
            "horizontal": SubplotSpacing.STANDARD_HORIZONTAL,
            "vertical": SubplotSpacing.STANDARD_VERTICAL
        }
    )


# ===== CONFIGURATION REGISTRY =====

HEATMAP_CONFIGS = {
    "flux_spectroscopy": create_flux_spectroscopy_config,
    "amplitude_spectroscopy": create_amplitude_spectroscopy_config,
    "power_rabi_2d": create_power_rabi_2d_config,
    "power_rabi_2d_state": create_power_rabi_2d_state_config,
}


def get_heatmap_config(config_name: str) -> HeatmapConfig:
    """Get a predefined heatmap configuration."""
    if config_name not in HEATMAP_CONFIGS:
        available = ", ".join(HEATMAP_CONFIGS.keys())
        raise ValueError(f"Unknown heatmap config '{config_name}'. Available: {available}")
    
    return HEATMAP_CONFIGS[config_name]()