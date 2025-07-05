"""
Pre-built configuration templates for common plotting scenarios.

This module provides ready-to-use configuration templates that developers
can use directly or customize for their specific needs, reducing boilerplate
and ensuring consistency across experiments.
"""

from typing import Optional, Dict, Any, Union

from .builder import PlotConfigurationBuilder
from .spectroscopy import SpectroscopyConfig
from .heatmap import HeatmapConfig
from .visual_standards import AxisLabels, Colors, LineStyles


class ConfigurationTemplates:
    """Collection of pre-built configuration templates for common use cases.
    
    These templates provide sensible defaults for various experiment types
    and can be used directly or as starting points for customization.
    """
    
    @staticmethod
    def simple_spectroscopy(
        x_source: str = "full_freq_GHz",
        y_source: str = "IQ_abs_mV",
        title: str = "Spectroscopy",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> SpectroscopyConfig:
        """Simple 1D spectroscopy template with raw data and fit.
        
        Args:
            x_source: Data source for X-axis (default: full_freq_GHz)
            y_source: Data source for Y-axis (default: IQ_abs_mV)
            title: Plot title
            x_label: X-axis label (defaults to frequency)
            y_label: Y-axis label (defaults to amplitude)
            
        Returns:
            Configured SpectroscopyConfig
        """
        return (PlotConfigurationBuilder()
            .title(title)
            .x_axis(x_label or AxisLabels.RF_FREQUENCY_GHZ)
            .y_axis(y_label or AxisLabels.IQ_AMPLITUDE_MV)
            .add_raw_trace(x_source, y_source, "Raw Data")
            .add_fit_trace(x_source, f"fitted_{y_source}", "Fit")
            .build())
    
    @staticmethod
    def resonator_spectroscopy() -> SpectroscopyConfig:
        """Standard resonator spectroscopy (02a) template."""
        return (PlotConfigurationBuilder()
            .title("Resonator Spectroscopy")
            .x_axis(AxisLabels.RF_FREQUENCY_GHZ)
            .y_axis(AxisLabels.IQ_AMPLITUDE_MV)
            .add_raw_trace("full_freq_GHz", "IQ_abs_mV", "Raw Data")
            .add_fit_trace("full_freq_GHz", "fitted_data_mV", "Lorentzian Fit")
            .add_dual_axis("Detuning [MHz]", "detuning_MHz", "{:.1f}")
            .plot_family("spectroscopy")
            .build())
    
    @staticmethod
    def phase_spectroscopy() -> SpectroscopyConfig:
        """Phase-based spectroscopy template."""
        return (PlotConfigurationBuilder()
            .title("Resonator Spectroscopy (Phase)")
            .x_axis(AxisLabels.RF_FREQUENCY_GHZ)
            .y_axis(AxisLabels.PHASE_DEG)
            .add_raw_trace("full_freq_GHz", "phase_deg", "Raw Phase")
            .add_fit_trace("full_freq_GHz", "fitted_phase_deg", "Phase Fit")
            .plot_family("spectroscopy")
            .build())
    
    @staticmethod
    def power_rabi_1d(
        include_prefactor_axis: bool = True
    ) -> SpectroscopyConfig:
        """1D Power Rabi template.
        
        Args:
            include_prefactor_axis: Whether to include dual axis for amplitude prefactor
            
        Returns:
            Configured SpectroscopyConfig
        """
        builder = (PlotConfigurationBuilder()
            .title("Power Rabi")
            .x_axis(AxisLabels.PULSE_AMPLITUDE_MV)
            .y_axis(AxisLabels.I_MV)
            .add_raw_trace("amp_mV", "I_mV", "Raw Data")
            .add_fit_trace("amp_mV", "fitted_data_mV", "Sinusoidal Fit")
            .plot_family("spectroscopy"))
        
        if include_prefactor_axis:
            builder.add_dual_axis("Amplitude Prefactor", "amp_prefactor", "{:.3f}")
            
        return builder.build()
    
    @staticmethod
    def power_rabi_2d() -> HeatmapConfig:
        """2D Power Rabi (Chevron) template."""
        return (PlotConfigurationBuilder()
            .title("Power Rabi Chevron")
            .x_axis(AxisLabels.PULSE_AMPLITUDE_MV)
            .y_axis(AxisLabels.NUMBER_OF_PULSES)
            .add_heatmap_trace("amp_mV", "nb_of_pulses", "IQ_abs_mV", "Signal")
            .plot_family("heatmap")
            .build())
    
    @staticmethod
    def flux_spectroscopy() -> HeatmapConfig:
        """Flux spectroscopy template."""
        return (PlotConfigurationBuilder()
            .title("Flux Spectroscopy")
            .x_axis(AxisLabels.FLUX_BIAS_V)
            .y_axis(AxisLabels.RF_FREQUENCY_GHZ)
            .add_heatmap_trace("flux_bias", "freq_GHz", "IQ_abs_mV", colorbar_title="|IQ| [mV]")
            .add_vertical_line("idle_offset", "outcome", "successful")
            .add_vertical_line("flux_min", "outcome", "successful")
            .add_optimal_marker("idle_offset", "sweet_spot_frequency")
            .plot_family("heatmap")
            .build())
    
    @staticmethod
    def amplitude_spectroscopy() -> HeatmapConfig:
        """Resonator spectroscopy vs amplitude template."""
        return (PlotConfigurationBuilder()
            .title("Resonator Spectroscopy vs Power")
            .x_axis(AxisLabels.DETUNING_MHZ)
            .y_axis(AxisLabels.POWER_DBM)
            .add_heatmap_trace("detuning_MHz", "power_dbm", "freq_shift_MHz", 
                             colorbar_title="Frequency Shift [MHz]")
            .add_optimal_marker("optimal_detuning", "optimal_power")
            .plot_family("heatmap")
            .build())
    
    @staticmethod
    def ramsey_experiment() -> SpectroscopyConfig:
        """Ramsey experiment template."""
        return (PlotConfigurationBuilder()
            .title("Ramsey Experiment")
            .x_axis(AxisLabels.IDLE_TIME_NS)
            .y_axis(AxisLabels.STATE)
            .add_raw_trace("idle_time_ns", "state", "Raw Data")
            .add_fit_trace("idle_time_ns", "fitted_state", "Oscillation Fit")
            .plot_family("spectroscopy")
            .build())
    
    @staticmethod
    def t1_experiment() -> SpectroscopyConfig:
        """T1 relaxation experiment template."""
        return (PlotConfigurationBuilder()
            .title("T1 Relaxation")
            .x_axis(AxisLabels.WAIT_TIME_NS)
            .y_axis(AxisLabels.STATE)
            .add_raw_trace("wait_time_ns", "state", "Raw Data")
            .add_fit_trace("wait_time_ns", "fitted_state", "Exponential Fit")
            .plot_family("spectroscopy")
            .build())
    
    @staticmethod
    def with_optimal_marker(
        config: SpectroscopyConfig,
        x_source: str,
        y_source: str
    ) -> SpectroscopyConfig:
        """Add an optimal point marker to any configuration.
        
        Args:
            config: Base configuration to modify
            x_source: Data source for marker X position
            y_source: Data source for marker Y position
            
        Returns:
            Modified configuration with marker
        """
        config.overlays.append(
            PlotConfigurationBuilder()
            ._create_optimal_marker(x_source, y_source)
        )
        return config
    
    @staticmethod
    def with_threshold_line(
        config: SpectroscopyConfig,
        threshold_value: float,
        orientation: str = "horizontal",
        label: str = "Threshold"
    ) -> SpectroscopyConfig:
        """Add a threshold line to any configuration.
        
        Args:
            config: Base configuration to modify
            threshold_value: Value for the threshold line
            orientation: "horizontal" or "vertical"
            label: Label for the threshold line
            
        Returns:
            Modified configuration with threshold line
        """
        # This would require extending the builder to support constant value lines
        # For now, this serves as a template for future enhancement
        return config
    
    @staticmethod
    def custom_heatmap(
        x_source: str,
        y_source: str,
        z_source: str,
        title: str,
        x_label: str,
        y_label: str,
        colorbar_title: str = "Value",
        colorscale: str = Colors.HEATMAP_COLORSCALE
    ) -> HeatmapConfig:
        """Customizable heatmap template.
        
        Args:
            x_source: X-axis data source
            y_source: Y-axis data source
            z_source: Z-axis (color) data source
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            colorbar_title: Colorbar title
            colorscale: Colorscale name
            
        Returns:
            Configured HeatmapConfig
        """
        return (PlotConfigurationBuilder()
            .title(title)
            .x_axis(x_label)
            .y_axis(y_label)
            .add_heatmap_trace(x_source, y_source, z_source, 
                             colorbar_title=colorbar_title,
                             colorscale=colorscale)
            .build())
    
    @staticmethod
    def multi_trace_spectroscopy(
        traces: Dict[str, Dict[str, str]],
        title: str = "Multi-Trace Spectroscopy",
        x_label: str = AxisLabels.RF_FREQUENCY_GHZ,
        y_label: str = AxisLabels.IQ_AMPLITUDE_MV
    ) -> SpectroscopyConfig:
        """Template for spectroscopy with multiple traces.
        
        Args:
            traces: Dictionary of trace definitions
                   {name: {"x": x_source, "y": y_source, "style": style_dict}}
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            
        Returns:
            Configured SpectroscopyConfig
        """
        builder = (PlotConfigurationBuilder()
            .title(title)
            .x_axis(x_label)
            .y_axis(y_label))
        
        for name, trace_def in traces.items():
            builder.add_raw_trace(
                trace_def["x"],
                trace_def["y"],
                name,
                style_override=trace_def.get("style")
            )
            
        return builder.build()


# Quick template access functions

def get_template(experiment_type: str) -> Union[SpectroscopyConfig, HeatmapConfig]:
    """Get a configuration template by experiment type name.
    
    Args:
        experiment_type: Type of experiment (e.g., "resonator_spectroscopy", "power_rabi")
        
    Returns:
        Pre-configured template
        
    Raises:
        ValueError: If experiment type is not recognized
    """
    templates = {
        "resonator_spectroscopy": ConfigurationTemplates.resonator_spectroscopy,
        "phase_spectroscopy": ConfigurationTemplates.phase_spectroscopy,
        "power_rabi": ConfigurationTemplates.power_rabi_1d,
        "power_rabi_1d": ConfigurationTemplates.power_rabi_1d,
        "power_rabi_2d": ConfigurationTemplates.power_rabi_2d,
        "flux_spectroscopy": ConfigurationTemplates.flux_spectroscopy,
        "amplitude_spectroscopy": ConfigurationTemplates.amplitude_spectroscopy,
        "ramsey": ConfigurationTemplates.ramsey_experiment,
        "t1": ConfigurationTemplates.t1_experiment,
    }
    
    if experiment_type not in templates:
        available = ", ".join(sorted(templates.keys()))
        raise ValueError(
            f"Unknown experiment type: {experiment_type}. "
            f"Available templates: {available}"
        )
    
    return templates[experiment_type]()


def customize_template(
    experiment_type: str,
    **customizations
) -> Union[SpectroscopyConfig, HeatmapConfig]:
    """Get a template and apply customizations.
    
    Args:
        experiment_type: Base experiment type
        **customizations: Keyword arguments to customize the template
                         Common options: title, x_source, y_source, x_label, y_label
                         
    Returns:
        Customized configuration
        
    Example:
        >>> config = customize_template(
        ...     "resonator_spectroscopy",
        ...     title="My Custom Resonator Sweep",
        ...     y_source="phase_deg",
        ...     y_label="Phase [deg]"
        ... )
    """
    # Get base template
    base_config = get_template(experiment_type)
    
    # Apply customizations
    if "title" in customizations:
        base_config.layout.title = customizations["title"]
        
    if "x_label" in customizations:
        base_config.layout.x_axis_title = customizations["x_label"]
        
    if "y_label" in customizations:
        base_config.layout.y_axis_title = customizations["y_label"]
        
    # Update trace sources if specified
    if "x_source" in customizations or "y_source" in customizations:
        for trace in base_config.traces:
            if "x_source" in customizations:
                trace.x_source = customizations["x_source"]
            if "y_source" in customizations:
                trace.y_source = customizations["y_source"]
                
        for trace in base_config.fit_traces:
            if "x_source" in customizations:
                trace.x_source = customizations["x_source"]
            if "y_source" in customizations:
                trace.y_source = f"fitted_{customizations['y_source']}"
    
    return base_config


# Template sets for related experiments

class TemplateSets:
    """Groups of related templates for comprehensive experiment campaigns."""
    
    @staticmethod
    def full_resonator_characterization() -> Dict[str, Union[SpectroscopyConfig, HeatmapConfig]]:
        """Complete set of resonator characterization templates.
        
        Returns:
            Dictionary of configurations for full resonator characterization
        """
        return {
            "basic_spectroscopy": ConfigurationTemplates.resonator_spectroscopy(),
            "phase_response": ConfigurationTemplates.phase_spectroscopy(),
            "power_dependence": ConfigurationTemplates.amplitude_spectroscopy(),
            "flux_dependence": customize_template(
                "flux_spectroscopy",
                title="Resonator vs Flux",
                y_label=AxisLabels.DETUNING_MHZ
            ),
        }
    
    @staticmethod
    def qubit_calibration_set() -> Dict[str, Union[SpectroscopyConfig, HeatmapConfig]]:
        """Standard qubit calibration template set.
        
        Returns:
            Dictionary of configurations for qubit calibration
        """
        return {
            "power_rabi_1d": ConfigurationTemplates.power_rabi_1d(),
            "power_rabi_chevron": ConfigurationTemplates.power_rabi_2d(),
            "ramsey": ConfigurationTemplates.ramsey_experiment(),
            "t1": ConfigurationTemplates.t1_experiment(),
        }