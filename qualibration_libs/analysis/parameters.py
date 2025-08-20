from typing import Any, Dict, List

from pydantic import BaseModel, Field

# =============================================================================
# Core Parameter Descriptors
# =============================================================================

class Parameter(BaseModel):
    """A descriptor for a single analysis or signal processing parameter."""
    value: Any = Field(..., description="The value of the parameter.")
    description: str = Field(..., description="A brief explanation of the parameter's role.")
    used_in: List[str] = Field(default_factory=list, description="List of functions/methods where this parameter is used.")

class AnalysisParameterGroup(BaseModel):
    """Base class for a group of analysis parameters."""
    pass

# =============================================================================
# Parameter Group Definitions
# =============================================================================

class CommonSignalProcessing(AnalysisParameterGroup):
    """
    Common parameters for signal processing, used across multiple analysis types,
    especially in feature_detection.py.
    """
    baseline_window_size: Parameter = Parameter(
        value=51,
        description="Window size for Savitzky-Golay filter for baseline estimation.",
        used_in=["_preprocess_signal_for_peak_detection"]
    )
    smooth_window_size: Parameter = Parameter(
        value=21,
        description="Window size for Savitzky-Golay filter for signal smoothing.",
        used_in=["_preprocess_signal_for_peak_detection", "_analyze_signal_for_resonance"]
    )
    polyorder: Parameter = Parameter(
        value=3,
        description="Polynomial order for the Savitzky-Golay filter.",
        used_in=["_preprocess_signal_for_peak_detection", "_analyze_signal_for_resonance"]
    )
    noise_prominence_factor: Parameter = Parameter(
        value=3.0,
        description="Factor to multiply by noise level to set peak detection prominence.",
        used_in=["_num_peaks", "_main_peak_snr", "_main_peak_asymmetry_skew", "_analyze_signal_for_resonance"]
    )
    peak_distance: Parameter = Parameter(
        value=10,
        description="Required minimal horizontal distance (in samples) between neighboring peaks.",
        used_in=["_detect_peaks_with_consistent_parameters", "_analyze_signal_for_resonance"]
    )
    peak_width: Parameter = Parameter(
        value=3,
        description="Required minimal peak width (in samples).",
        used_in=["_detect_peaks_with_consistent_parameters", "_analyze_signal_for_resonance"]
    )
    min_skew_window_size: Parameter = Parameter(
        value=3,
        description="Minimum window size for calculating skewness.",
        used_in=["_main_peak_asymmetry_skew"]
    )
    min_window_radius: Parameter = Parameter(
        value=5,
        description="Minimum radius for the window used in skewness calculation.",
        used_in=["_main_peak_asymmetry_skew"]
    )
    noise_floor: Parameter = Parameter(
        value=1e-12,
        description="A small value to prevent division by zero when noise is zero.",
        used_in=["_main_peak_snr"]
    )
    epsilon: Parameter = Parameter(
        value=1e-9,
        description="A small constant to prevent division by zero in various calculations.",
        used_in=["_calculate_snr", "_has_resonator_trace"]
    )

class ResonatorSpectroscopyQualityChecks(AnalysisParameterGroup):
    """Parameters used for quality checks in Resonator Spectroscopy analysis (02a)."""
    min_snr: Parameter = Parameter(
        value=2.5,
        description="Minimum acceptable Signal-to-Noise Ratio.",
        used_in=["_determine_resonator_outcome"]
    )
    snr_for_distortion: Parameter = Parameter(
        value=5.0,
        description="SNR threshold above which stricter distortion checks are applied.",
        used_in=["_is_peak_too_wide", "_determine_resonator_outcome"]
    )
    min_asymmetry: Parameter = Parameter(
        value=0.4,
        description="Minimum acceptable asymmetry ratio (right_width / left_width).",
        used_in=["_is_peak_shape_distorted"]
    )
    max_asymmetry: Parameter = Parameter(
        value=2.5,
        description="Maximum acceptable asymmetry ratio (right_width / left_width).",
        used_in=["_is_peak_shape_distorted"]
    )
    max_skewness: Parameter = Parameter(
        value=1.1,
        description="Maximum acceptable absolute skewness of the peak.",
        used_in=["_is_peak_shape_distorted"]
    )
    distorted_fraction_low_snr: Parameter = Parameter(
        value=0.15,
        description="Maximum FWHM-to-sweep-span ratio for peaks with low SNR.",
        used_in=["_is_peak_too_wide"]
    )
    distorted_fraction_high_snr: Parameter = Parameter(
        value=0.25,
        description="Maximum FWHM-to-sweep-span ratio for peaks with high SNR.",
        used_in=["_is_peak_too_wide"]
    )
    fwhm_absolute_threshold_hz: Parameter = Parameter(
        value=3e6,
        description="Absolute threshold for FWHM in Hz; peaks wider than this are flagged.",
        used_in=["_is_peak_too_wide"]
    )
    nrmse_threshold: Parameter = Parameter(
        value=0.3,
        description="Normalized Root Mean Square Error threshold for goodness-of-fit.",
        used_in=["_is_peak_shape_distorted", "_determine_resonator_outcome"]
    )
    fit_window_radius_multiple: Parameter = Parameter(
        value=1.5,
        description="Multiple of the FWHM to define the fitting window radius.",
        used_in=["_fit_multi_peak_resonator"],
    )
    maxfev: Parameter = Parameter(
        value=20000,
        description="Maximum number of function evaluations for curve_fit.",
        used_in=["_fit_multi_peak_resonator"],
    )

class ResonatorSpectroscopyVsPowerQualityChecks(AnalysisParameterGroup):
    """Parameters for Resonator Spectroscopy vs Power analysis (02b)."""
    snr_threshold: Parameter = Parameter(value=2.0, description="SNR threshold for a single power slice to be considered a 'good' fit.", used_in=["_analyze_signal_for_resonance"])
    flatness_n_points: Parameter = Parameter(value=5, description="Number of initial points to check for flatness in the resonance trace.", used_in=["_calculate_comprehensive_fit_metrics"])
    noise_n_points: Parameter = Parameter(value=5, description="Number of initial points to use for noise and standard deviation calculations.", used_in=["_calculate_comprehensive_fit_metrics"])
    snr_min: Parameter = Parameter(value=2.0, description="Overall minimum SNR for the experiment to be successful.", used_in=["_determine_amplitude_outcome"])
    low_power_center_std_threshold_hz: Parameter = Parameter(value=1e6, description="Threshold for the standard deviation of the resonator center at low powers.", used_in=["_determine_amplitude_outcome"])
    gs_detection_n_low_powers: Parameter = Parameter(value=8, description="Number of low power points to check for ground state presence.", used_in=["_determine_amplitude_outcome", "_detect_ground_state_presence"])
    gs_detection_min_count: Parameter = Parameter(value=5, description="Minimum number of points required to confirm ground state detection.", used_in=["_determine_amplitude_outcome", "_detect_ground_state_presence"])
    gs_detection_window: Parameter = Parameter(value=2, description="Window (in indices) around the mode to count points for ground state detection.", used_in=["_determine_amplitude_outcome", "_detect_ground_state_presence"])
    optimal_power_n_clusters: Parameter = Parameter(value=2, description="Number of clusters (typically ground and excited) to use for KMeans optimal power detection.", used_in=["_assign_optimal_power_clustering"])
    optimal_power_margin_db: Parameter = Parameter(value=3.0, description="Safety margin in dB to subtract from the detected optimal power.", used_in=["_assign_optimal_power_clustering"])
    
class ResonatorSpectroscopyVsFluxQualityChecks(AnalysisParameterGroup):
    """Parameters for Resonator Spectroscopy vs Flux analysis (02c)."""
    nan_fraction_threshold: Parameter = Parameter(value=0.8, description="Maximum fraction of NaN values in peak frequencies before failing a qubit.", used_in=["_evaluate_qubit_data_quality"])
    flat_std_rel_threshold: Parameter = Parameter(value=1e-6, description="Relative threshold for standard deviation to detect a flat (no modulation) signal.", used_in=["_evaluate_qubit_data_quality"])
    amp_rel_threshold: Parameter = Parameter(value=0.01, description="Relative amplitude threshold to confirm that oscillations are present in the fit.", used_in=["_has_oscillations_in_fit"])
    snr_min: Parameter = Parameter(value=2.0, description="Minimum SNR for the overall experiment to be considered successful.", used_in=["_determine_flux_outcome"])
    resonator_trace_smooth_sigma: Parameter = Parameter(value=1.5, description="Sigma for Gaussian filter when smoothing the resonator trace.", used_in=["_has_resonator_trace"])
    resonator_trace_dip_threshold: Parameter = Parameter(value=0.01, description="Minimum relative dip depth to be considered a valid resonator trace.", used_in=["_has_resonator_trace"])
    resonator_trace_gradient_threshold: Parameter = Parameter(value=0.001, description="Minimum gradient of the resonator trace to be considered valid.", used_in=["_has_resonator_trace"])
    min_flux_modulation_hz: Parameter = Parameter(value=1e6, description="Minimum frequency modulation range in Hz to consider the flux modulation significant.", used_in=["_has_insufficient_flux_modulation"])
    flux_modulation_smooth_sigma: Parameter = Parameter(value=1.5, description="Sigma for Gaussian filter when smoothing the flux modulation curve.", used_in=["_has_insufficient_flux_modulation"])
    resonance_correction_confidence_threshold: Parameter = Parameter(value=-0.6, description="Confidence threshold below which resonator frequency is re-calculated.", used_in=["_apply_frequency_correction_if_needed"])
    resonance_correction_window: Parameter = Parameter(value=5, description="Window size around the idle flux point to check for fit confidence.", used_in=["_apply_frequency_correction_if_needed"])
    sweet_spot_threshold_fraction: Parameter = Parameter(value=0.05, description="Threshold for sweet spot refinement. Defines the tolerance above the minimum frequency as a fraction of the total oscillation amplitude to identify the 'flat' region.", used_in=["_refine_min_offset"])

class PowerRabiQualityChecks(AnalysisParameterGroup):
    """Parameters for Power Rabi analysis (04b)."""
    min_fit_quality: Parameter = Parameter(value=0.5, description="Minimum R-squared value for a fit to be considered successful.", used_in=["_determine_qubit_outcome"])
    min_amplitude: Parameter = Parameter(value=0.01, description="Minimum signal amplitude to be considered a valid Rabi oscillation.", used_in=["_determine_qubit_outcome"])
    max_amp_prefactor: Parameter = Parameter(value=10.0, description="Maximum allowed amplitude prefactor to prevent runaway values.", used_in=["_determine_qubit_outcome"])
    snr_min: Parameter = Parameter(value=2.5, description="Minimum acceptable Signal-to-Noise ratio.", used_in=["_determine_qubit_outcome"])
    autocorrelation_r1_threshold: Parameter = Parameter(value=0.8, description="Autocorrelation at lag 1 threshold to determine if a signal is fit-worthy.", used_in=["_evaluate_signal_fit_worthiness"])
    autocorrelation_r2_threshold: Parameter = Parameter(value=0.5, description="Autocorrelation at lag 2 threshold to determine if a signal is fit-worthy.", used_in=["_evaluate_signal_fit_worthiness"])
    chevron_modulation_threshold: Parameter = Parameter(value=0.4, description="Threshold for detecting chevron-like modulation in 2D Power Rabi.", used_in=["_detect_chevron_modulation"])

# =============================================================================
# Central Configuration Manager
# =============================================================================

class AnalysisConfigManager:
    """A singleton manager for all analysis-related parameter configurations."""
    _instance = None
    _registry: Dict[str, AnalysisParameterGroup] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, name: str, config_group: AnalysisParameterGroup):
        """Registers a parameter group with a unique name."""
        if name in self._registry:
            # You might want to log a warning here in a real application
            pass
        self._registry[name] = config_group

    def get(self, name: str) -> AnalysisParameterGroup:
        """Retrieves a registered parameter group by its name."""
        if name not in self._registry:
            raise KeyError(f"No configuration group registered for '{name}'.")
        return self._registry[name]

    def update_from_dict(self, config_dict: Dict[str, Dict[str, Any]]):
        """
        Updates the configuration from a dictionary.
        
        Example:
            config_manager.update_from_dict({
                "resonator_spectroscopy_qc": {
                    "min_snr": {"value": 3.0}
                }
            })
        """
        for group_name, group_updates in config_dict.items():
            if group_name in self._registry:
                group_instance = self._registry[group_name]
                updated_data = group_instance.dict()
                for param_name, param_updates in group_updates.items():
                    if param_name in updated_data and 'value' in param_updates:
                         updated_data[param_name]['value'] = param_updates['value']
                
                # Re-create the Pydantic model with updated data
                self._registry[group_name] = group_instance.__class__(**updated_data)


# =============================================================================
# Instantiation and Registration
# =============================================================================

analysis_config_manager = AnalysisConfigManager()

# Register all parameter groups
analysis_config_manager.register("common_signal_processing", CommonSignalProcessing())
analysis_config_manager.register("resonator_spectroscopy_qc", ResonatorSpectroscopyQualityChecks())
analysis_config_manager.register("resonator_spectroscopy_vs_power_qc", ResonatorSpectroscopyVsPowerQualityChecks())
analysis_config_manager.register("resonator_spectroscopy_vs_flux_qc", ResonatorSpectroscopyVsFluxQualityChecks())
analysis_config_manager.register("power_rabi_qc", PowerRabiQualityChecks()) 