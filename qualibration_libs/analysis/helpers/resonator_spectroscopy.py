from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis.feature_detection import (
    find_all_peaks_from_signal, is_peak_shape_distorted, is_peak_too_wide)
from qualibration_libs.analysis.fitting import (calculate_quality_metrics,
                                                generate_lorentzian_fit)
from qualibration_libs.analysis.models import lorentzian_dip
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.analysis.processing import calculate_sweep_span
from scipy.optimize import curve_fit

# Access quality check parameters from the config object
qc_params = analysis_config_manager.get("resonator_spectroscopy_qc")


def fit_multi_peak_resonator(
    ds_qubit: xr.Dataset, fit_qubit: xr.Dataset
) -> Tuple[xr.Dataset, None]:
    """
    Apply advanced fitting for a qubit with multiple detected peaks.

    This method implements a robust fitting pipeline for signals with multiple resonance features:
    1. It inverts the signal to treat dips as peaks.
    2. Finds all significant peaks in the signal.
    3. Selects the most prominent peak as the candidate for the main resonance.
    4. Estimates its width using the signal's curvature (2nd derivative).
    5. Defines a narrow fitting window around the candidate peak.
    6. Guesses initial parameters from the data within the window.
    7. Performs a Lorentzian fit on the windowed data.
    8. Validates the fit and returns the parameters.

    Args:
        ds_qubit: The raw dataset for a single qubit.
        fit_qubit: The initial fit dataset from `peaks_dips`.

    Returns:
        A tuple containing the final fit dataset and the fit parameters dictionary for the qubit.
    """
    signal_1d = ds_qubit.IQ_abs.values
    detuning_axis = ds_qubit.detuning.values

    # Find all dips by inverting the signal for the peak finder
    peaks, properties, _, _ = find_all_peaks_from_signal(-signal_1d)

    if len(peaks) == 0:
        return fit_qubit, None

    main_peak_idx = np.argmax(properties["prominences"])
    main_peak_pos = peaks[main_peak_idx]

    # Estimate width from scipy.signal.peak_widths and define a fit window
    width_samples = properties["widths"][main_peak_idx]

    fit_window_radius = int(width_samples * qc_params.fit_window_radius_multiple.value)
    fit_start = max(0, main_peak_pos - fit_window_radius)
    fit_end = min(len(signal_1d), main_peak_pos + fit_window_radius + 1)

    x_window = detuning_axis[fit_start:fit_end]
    y_window = signal_1d[fit_start:fit_end]

    # Guess initial fit parameters for a dip
    p0_amplitude = properties["prominences"][
        main_peak_idx
    ]  # Positive amplitude for dip depth
    p0_center = detuning_axis[main_peak_pos]
    detuning_step = abs(detuning_axis[1] - detuning_axis[0])
    p0_width = width_samples * detuning_step
    # Use the mean of the window edges for a robust baseline guess
    p0_offset = (
        (y_window[0] + y_window[-1]) / 2 if len(y_window) > 1 else np.mean(y_window)
    )

    # Define bounds for the fit parameters to stabilize the fit
    lower_bounds = [
        0,  # Amplitude > 0
        x_window[0],  # Center within window
        0,  # Width > 0
        np.min(signal_1d),  # Offset within signal range
    ]
    upper_bounds = [
        (np.max(y_window) - np.min(y_window)) * 1.5,  # Amplitude < 1.5x window range
        x_window[-1],  # Center within window
        (x_window[-1] - x_window[0]),  # Width < window size
        np.max(signal_1d),  # Offset within signal range
    ]
    bounds = (lower_bounds, upper_bounds)

    try:
        popt, _ = curve_fit(
            lorentzian_dip,
            x_window,
            y_window,
            p0=[p0_amplitude, p0_center, p0_width / 2, p0_offset],  # HWHM for fit
            bounds=bounds,
            maxfev=qc_params.maxfev.value,
        )
        fit_amp, fit_center, fit_hwhm, fit_offset = popt
        fit_fwhm = abs(fit_hwhm * 2)

        # Update the fit dataset with new values
        fit_qubit["position"] = fit_center
        fit_qubit["width"] = fit_fwhm
        fit_qubit["amplitude"] = fit_amp  # Amplitude is now positive

        # Create a new baseline array with the fitted offset
        new_baseline = xr.DataArray(
            np.full_like(fit_qubit.base_line.values, fit_offset),
            dims=fit_qubit.base_line.dims,
            coords=fit_qubit.base_line.coords,
        )
        fit_qubit["base_line"] = new_baseline

        return fit_qubit, None

    except RuntimeError:
        # If the advanced fit fails, we return the original `peaks_dips` result
        return fit_qubit, None


def calculate_resonator_parameters(
    fit: xr.Dataset, rf_frequencies: np.ndarray
) -> xr.Dataset:
    """
    Calculate resonator frequency and FWHM from fit results.

    Args:
        fit : xr.Dataset
            Dataset with peak detection results
        rf_frequencies : np.ndarray
            Array of RF frequencies for the qubits.

    Returns:
        xr.Dataset
            Dataset with resonator parameters added
    """
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    # Calculate fitted resonator frequency
    res_freq = fit.position + rf_frequencies
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    # Calculate FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    return fit




def extract_qubit_fit_metrics(
    ds_raw: xr.Dataset, fit: xr.Dataset, qubit_name: str
) -> Dict[str, float]:
    """
    Extract all relevant fit metrics for a single qubit.

    Args:
        ds_raw : xr.Dataset
            The raw dataset containing IQ data
        fit : xr.Dataset
            Dataset containing fit results
        qubit_name : str
            Name of the qubit to extract metrics for

    Returns:
        Dict[str, float]
            Dictionary containing all fit metrics for the qubit
    """
    qubit_data = fit.sel(qubit=qubit_name)
    sweep_span = calculate_sweep_span(fit)

    # Get raw data for quality metrics
    raw_data = ds_raw.IQ_abs.sel(qubit=qubit_name).values
    detuning_values = ds_raw.detuning.values

    # Generate the Lorentzian fit and calculate quality metrics
    fitted_data = generate_lorentzian_fit(qubit_data, detuning_values)
    quality_metrics = calculate_quality_metrics(raw_data, fitted_data)

    return {
        "num_peaks": int(qubit_data.num_peaks.values),
        "raw_num_peaks": int(qubit_data.raw_num_peaks.values),
        "snr": float(qubit_data.snr.values),
        "fwhm": float(qubit_data.fwhm.values),
        "sweep_span": sweep_span,
        "asymmetry": float(qubit_data.asymmetry.values),
        "skewness": float(qubit_data.skewness.values),
        **quality_metrics,
    }


def determine_resonator_outcome(
    metrics: Dict[str, float],
) -> str:
    """
    Determine the outcome for resonator spectroscopy based on fit metrics.

    Args:
        metrics : Dict[str, float]
            Dictionary containing fit metrics

    Returns:
        str
            Outcome description
    """
    num_peaks = metrics["num_peaks"]
    raw_num_peaks = metrics["raw_num_peaks"]
    snr = metrics["snr"]
    fwhm = metrics["fwhm"]
    sweep_span = metrics["sweep_span"]
    asymmetry = metrics["asymmetry"]
    skewness = metrics["skewness"]
    nrmse = metrics.get("nrmse", np.inf)

    # Check SNR first
    if snr < qc_params.min_snr.value:
        return "The SNR isn't large enough, consider increasing the number of shots"

    # Check number of peaks
    if num_peaks == 0:
        if snr < qc_params.min_snr.value:
            return (
                "The SNR isn't large enough, consider increasing the number of shots "
                "and ensure you are looking at the correct frequency range"
            )
        return "No peaks were detected, consider changing the frequency range"

    # Check peak shape quality and multiple resonances
    if is_peak_shape_distorted(
        asymmetry,
        skewness,
        nrmse,
        min_asymmetry=qc_params.min_asymmetry.value,
        max_asymmetry=qc_params.max_asymmetry.value,
        max_skewness=qc_params.max_skewness.value,
        nrmse_threshold=qc_params.nrmse_threshold.value,
    ):
        if raw_num_peaks > 1:
            if nrmse <= qc_params.nrmse_threshold.value:
                return "successful"
            else:
                return "Multiple resonances detected, consider adjusting the span of the frequency range"
        return "The peak shape is distorted"

    # Check for peak width issues
    if is_peak_too_wide(
        fwhm,
        sweep_span,
        snr,
        distorted_fraction_low_snr=qc_params.distorted_fraction_low_snr.value,
        snr_for_distortion=qc_params.snr_for_distortion.value,
        distorted_fraction_high_snr=qc_params.distorted_fraction_high_snr.value,
        fwhm_absolute_threshold_hz=qc_params.fwhm_absolute_threshold_hz.value,
    ):
        if snr < qc_params.snr_for_distortion.value:
            return "The SNR isn't large enough and the peak shape is distorted"
        else:
            return "Distorted peak detected"

    # All checks passed
    return "successful" 