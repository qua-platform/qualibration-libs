from __future__ import annotations

import logging
from typing import Dict, Type

import numpy as np
import xarray as xr
from qualibration_libs.analysis.feature_detection import (
    has_insufficient_flux_modulation, has_resonator_trace)
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.analysis.signal import calculate_snr_from_dataset
from scipy.ndimage import gaussian_filter1d

# Access parameter groups from the config manager
qc_params = analysis_config_manager.get("resonator_spectroscopy_vs_flux_qc")
sp_params = analysis_config_manager.get("common_signal_processing")



def evaluate_qubit_data_quality(
    ds: xr.Dataset, peak_freq: xr.Dataset
) -> Dict[str, str]:
    """
    Evaluate data quality for each qubit to determine which can be fit.

    Args:
        ds : xr.Dataset
            Raw dataset
        peak_freq : xr.Dataset
            Peak frequency data

    Returns:
        Dict[str, str]
            Dictionary mapping qubit names to quality assessment outcomes
    """
    qubit_outcomes = {}

    for q in ds.qubit.values:
        # Check for resonator trace
        if not has_resonator_trace(
            ds,
            q,
            smooth_sigma=qc_params.resonator_trace_smooth_sigma.value,
            dip_threshold=qc_params.resonator_trace_dip_threshold.value,
            gradient_threshold=qc_params.resonator_trace_gradient_threshold.value,
        ):
            qubit_outcomes[q] = "no_peaks"
            continue

        # Check for insufficient flux modulation
        if has_insufficient_flux_modulation(
            ds,
            q,
            smooth_sigma=qc_params.flux_modulation_smooth_sigma.value,
            min_modulation_hz=qc_params.min_flux_modulation_hz.value,
        ):
            qubit_outcomes[q] = "no_oscillations"
            continue

        # Check peak frequency quality
        pf = peak_freq.sel(qubit=q)
        pf_vals = pf.values
        n_nan = np.sum(np.isnan(pf_vals))
        frac_nan = n_nan / pf_vals.size
        pf_valid = pf_vals[~np.isnan(pf_vals)]

        if pf_valid.size == 0 or frac_nan > qc_params.nan_fraction_threshold.value:
            qubit_outcomes[q] = "no_peaks"
        else:
            pf_std = np.nanstd(pf_valid)
            pf_mean = np.nanmean(np.abs(pf_valid))
            if pf_mean == 0 or pf_std < qc_params.flat_std_rel_threshold.value * pf_mean:
                qubit_outcomes[q] = "no_peaks"
            else:
                qubit_outcomes[q] = "fit"

    return qubit_outcomes


def calculate_flux_bias_parameters(fit: xr.Dataset) -> xr.Dataset:
    """Calculate flux bias parameters from fit results."""
    # Ensure phase is between -π and π
    flux_idle = -fit.sel(fit_vals="phi").fit_results
    flux_idle = np.mod(flux_idle + np.pi, 2 * np.pi) - np.pi

    # Convert phase to voltage
    flux_idle = flux_idle / fit.sel(fit_vals="f").fit_results / 2 / np.pi
    fit = fit.assign_coords(idle_offset=("qubit", flux_idle.data))
    fit.idle_offset.attrs = {"long_name": "idle flux bias", "units": "V"}

    # Find minimum frequency flux point
    flux_min = flux_idle + ((flux_idle < 0) - 0.5) / fit.sel(fit_vals="f").fit_results
    flux_min = (
        flux_min * (np.abs(flux_min) < 0.5)
        + 0.5 * (flux_min > 0.5)
        - 0.5 * (flux_min < -0.5)
    )
    fit = fit.assign_coords(flux_min=("qubit", flux_min.data))
    fit.flux_min.attrs = {"long_name": "minimum frequency flux bias", "units": "V"}

    return fit


def refine_min_offset(fit: xr.Dataset) -> xr.Dataset:
    """
    Refines the minimum frequency flux offset by finding the center of the
    sweet spot in the data near the fitted value. This corrects for
    deviations of the data from a perfect cosine shape and is robust to noise.
    """
    if "flux_min" not in fit.coords or "peak_freq" not in fit.data_vars:
        return fit

    refined_flux_min_values = fit.flux_min.copy(deep=True)

    for q_name in fit.qubit.values:
        if q_name not in refined_flux_min_values.qubit.values:
            continue

        initial_flux_min = fit.flux_min.sel(qubit=q_name).item()
        if np.isnan(initial_flux_min):
            continue

        peak_freq_data = fit.peak_freq.sel(qubit=q_name).dropna(dim="flux_bias")
        flux_axis = peak_freq_data.flux_bias.values
        freq_values = peak_freq_data.values

        if len(flux_axis) == 0:
            continue

        # Define a search window around the initial fitted minimum.
        # The window is taken as 1/4 of the flux period on either side.
        period = 1.0 / fit.sel(qubit=q_name, fit_vals="f").fit_results.item()
        window_half_width = period / 4.0

        min_flux_in_data = flux_axis.min()
        max_flux_in_data = flux_axis.max()

        search_mask = (
            flux_axis >= max(initial_flux_min - window_half_width, min_flux_in_data)
        ) & (flux_axis <= min(initial_flux_min + window_half_width, max_flux_in_data))

        # If the window is empty for some reason, fall back to the closest point
        if not np.any(search_mask):
            closest_idx = np.argmin(np.abs(flux_axis - initial_flux_min))
            search_mask = np.zeros_like(flux_axis, dtype=bool)
            search_mask[closest_idx] = True

        windowed_flux_axis = flux_axis[search_mask]
        windowed_freq_values = freq_values[search_mask]

        if len(windowed_freq_values) == 0:
            # This should not happen if the fallback is working, but as a safeguard.
            continue

        # Find the minimum frequency in the window and define a threshold
        # to identify the entire sweet spot region.
        min_freq_in_window = np.min(windowed_freq_values)
        fit_amplitude = fit.sel(qubit=q_name, fit_vals="a").fit_results.item()

        # The threshold is a small fraction of the total oscillation amplitude
        # above the minimum, capturing the "flat" part of the curve.
        freq_threshold = min_freq_in_window + (
            qc_params.sweet_spot_threshold_fraction.value * np.abs(fit_amplitude)
        )

        # Find all points within this sweet spot threshold
        sweet_spot_mask = windowed_freq_values <= freq_threshold
        sweet_spot_fluxes = windowed_flux_axis[sweet_spot_mask]

        if len(sweet_spot_fluxes) == 0:
            # Fallback for very sharp "V" shapes where no other points fall
            # within the threshold. In this case, the minimum is the best estimate.
            refined_flux_min = windowed_flux_axis[np.argmin(windowed_freq_values)]
        else:
            # The refined offset is the mean of the flux values in the flat region,
            # which gives the center of the sweet spot and is robust to noise.
            refined_flux_min = np.mean(sweet_spot_fluxes)

        # Update the value in our new DataArray
        refined_flux_min_values.loc[dict(qubit=q_name)] = refined_flux_min

    fit = fit.assign_coords(flux_min=refined_flux_min_values)
    return fit


def calculate_flux_frequency_parameters(
    fit: xr.Dataset, rf_frequencies: np.ndarray
) -> xr.Dataset:
    """Calculate frequency parameters from flux dependence."""
    # Calculate frequency shift at sweet spot
    flux_idle = fit.idle_offset
    freq_shift = fit.peak_freq.sel(flux_bias=flux_idle, method="nearest")
    fit = fit.assign_coords(freq_shift=("qubit", freq_shift.data))
    fit.freq_shift.attrs = {"long_name": "frequency shift", "units": "Hz"}

    # Calculate sweet spot frequency
    fit = fit.assign_coords(
        sweet_spot_frequency=("qubit", freq_shift.data + rf_frequencies)
    )
    fit.sweet_spot_frequency.attrs = {
        "long_name": "sweet spot frequency",
        "units": "Hz",
    }

    return fit



def has_oscillations_in_fit(
    fit: xr.Dataset,
    qubit: str,
    qubit_outcomes: Dict[str, str],
) -> bool:
    """Determine if oscillations were detected in the fit for a given qubit."""
    if qubit_outcomes.get(qubit) in ["no_peaks", "no_oscillations"]:
        return False

    try:
        # Extract fit amplitude
        amp = float(fit.sel(fit_vals="a", qubit=qubit).fit_results.data)
        pf_valid = fit.peak_freq.sel(qubit=qubit).values
        pf_valid = pf_valid[~np.isnan(pf_valid)]
        pf_median = np.median(np.abs(pf_valid)) if pf_valid.size > 0 else 1.0
        return np.abs(amp) >= qc_params.amp_rel_threshold.value * pf_median
    except:
        return False


def determine_flux_outcome(
    params: Dict[str, float],
    initial_outcome: str,
) -> str:
    """Determine the outcome for flux spectroscopy based on parameters."""
    freq_shift = params["freq_shift"]
    flux_min = params["flux_min"]
    flux_idle = params["flux_idle"]
    frequency_span_in_mhz = params["frequency_span_in_mhz"]
    snr = params["snr"]
    has_oscillations = params["has_oscillations"]
    has_anticrossings = params["has_anticrossings"]

    if not has_oscillations:
        return (
            "No oscillations were detected, "
            "consider checking that the flux line is connected or increase the flux range"
        )

    if has_anticrossings:
        return "Anti-crossings were detected, consider adjusting the flux range or checking the device setup"

    snr_low = snr is not None and snr < qc_params.snr_min.value
    if snr_low:
        return "The SNR isn't large enough, consider increasing the number of shots"

    if np.isnan(freq_shift) or np.isnan(flux_min) or np.isnan(flux_idle):
        return "No peaks were detected, consider changing the frequency range"

    if np.abs(freq_shift) >= frequency_span_in_mhz * 1e6:
        return f"Frequency shift {freq_shift * 1e-6:.0f} MHz exceeds span {frequency_span_in_mhz} MHz"

    return "successful"


def extract_outcome_parameters(
    fit: xr.Dataset,
    qubit: str,
    raw_ds: xr.Dataset,
    frequency_span_in_mhz: float,
    qubit_outcomes: Dict[str, str],
) -> Dict[str, float]:
    """Extract all parameters needed for outcome determination."""
    flux_idle_val = (
        float(fit.idle_offset.sel(qubit=qubit).data)
        if not np.isnan(fit.idle_offset.sel(qubit=qubit).data)
        else np.nan
    )
    flux_min_val = (
        float(fit.flux_min.sel(qubit=qubit).data)
        if not np.isnan(fit.flux_min.sel(qubit=qubit).data)
        else np.nan
    )
    freq_shift_val = (
        float(fit.freq_shift.sel(qubit=qubit).values)
        if not np.isnan(fit.freq_shift.sel(qubit=qubit).values)
        else np.nan
    )

    # Calculate SNR
    snr = calculate_snr_from_dataset(raw_ds, qubit, epsilon=sp_params.epsilon.value)

    # Check for oscillations
    has_oscillations = has_oscillations_in_fit(fit, qubit, qubit_outcomes)

    return {
        "freq_shift": freq_shift_val,
        "flux_min": flux_min_val,
        "flux_idle": flux_idle_val,
        "frequency_span_in_mhz": frequency_span_in_mhz,
        "snr": snr,
        "has_oscillations": has_oscillations,
        "has_anticrossings": False,  # Not currently detected
    }


def correct_resonator_frequency(
    ds_raw: xr.Dataset, qubit: str, ds_fit: xr.Dataset
) -> float:
    """Compute corrected resonator frequency using |IQ| minimum at fitted idle flux."""
    try:
        # Extract data
        IQ_abs = ds_raw["IQ_abs"].sel(qubit=qubit)
        flux_vals = ds_raw["flux_bias"].values
        freq_vals = ds_raw["full_freq"].sel(qubit=qubit).values / 1e9  # GHz

        # Get idle flux from fit
        idle_flux = float(ds_fit.sel(qubit=qubit)["idle_offset"].values)

        # Find nearest flux point and get minimum
        nearest_flux = flux_vals[np.argmin(np.abs(flux_vals - idle_flux))]
        trace = ds_raw["IQ_abs"].sel(qubit=qubit, flux_bias=nearest_flux)
        smoothed = gaussian_filter1d(trace.values, sigma=2)
        min_idx = np.argmin(smoothed)
        corrected_freq = freq_vals[min_idx] * 1e9  # Hz

        return corrected_freq
    except:
        # Fallback to original value
        return float(ds_fit.sel(qubit=qubit)["sweet_spot_frequency"].values)


def apply_frequency_correction_if_needed(
    fit: xr.Dataset,
    qubit: str,
    base_freq: float,
    raw_ds: xr.Dataset,
    outcome_params: Dict[str, float],
) -> float:
    """Apply frequency correction if confidence is low."""
    # Calculate confidence metric
    fitted_flux_idle = outcome_params["flux_idle"]
    peak_freq_vs_flux = fit.peak_freq.sel(qubit=qubit)
    flux_biases = peak_freq_vs_flux.coords["flux_bias"].values
    freq_vals = peak_freq_vs_flux.values

    # Check confidence around idle flux
    idx_idle = np.argmin(np.abs(flux_biases - fitted_flux_idle))
    freq_at_idle = freq_vals[idx_idle]
    window = qc_params.resonance_correction_window.value
    start = max(0, idx_idle - window)
    end = min(len(freq_vals), idx_idle + window + 1)
    local_min = np.min(freq_vals[start:end])
    local_mean = np.mean(freq_vals[start:end])
    global_min = np.min(freq_vals)
    global_max = np.max(freq_vals)
    confidence = (local_mean - freq_at_idle) / (global_max - global_min + 1e-12)

    # Correct frequency if confidence is low
    if confidence < qc_params.resonance_correction_confidence_threshold.value:
        corrected_freq = correct_resonator_frequency(
            ds_raw=raw_ds,
            qubit=qubit,
            ds_fit=fit,
        )
        # Update the fit dataset
        fit["sweet_spot_frequency"].loc[dict(qubit=qubit)] = corrected_freq
        fit["freq_shift"].loc[dict(qubit=qubit)] = corrected_freq - base_freq
        return corrected_freq
    else:
        return outcome_params["freq_shift"] + base_freq


def create_flux_fit_parameters(
    fit: xr.Dataset,
    qubit: str,
    node_params,
    qubits: list,
    outcome: str,
    corrected_frequency: float,
    fit_params_class: Type,
) -> any:
    """Create FitParameters object for a qubit."""
    # Get attenuation factor
    attenuation_factor = 10 ** (-node_params.line_attenuation_in_db / 20)

    # Calculate m_pH
    frequency_factor = fit.sel(fit_vals="f", qubit=qubit).fit_results.data
    m_pH = (
        1e12
        * 2.068e-15
        / (1 / frequency_factor)
        / node_params.input_line_impedance_in_ohm
        * attenuation_factor
    )

    # Get base frequency
    full_freq = np.array([q.resonator.RF_frequency for q in qubits])
    base_freq = full_freq[[q.name for q in qubits].index(qubit)]

    return fit_params_class(
        resonator_frequency=corrected_frequency
        if outcome == "successful"
        else np.nan,
        frequency_shift=corrected_frequency - base_freq
        if outcome == "successful"
        else float(fit.sel(qubit=qubit).freq_shift.values),
        min_offset=float(fit.sel(qubit=qubit).flux_min.data),
        idle_offset=float(fit.sel(qubit=qubit).idle_offset.data),
        dv_phi0=1 / frequency_factor if outcome == "successful" else np.nan,
        phi0_current=(
            1
            / frequency_factor
            * node_params.input_line_impedance_in_ohm
            * attenuation_factor
            if outcome == "successful"
            else np.nan
        ),
        m_pH=m_pH if outcome == "successful" else np.nan,
        outcome=outcome,
    ) 