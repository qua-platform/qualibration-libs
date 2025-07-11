from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Type

import numpy as np
import xarray as xr
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.analysis.feature_detection import \
    detect_chevron_modulation
from qualibration_libs.analysis.fitting import extract_fit_quality
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.analysis.processing import extract_qubit_signal
from qualibration_libs.analysis.signal import (compute_signal_to_noise_ratio,
                                               evaluate_signal_fit_worthiness)

from qualibrate import QualibrationNode

# Access parameter groups from the config manager
qc_params = analysis_config_manager.get("power_rabi_qc")


def fit_single_pulse_experiment(
    ds: xr.Dataset, use_state_discrimination: bool
) -> xr.Dataset:
    """
    Fit single pulse power Rabi experiment data.

    Args:
        ds : xr.Dataset
            Raw dataset
        use_state_discrimination : bool
            Whether state discrimination is used

    Returns:
        xr.Dataset
            Dataset with oscillation fit results
    """
    ds_fit = ds.sel(nb_of_pulses=1)

    # Choose signal based on discrimination method
    signal_data = ds_fit.state if use_state_discrimination else ds_fit.I
    fit_vals = fit_oscillation(da=signal_data, dim="amp_prefactor", method="fft_based")

    return xr.merge([ds, fit_vals.rename("fit")])


def fit_multi_pulse_experiment(
    ds: xr.Dataset, use_state_discrimination: bool, operation: str
) -> xr.Dataset:
    """
    Fit multi-pulse power Rabi experiment data by finding optimal amplitude.

    Args:
        ds : xr.Dataset
            Raw dataset
        use_state_discrimination : bool
            Whether state discrimination is used
        operation : str
            The operation being performed (e.g., "x180")

    Returns:
        xr.Dataset
            Dataset with optimal amplitude prefactor
    """
    ds_fit = ds.copy()

    # Calculate mean across pulse dimension
    signal_data = ds.state if use_state_discrimination else ds.I
    ds_fit["data_mean"] = signal_data.mean(dim="nb_of_pulses")

    # Determine optimization direction based on pulse parity and operation
    first_pulse_even = ds.nb_of_pulses.data[0] % 2 == 0
    is_x180_operation = operation == "x180"

    should_minimize = (first_pulse_even and is_x180_operation) or (
        not first_pulse_even and not is_x180_operation
    )

    if should_minimize:
        ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
    else:
        ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")

    return ds_fit


def calculate_single_pulse_parameters(
    fit: xr.Dataset, operation: str, qubits: list
) -> xr.Dataset:
    """Calculate optimal amplitude parameters for single pulse experiments."""
    # Extract phase and correct for wrapping
    phase = fit.fit.sel(fit_vals="phi") - np.pi * (
        fit.fit.sel(fit_vals="phi") > np.pi / 2
    )

    # Calculate amplitude factor for π pulse
    factor = (np.pi - phase) / (2 * np.pi * fit.fit.sel(fit_vals="f"))
    fit = fit.assign({"opt_amp_prefactor": factor})
    fit.opt_amp_prefactor.attrs = {
        "long_name": "factor to get a pi pulse",
        "units": "Hz",
    }

    # Calculate optimal amplitude
    current_amps = xr.DataArray(
        [q.xy.operations[operation].amplitude for q in qubits],
        coords=dict(qubit=fit.qubit.data),
    )
    opt_amp = factor * current_amps
    fit = fit.assign({"opt_amp": opt_amp})
    fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}

    return fit


def calculate_multi_pulse_parameters(
    fit: xr.Dataset, operation: str, qubits: list
) -> xr.Dataset:
    """Calculate optimal amplitude parameters for multi-pulse experiments."""
    current_amps = xr.DataArray(
        [q.xy.operations[operation].amplitude for q in qubits],
        coords=dict(qubit=fit.qubit.data),
    )
    fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
    fit.opt_amp.attrs = {
        "long_name": f"{operation} pulse amplitude",
        "units": "V",
    }
    return fit


def assess_parameter_validity(fit: xr.Dataset) -> xr.Dataset:
    """Check for NaN values in fit parameters."""
    nan_success = ~(np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp))
    return fit.assign_coords(nan_success=("qubit", nan_success.data))


def calculate_signal_quality_metrics(fit: xr.Dataset) -> xr.Dataset:
    """Calculate signal-to-noise ratio for each qubit."""
    snrs = []
    for q in fit.qubit.values:
        # Choose appropriate signal data
        if hasattr(fit, "I") and hasattr(fit.I, "sel"):
            signal = fit.I.sel(qubit=q).values
        elif hasattr(fit, "state") and hasattr(fit.state, "sel"):
            signal = fit.state.sel(qubit=q).values
        else:
            logging.warning(
                f"Could not find 'I' or 'state' data for qubit {q}. Returning zeros."
            )
            signal = np.zeros(2)

        snrs.append(compute_signal_to_noise_ratio(signal))

    fit = fit.assign_coords(snr=("qubit", snrs))
    fit.snr.attrs = {"long_name": "signal-to-noise ratio", "units": ""}
    return fit


def evaluate_amplitude_constraints(fit: xr.Dataset, limits: list) -> xr.Dataset:
    """Check if calculated amplitudes are within hardware limits."""
    # Use first limit as they should be consistent across qubits
    max_amp = limits[0].max_x180_wf_amplitude
    amp_success = fit.opt_amp < max_amp
    return fit.assign_coords(amp_success=("qubit", amp_success.data))


def determine_fit_outcomes(
    fit: xr.Dataset, max_number_pulses_per_sweep: int, limits: list, node: QualibrationNode) -> xr.Dataset:
    """Determine analysis outcomes for all qubits based on quality checks."""
    # Check if signals are suitable for fitting
    signal_key = "state" if hasattr(fit, "state") else "I"
    fit_flags = evaluate_signal_fit_worthiness(
        fit,
        signal_key,
        qc_params.autocorrelation_r1_threshold.value,
        qc_params.autocorrelation_r2_threshold.value,
    )

    outcomes = []
    for i, q in enumerate(fit.qubit.values):
        # Extract fit quality if available
        fit_quality = extract_fit_quality(fit)

        # Check for chevron modulation in 2D case
        has_structure = True
        if max_number_pulses_per_sweep > 1:
            signal = extract_qubit_signal(fit, q)
            has_structure = detect_chevron_modulation(
                signal, qc_params.chevron_modulation_threshold.value
            )

        outcome = determine_qubit_outcome(
            nan_success=bool(fit.nan_success.sel(qubit=q).values),
            amp_success=bool(fit.amp_success.sel(qubit=q).values),
            opt_amp=float(fit.sel(qubit=q).opt_amp.values),
            max_amplitude=float(limits[i].max_x180_wf_amplitude),
            fit_quality=fit_quality,
            amp_prefactor=float(fit.sel(qubit=q).opt_amp_prefactor.values),
            snr=float(fit.sel(qubit=q).snr.values),
            should_fit=fit_flags[q],
            is_1d_dataset=max_number_pulses_per_sweep == 1,
            has_structure=has_structure,
            fit=fit,
            qubit=q,
            node=node,
        )
        outcomes.append(outcome)

    fit = fit.assign_coords(outcome=("qubit", outcomes))
    fit.outcome.attrs = {"long_name": "fit outcome", "units": ""}
    return fit


def create_fit_results_dictionary(
    fit: xr.Dataset, operation: str, fit_params_class: Type
) -> Dict[str, any]:
    """Create the final fit results dictionary."""
    fit_quality = extract_fit_quality(fit)

    return {
        q: fit_params_class(
            opt_amp_prefactor=fit.sel(qubit=q).opt_amp_prefactor.values.item(),
            opt_amp=fit.sel(qubit=q).opt_amp.values.item(),
            operation=operation,
            outcome=str(fit.sel(qubit=q).outcome.values),
            fit_quality=fit_quality,
        )
        for q in fit.qubit.values
    }


def determine_detuning_direction(
    fit: xr.Dataset, qubit: str, node: QualibrationNode = None) -> str:
    """
    Determine detuning direction from frequency information.

    Args:
        fit : xr.Dataset
            Fit dataset
        qubit : str
            Qubit identifier
        node : QualibrationNode, optional
            Experiment node

    Returns:
        str
            Detuning direction ('positive' or 'negative')
    """
    try:
        # Try to get frequencies from node
        if node and hasattr(node, "namespace") and "qubits" in node.namespace:
            qubits_dict = node.namespace["qubits"]
            for qubit_obj in (
                qubits_dict.values()
                if isinstance(qubits_dict, dict)
                else qubits_dict
            ):
                if hasattr(qubit_obj, "id") and getattr(qubit_obj, "id", None) == qubit:
                    drive_freq = getattr(qubit_obj.xy, "RF_frequency", None)
                    qubit_freq = getattr(qubit_obj, "f_01", None)
                    if drive_freq is not None and qubit_freq is not None:
                        detuning = drive_freq - qubit_freq
                        return "positive" if detuning > 0 else "negative"

        # Fallback: analyze fit parameters
        if "fit" in fit:
            fit_params = fit.fit.sel(qubit=qubit).values
            fit_vals = fit.fit.sel(qubit=qubit, fit_vals=slice(None)).coords[
                "fit_vals"
            ].values
            fit_dict = {k: v for k, v in zip(fit_vals, fit_params)}

            phase = float(fit_dict.get("phi", np.nan))
            freq = float(fit_dict.get("f", np.nan))

            if not (np.isnan(phase) or np.isnan(freq)):
                phase_norm = (phase + np.pi) % (2 * np.pi) - np.pi
                if abs(phase_norm) < (np.pi / 2):
                    return "positive" if freq > 0 else "negative"
                else:
                    return "positive" if phase_norm < 0 else "negative"

    except (AttributeError, KeyError, ValueError):
        pass

    return ""


def determine_qubit_outcome(
    nan_success: bool,
    amp_success: bool,
    opt_amp: float,
    max_amplitude: float,
    fit_quality: Optional[float] = None,
    amp_prefactor: Optional[float] = None,
    snr: Optional[float] = None,
    should_fit: bool = True,
    is_1d_dataset: bool = True,
    has_structure: bool = True,
    fit: Optional[xr.Dataset] = None,
    qubit: Optional[str] = None,
    node: Optional[QualibrationNode] = None,
) -> str:
    """
    Determine the outcome for a single qubit based on comprehensive quality checks.

    Args:
        nan_success : bool
            Whether fit parameters are free of NaN values
        amp_success : bool
            Whether amplitude is within hardware limits
        opt_amp : float
            Optimal amplitude value
        max_amplitude : float
            Maximum allowed amplitude
        fit_quality : float, optional
            R² value of the fit
        amp_prefactor : float, optional
            Amplitude prefactor value
        snr : float, optional
            Signal-to-noise ratio
        should_fit : bool
            Whether data quality is sufficient for fitting
        is_1d_dataset : bool
            Whether this is a 1D dataset
        has_structure : bool
            Whether data shows expected structure
        fit : xr.Dataset, optional
            Fit dataset for additional checks
        qubit : str, optional
            Qubit identifier
        node : QualibrationNode, optional
            Analysis node

    Returns:
        str
            Outcome description
    """

    # Check for invalid fit parameters
    if not nan_success:
        return "Fit parameters are invalid (NaN values detected)"

    # Check for low amplitude with poor fit quality (1D only)
    if is_1d_dataset and opt_amp < qc_params.min_amplitude.value:
        if fit_quality is not None and fit_quality < qc_params.min_fit_quality.value:
            return (
                "There is too much noise in the data, consider increasing averaging or shot count"
            )
        else:
            return "There is too much noise in the data, consider increasing averaging or shot count"

    # Check for missing structure in 2D datasets
    if not has_structure and not is_1d_dataset:
        return "No chevron modulation detected. Please check the drive frequency and amplitude sweep range"

    # Check signal-to-noise ratio
    if snr is not None and snr < qc_params.snr_min.value:
        return f"SNR too low (SNR = {snr:.2f} < {qc_params.snr_min.value})"

    # Check amplitude hardware limits
    if not amp_success:
        detuning_direction = ""
        if fit is not None and qubit is not None:
            detuning_direction = determine_detuning_direction(fit, qubit, node)

        direction_str = f"{detuning_direction} detuning" if detuning_direction else ""
        return f"The drive frequency is off-resonant with high {direction_str}, please adjust the drive frequency"

    # Check amplitude prefactor bounds
    if (
        amp_prefactor is not None
        and abs(amp_prefactor) > qc_params.max_amp_prefactor.value
    ):
        return f"Amplitude prefactor too large (|{amp_prefactor:.2f}| > {qc_params.max_amp_prefactor.value})"

    # Check if signal should be fit (1D only)
    if not should_fit and is_1d_dataset:
        return "There is too much noise in the data, consider increasing averaging or shot count"

    # Check fit quality
    if fit_quality is not None and fit_quality < qc_params.min_fit_quality.value:
        # Special check for 1D datasets with too many oscillations
        if (
            is_1d_dataset
            and fit is not None
            and qubit is not None
            and "fit" in fit
            and hasattr(fit.fit, "sel")
        ):
            try:
                freq = float(fit.fit.sel(qubit=qubit, fit_vals="f").values)
                if freq > 1.0:
                    return "Fit quality poor due to large pulse amplitude range, consider a smaller range"
            except (ValueError, KeyError, AttributeError):
                pass

        return (
            f"Poor fit quality (R² = {fit_quality:.3f} < {qc_params.min_fit_quality.value})"
        )

    return "successful" 