
import logging
from typing import Dict

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate


def get_safe_savgol_window_length(data_length: int, window_size: int, polyorder: int) -> int:
    """Safely determine the window length for savgol_filter.

    Args:
        data_length: The length of the data array.
        window_size: The desired window size.
        polyorder: The polynomial order for the filter.

    Returns:
        A valid window length, or -1 if no valid length can be found.
    """
    # Ensure window_length is odd and smaller than data_length
    window_length = min(window_size, data_length)
    if window_length % 2 == 0:
        window_length -= 1

    # Ensure window_length is greater than polyorder
    if window_length <= polyorder:
        if data_length > polyorder:
            window_length = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2
            if window_length > data_length:
                return -1  # Not possible to find a valid window
        else:
            return -1  # Not possible to find a valid window

    return window_length


def calculate_snr_from_dataset(
    ds: xr.Dataset, qubit: str, signal_var: str = "IQ_abs", epsilon: float = 1e-9
) -> float:
    """Calculate signal-to-noise ratio for a qubit's spectroscopy data from a dataset."""
    try:
        data = ds[signal_var].sel(qubit=qubit)
        signal = np.ptp(data.values)
        noise = np.std(data.values)
        return signal / (noise + epsilon) if noise > 0 else np.inf
    except:
        return np.nan


def compute_signal_to_noise_ratio(signal: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio for a 1D signal array.

    Args:
        signal: 1D array of signal values

    Returns:
        Signal-to-noise ratio
    """
    if signal.size < 2:
        return 0.0

    try:
        # Smooth signal to estimate underlying trend
        smooth = uniform_filter1d(signal, size=max(3, signal.size // 10))
        # Estimate noise as residual
        noise = signal - smooth
        noise_std = np.std(noise)

        if noise_std == 0:
            return np.inf

        # Signal strength as peak-to-peak range
        signal_strength = np.ptp(signal)
        return signal_strength / noise_std

    except ImportError:
        # Fallback if scipy not available
        signal_std = np.std(signal)
        return np.ptp(signal) / signal_std if signal_std > 0 else np.inf


def evaluate_signal_fit_worthiness(
    ds: xr.Dataset,
    signal_key: str = "state",
    autocorrelation_r1_threshold: float = 0.1,
    autocorrelation_r2_threshold: float = -0.1,
) -> Dict[str, bool]:
    """
    Evaluate whether each qubit's signal is suitable for fitting based on autocorrelation.

    Args:
        ds : xr.Dataset
            Dataset containing signal data
        signal_key : str
            Key for signal data ("state" or "I")
        autocorrelation_r1_threshold : float
            Threshold for autocorrelation at lag-1
        autocorrelation_r2_threshold : float
            Threshold for autocorrelation at lag-2

    Returns:
        Dict[str, bool]
            Dictionary mapping qubit names to fit-worthiness
    """
    result = {}

    for q in ds.qubit.values:
        try:
            signal = ds[signal_key].sel(qubit=q).squeeze().values
            if signal.ndim == 2:
                signal = signal.mean(axis=0)

            # Check if signal has meaningful variation
            ptp = np.ptp(signal)
            if ptp == 0 or np.isnan(ptp):
                result[q] = False
                continue

            # Normalize signal
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

            # Calculate autocorrelation
            autocorr = correlate(signal_norm, signal_norm, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]  # positive lags only
            autocorr /= autocorr[0]  # normalize

            # Check autocorrelation at lag-1 and lag-2
            r1, r2 = autocorr[1], autocorr[2]
            result[q] = (
                r1 > autocorrelation_r1_threshold
                and r2 > autocorrelation_r2_threshold
            )

        except Exception:
            result[q] = False

    return result