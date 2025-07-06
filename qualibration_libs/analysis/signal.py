
import numpy as np
from scipy.ndimage import uniform_filter1d


def get_safe_savgol_window_length(data_length: int, window_size: int, polyorder: int) -> int:
    """Safely determine the window length for savgol_filter.

    Parameters
    ----------
    data_length : int
        The length of the data array.
    window_size : int
        The desired window size.
    polyorder : int
        The polynomial order for the filter.

    Returns
    -------
    int
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


def compute_signal_to_noise_ratio(signal: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio for a 1D signal array.

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values

    Returns
    -------
    float
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