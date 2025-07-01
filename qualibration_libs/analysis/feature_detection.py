from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sparse
import xarray as xr
from scipy.fft import fft
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import skew

__all__ = ["peaks_dips"]

# Constants for signal processing parameters
BASELINE_WINDOW_SIZE = 51
SMOOTH_WINDOW_SIZE = 21
POLYORDER = 3
NOISE_PROMINENCE_FACTOR = 3
PEAK_DISTANCE = 10
PEAK_WIDTH = 3
MIN_SKEW_WINDOW_SIZE = 3
MIN_WINDOW_RADIUS = 5
NOISE_FLOOR = 1e-12


def _preprocess_signal_for_peak_detection(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply consistent preprocessing to signal data for peak detection.
    
    This function performs baseline correction using ALS, applies smoothing with
    Savitzky-Golay filter, and estimates noise from the residual. All functions
    use identical preprocessing parameters to ensure consistency.
    
    Parameters
    ----------
    arr : np.ndarray
        Input signal array to preprocess
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'smoothed': Baseline-corrected and smoothed signal
        - 'noise': Estimated noise level from residual
        - 'baseline': Original baseline estimate
        - 'baseline_corrected': Signal with baseline removed
    """
    # Baseline correction (ALS)
    baseline_window = _get_savgol_window_length(len(arr), BASELINE_WINDOW_SIZE, POLYORDER)
    baseline = (savgol_filter(arr, window_length=baseline_window, polyorder=POLYORDER) 
                if baseline_window > 0 else arr)
    arr_bc = arr - baseline
    
    # Smoothing
    smooth_window = _get_savgol_window_length(len(arr_bc), SMOOTH_WINDOW_SIZE, POLYORDER)
    smoothed = (savgol_filter(arr_bc, window_length=smooth_window, polyorder=POLYORDER) 
                if smooth_window > 0 else arr_bc)
    
    # Noise estimation from residual
    noise = np.std(arr_bc - smoothed)
    
    return {
        'smoothed': smoothed,
        'noise': noise,
        'baseline': baseline,
        'baseline_corrected': arr_bc
    }


def _detect_peaks_with_consistent_parameters(signal: np.ndarray, prominence: float) -> Tuple[np.ndarray, Dict]:
    """
    Detect peaks using consistent parameters across all detection functions.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal to analyze for peaks
    prominence : float
        Minimum prominence threshold for peak detection
        
    Returns
    -------
    Tuple[np.ndarray, Dict]
        Peaks and properties from scipy.signal.find_peaks
    """
    return find_peaks(
        signal,
        prominence=prominence,
        distance=PEAK_DISTANCE,
        width=PEAK_WIDTH
    )


def _get_savgol_window_length(data_length: int, window_size: int, polyorder: int) -> int:
    """Safely determine the window length for savgol_filter."""
    # Ensure window_length is odd and smaller than data_length
    window_length = min(window_size, data_length)
    if window_length % 2 == 0:
        window_length -= 1
    # Ensure window_length is greater than polyorder
    if window_length <= polyorder:
        # If not possible, find the smallest valid window_length
        if data_length > polyorder:
            window_length = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2
            if window_length > data_length:
                return -1  # Not possible to find a valid window
        else:
            return -1  # Not possible to find a valid window
    return window_length


def peaks_dips(
    da, dim, prominence_factor=2, number=1, remove_baseline=True
) -> xr.Dataset:
    """
    Searches in a data array along the specified dimension for the most prominent peak or dip, and returns a xarray
    dataset with its location, width, and amplitude, along with a smooth baseline from which the peak emerges.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray.
    dim : str
        The dimension along which to perform the fit.
    prominence_factor : float
        How prominent the peak must be compared with noise, as defined by the standard deviation.
    number : int
        Determines which peak the function returns. 1 is the most prominent peak, 2 is the second most prominent, etc.
    remove_baseline : bool, optional
        If True, the function will remove the baseline from the data before finding the peak (default is False).

    Returns
    -------
    xr.Dataset
        A dataset with the following values:
        - 'amp': Peak amplitude above the base.
        - 'position': Peak location along 'dim'.
        - 'width': Peak full width at half maximum (FWHM).
        - 'baseline': A vector whose dimension is the same as 'dim'. It is the baseline from which the peak is found.
          This is important for fitting resonator spectroscopy measurements.

    Notes
    -----
    - The function identifies the most prominent peak or dip in the data array along the specified dimension.
    - The baseline is smoothed and subtracted if `remove_baseline` is True.
    """

    def _baseline_als(y, lam, p, niter=10):
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def _index_of_largest_peak(arr, prominence):
        peaks = find_peaks(arr.copy(), prominence=prominence)
        if len(peaks[0]) > 0:
            # finding the largest peak and it's width
            prom_peak_index = (
                1.0 * peaks[0][np.argsort(peaks[1]["prominences"])][-number]
            )
        else:
            prom_peak_index = np.nan
        return prom_peak_index

    def _position_from_index(x_axis_vals, position):
        res = []
        if not (np.isnan(position)):
            res.append(x_axis_vals[int(position)])
        else:
            res.append(np.nan)
        return np.array(res)

    def _width_from_index(da, position):
        res = []
        if not (np.isnan(position)):
            res.append(peak_widths(da.copy(), peaks=[int(position)])[0][0])
        else:
            res.append(np.nan)
        return np.array(res)

    def _num_peaks(arr: np.ndarray, prominence: float) -> np.ndarray:
        """Count the number of peaks in the signal using consistent preprocessing."""
        processed = _preprocess_signal_for_peak_detection(arr)
        peaks, _ = _detect_peaks_with_consistent_parameters(
            processed['smoothed'], 
            NOISE_PROMINENCE_FACTOR * processed['noise']
        )
        return np.array([len(peaks)])

    def _main_peak_snr(arr: np.ndarray, prominence: float) -> np.ndarray:
        """Calculate signal-to-noise ratio of the main peak."""
        processed = _preprocess_signal_for_peak_detection(arr)
        peaks, _ = _detect_peaks_with_consistent_parameters(
            processed['smoothed'], 
            NOISE_PROMINENCE_FACTOR * processed['noise']
        )
        
        if len(peaks) == 0:
            return np.array([0.0])
            
        heights = processed['smoothed'][peaks]
        # Main peak: largest amplitude change (height)
        main_idx = np.argmax(np.abs(heights))
        main_height = np.abs(heights[main_idx])
        snr = main_height / (processed['noise'] if processed['noise'] > 0 else NOISE_FLOOR)
        return np.array([snr])

    def _main_peak_asymmetry_skew(arr: np.ndarray, prominence: float) -> np.ndarray:
        """Calculate asymmetry and skewness of the main peak."""
        processed = _preprocess_signal_for_peak_detection(arr)
        peaks, _ = _detect_peaks_with_consistent_parameters(
            processed['smoothed'], 
            NOISE_PROMINENCE_FACTOR * processed['noise']
        )
        
        if len(peaks) == 0:
            return np.array([np.nan, np.nan])
            
        main_idx = np.argmax(np.abs(processed['smoothed'][peaks]))
        main_peak = peaks[main_idx]
        
        # Asymmetry: left/right width at half-max
        results_half = peak_widths(processed['smoothed'], peaks=[main_peak], rel_height=0.5)
        left_ips, right_ips = results_half[2][0], results_half[3][0]
        left_width = abs(main_peak - left_ips)
        right_width = abs(right_ips - main_peak)
        asymmetry_ratio = right_width / left_width if left_width > 0 else np.nan
        
        # Skewness: window around peak (±width)
        window_radius = int(max(left_width, right_width, MIN_WINDOW_RADIUS))
        start = max(0, int(main_peak - window_radius))
        end = min(len(processed['smoothed']), int(main_peak + window_radius + 1))
        window = processed['smoothed'][start:end]
        skewness = skew(window) if len(window) > MIN_SKEW_WINDOW_SIZE else np.nan
        return np.array([asymmetry_ratio, skewness])

    def _opx_bandwidth_artifact(
        arr: np.ndarray, 
        prominence: float, 
        window: int = 20, 
        exclusion: int = 3, 
        artifact_prominence_factor: float = 2.0
    ) -> np.ndarray:
        """Detect OPX bandwidth artifacts around the main dip."""
        processed = _preprocess_signal_for_peak_detection(arr)
        
        # Find main dip (minimum)
        main_idx = np.argmin(processed['smoothed'])
        
        # Define window around main dip, excluding center
        start = max(0, main_idx - window)
        end = min(len(processed['smoothed']), main_idx + window + 1)
        exclusion_start = max(start, main_idx - exclusion)
        exclusion_end = min(end, main_idx + exclusion + 1)
        
        # Search for local maxima (peaks) in the window, excluding the dip center
        search_region = np.concatenate([
            processed['smoothed'][start:exclusion_start],
            processed['smoothed'][exclusion_end:end]
        ])
        
        if len(search_region) == 0:
            return np.array([False])
            
        # Find peaks in the search region
        peaks, _ = find_peaks(search_region, prominence=artifact_prominence_factor * processed['noise'])
        return np.array([len(peaks) > 0])

    peaks_inversion = (
        2.0 * (da.mean(dim=dim) - da.min(dim=dim) < da.max(dim=dim) - da.mean(dim=dim))
        - 1
    )
    da = da * peaks_inversion

    base_line = xr.apply_ufunc(
        _baseline_als,
        da,
        1e8,
        0.001,
        input_core_dims=[[dim], [], []],
        output_core_dims=[[dim]],
        vectorize=True,
    )

    if remove_baseline:
        da = da - base_line

    base_line = base_line * peaks_inversion

    dim_step = da.coords[dim].diff(dim=dim).values[0]

    # Taking a rolling mean and subtracting it to estimate the noise of the signal
    rolling = da.rolling({dim: 10}, center=True).mean(dim=dim)
    std = float((da - rolling).std())

    prom_peak_index = xr.apply_ufunc(
        _index_of_largest_peak,
        da,
        prominence_factor * std,
        input_core_dims=[[dim], []],
        vectorize=True,
    )
    num_peaks = xr.apply_ufunc(
        _num_peaks,
        da,
        prominence_factor * std,
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        vectorize=True,
    )
    snr = xr.apply_ufunc(
        _main_peak_snr,
        da,
        prominence_factor * std,
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        vectorize=True,
    )
    peak_position = xr.apply_ufunc(
        _position_from_index,
        1.0 * da.coords[dim],
        prom_peak_index,
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        vectorize=True,
    )
    peak_width = (
        xr.apply_ufunc(
            _width_from_index,
            da,
            prom_peak_index,
            input_core_dims=[[dim], []],
            output_core_dims=[[]],
            vectorize=True,
        )
        * dim_step
    )
    peak_amp = da.max(dim=dim) - da.min(dim=dim) - std

    asymmetry_skew = xr.apply_ufunc(
        _main_peak_asymmetry_skew,
        da,
        prominence_factor * std,
        input_core_dims=[[dim], []],
        output_core_dims=[["asymmetry_skew"]],
        vectorize=True,
    )

    opx_bandwidth_artifact = xr.apply_ufunc(
        _opx_bandwidth_artifact,
        da,
        prominence_factor * std,
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        vectorize=True,
    )

    return xr.merge(
        [
            peak_position.rename("position"),
            peak_width.rename("width"),
            peak_amp.rename("amplitude"),
            base_line.rename("base_line"),
            num_peaks.rename("num_peaks"),
            snr.rename("snr"),
            xr.DataArray(asymmetry_skew[..., 0], dims=peak_position.dims).rename("asymmetry"),
            xr.DataArray(asymmetry_skew[..., 1], dims=peak_position.dims).rename("skewness"),
            opx_bandwidth_artifact.rename("opx_bandwidth_artifact"),
        ]
    )


# def extract_dominant_frequencies(da, dim="idle_time"):
#     def extract_dominant_frequency(signal, sample_rate):
#         fft_result = fft(signal)
#         frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
#         positive_freq_idx = np.where(frequencies > 0)
#         dominant_idx = np.argmax(np.abs(fft_result[positive_freq_idx]))
#         return frequencies[positive_freq_idx][dominant_idx]
#
#     def extract_dominant_frequency_wrapper(signal):
#         sample_rate = 1 / (
#             da.coords[dim][1].values - da.coords[dim][0].values
#         )  # Assuming uniform sampling
#         return extract_dominant_frequency(signal, sample_rate)
#
#     dominant_frequencies = xr.apply_ufunc(
#         extract_dominant_frequency_wrapper,
#         da,
#         input_core_dims=[[dim]],
#         output_core_dims=[[]],
#         vectorize=True,
#     )
#
#     return dominant_frequencies
