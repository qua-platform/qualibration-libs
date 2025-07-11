from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sparse
import xarray as xr
from scipy.fft import fft
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import skew

from .parameters import analysis_config_manager

__all__ = [
    "peaks_dips",
    "find_all_peaks_from_signal",
    "estimate_width_from_curvature",
    "is_peak_shape_distorted",
    "is_peak_too_wide",
    "detect_chevron_modulation",
    "has_resonator_trace",
    "has_insufficient_flux_modulation",
]

# Access signal processing parameters from the config object
sp_params = analysis_config_manager.get("common_signal_processing")


def _preprocess_signal_for_peak_detection(
    arr: np.ndarray, apply_baseline_correction: bool = True
) -> Dict[str, np.ndarray]:
    """
    Apply consistent preprocessing to signal data for peak detection.
    
    This function performs baseline correction using ALS, applies smoothing with
    Savitzky-Golay filter, and estimates noise from the residual. All functions
    use identical preprocessing parameters to ensure consistency.
    
    Args:
        arr: Input signal array to preprocess
        apply_baseline_correction: Whether to apply baseline correction
        
    Returns:
        Dictionary containing:
        - 'smoothed': Baseline-corrected and smoothed signal
        - 'noise': Estimated noise level from residual
        - 'baseline': Original baseline estimate
        - 'baseline_corrected': Signal with baseline removed
    """
    # Baseline correction (ALS)
    if apply_baseline_correction:
        baseline_window = _get_savgol_window_length(
            len(arr), sp_params.baseline_window_size.value, sp_params.polyorder.value
        )
        baseline = (savgol_filter(arr, window_length=baseline_window, polyorder=sp_params.polyorder.value) 
                    if baseline_window > 0 else arr)
        arr_bc = arr - baseline
    else:
        baseline = np.zeros_like(arr)
        arr_bc = arr
    
    # Smoothing
    smooth_window = _get_savgol_window_length(
        len(arr_bc), sp_params.smooth_window_size.value, sp_params.polyorder.value
    )
    smoothed = (savgol_filter(arr_bc, window_length=smooth_window, polyorder=sp_params.polyorder.value) 
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
    
    Args:
        signal: Signal to analyze for peaks
        prominence: Minimum prominence threshold for peak detection
        
    Returns:
        Peaks and properties from scipy.signal.find_peaks
    """
    return find_peaks(
        signal,
        prominence=prominence,
        distance=sp_params.peak_distance.value,
        width=sp_params.peak_width.value
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


def find_all_peaks_from_signal(signal: np.ndarray, prominence_factor: float = 2.0):
    """
    Finds all peaks in a signal using consistent preprocessing and returns them.

    Args:
        signal (np.ndarray): The input signal array.
        prominence_factor (float): Factor to determine peak prominence relative to noise.

    Returns:
        tuple: A tuple containing:
            - peaks (np.ndarray): Indices of the detected peaks.
            - properties (dict): Dictionary of peak properties from `scipy.signal.find_peaks`.
            - smoothed (np.ndarray): The smoothed signal after baseline correction.
            - baseline (np.ndarray): The detected baseline of the signal.
    """
    processed = _preprocess_signal_for_peak_detection(signal)
    prominence = prominence_factor * processed["noise"]
    peaks, properties = _detect_peaks_with_consistent_parameters(
        processed["smoothed"], prominence
    )
    return peaks, properties, processed["smoothed"], processed["baseline"]


def estimate_width_from_curvature(signal: np.ndarray, peak_idx: int) -> float:
    """
    Estimates the FWHM of a peak from its curvature (2nd derivative).
    
    This method finds the inflection points of a peak by finding the minima
    of its second derivative on either side of the peak's center. The distance
    between these inflection points gives an estimate of the peak's width.
    
    Args:
        signal: The input signal array containing the peak.
        peak_idx: The index of the peak's maximum in the signal array.
        
    Returns:
        The estimated width of the peak in samples (float). Returns np.nan if
        the width cannot be determined.
    """
    if peak_idx <= 0 or peak_idx >= len(signal) - 1:
        return np.nan
        
    curvature = np.gradient(np.gradient(signal))
    
    # For an inverted dip (which is a peak), inflection points are where curvature is at a minimum.
    left_curvature = curvature[:peak_idx]
    right_curvature = curvature[peak_idx + 1:]

    if len(left_curvature) == 0 or len(right_curvature) == 0:
        return np.nan

    left_infl_idx = np.argmin(left_curvature)
    right_infl_idx = np.argmin(right_curvature) + peak_idx + 1
    
    width_in_samples = right_infl_idx - left_infl_idx
    return float(width_in_samples)


def is_peak_shape_distorted(
    asymmetry: float,
    skewness: float,
    nrmse: float,
    min_asymmetry: float,
    max_asymmetry: float,
    max_skewness: float,
    nrmse_threshold: float,
) -> bool:
    """
    Check if peak shape indicates distortion based on asymmetry and skewness.

    Args:
        asymmetry : float
            Peak asymmetry value
        skewness : float
            Peak skewness value
        nrmse : float
            Normalized Root Mean Square Error
        min_asymmetry : float
            Minimum allowed asymmetry
        max_asymmetry : float
            Maximum allowed asymmetry
        max_skewness : float
            Maximum allowed skewness
        nrmse_threshold : float
            NRMSE threshold for a good fit

    Returns:
        bool
            True if peak shape is distorted
    """
    asymmetry_bad = (
        asymmetry is not None and (asymmetry < min_asymmetry or asymmetry > max_asymmetry)
    )
    skewness_bad = skewness is not None and abs(skewness) > max_skewness
    poor_geom_fit = nrmse > nrmse_threshold

    return asymmetry_bad or skewness_bad or poor_geom_fit


def is_peak_too_wide(
    fwhm: float,
    sweep_span: float,
    snr: float,
    distorted_fraction_low_snr: float,
    snr_for_distortion: float,
    distorted_fraction_high_snr: float,
    fwhm_absolute_threshold_hz: float,
) -> bool:
    """
    Check if peak is too wide relative to sweep or absolutely.

    Args:
        fwhm : float
            Full width at half maximum
        sweep_span : float
            Total sweep span
        snr : float
            Signal-to-noise ratio
        distorted_fraction_low_snr : float
            Distorted fraction for low SNR
        snr_for_distortion : float
            SNR threshold to determine which distorted fraction to use
        distorted_fraction_high_snr : float
            Distorted fraction for high SNR
        fwhm_absolute_threshold_hz : float
            Absolute FWHM threshold in Hz

    Returns:
        bool
            True if peak is too wide
    """
    # Determine distortion threshold based on SNR
    distorted_fraction = (
        distorted_fraction_low_snr
        if snr < snr_for_distortion
        else distorted_fraction_high_snr
    )

    # Check relative width
    relative_width_bad = sweep_span > 0 and (fwhm / sweep_span > distorted_fraction)

    # Check absolute width
    absolute_width_bad = fwhm > fwhm_absolute_threshold_hz

    return relative_width_bad or absolute_width_bad


def detect_chevron_modulation(
    signal_2d: np.ndarray, threshold: float
) -> bool:
    """
    Detect chevron-like modulation in 2D signal data.

    Args:
        signal_2d : np.ndarray
            2D array of signal values
        threshold : float
            Minimum modulation threshold

    Returns:
        bool
            True if chevron modulation detected
    """
    if signal_2d.ndim != 2:
        return False

    # Calculate peak-to-peak amplitude across frequency axis
    ptps = np.ptp(signal_2d, axis=0)
    modulation_range = np.ptp(ptps)
    avg_ptp = np.mean(ptps)

    # Check if modulation is significant
    return (modulation_range / (avg_ptp + 1e-12)) > threshold


def has_resonator_trace(
    ds: xr.Dataset,
    qubit: str,
    smooth_sigma: float,
    dip_threshold: float,
    gradient_threshold: float,
    var_name: str = "IQ_abs",
    freq_dim: str = "detuning",
    epsilon: float = 1e-9,
) -> bool:
    """Detect whether a resonator-like frequency trace exists."""
    try:
        # Extract and normalize data
        da = ds[var_name].sel(qubit=qubit)
        da_norm = da / (da.mean(dim=freq_dim) + epsilon)

        # Get minimum trace across frequency sweep
        min_trace = da_norm.min(dim=freq_dim).values

        # Smooth to suppress noise
        min_trace_smooth = savgol_filter(min_trace, int(smooth_sigma * 5) | 1, 2)

        # Calculate peak-to-peak dip depth and variation
        dip_depth = np.max(min_trace_smooth) - np.min(min_trace_smooth)
        gradient = np.max(np.abs(np.gradient(min_trace_smooth)))

        return (dip_depth > dip_threshold) and (gradient > gradient_threshold)
    except:
        return False


def has_insufficient_flux_modulation(
    ds: xr.Dataset,
    qubit: str,
    smooth_sigma: float,
    min_modulation_hz: float,
) -> bool:
    """Detect whether the flux modulation of the resonator is too small."""
    try:
        # Calculate peak frequency from IQ_abs minimum
        peak_idx = ds.IQ_abs.argmin(dim="detuning")
        peak_freq = ds.full_freq.isel(detuning=peak_idx)
        peak_freq = peak_freq.transpose("qubit", "flux_bias")

        pf = peak_freq.sel(qubit=qubit)
        if pf.isnull().all():
            return True

        freq_vals = pf.values
        freq_vals = freq_vals[~np.isnan(freq_vals)]

        if freq_vals.size < 2:
            return True

        smoothed = savgol_filter(freq_vals, int(smooth_sigma * 5) | 1, 2)
        modulation_range = np.max(smoothed) - np.min(smoothed)

        return modulation_range < min_modulation_hz
    except:
        return True


def peaks_dips(
    da, dim, prominence_factor=2, number=1, remove_baseline=True
) -> xr.Dataset:
    """
    Searches in a data array along the specified dimension for the most prominent peak or dip, and returns a xarray
    dataset with its location, width, and amplitude, along with a smooth baseline from which the peak emerges.

    Args:
        da: The input xarray DataArray.
        dim: The dimension along which to perform the fit.
        prominence_factor: How prominent the peak must be compared with noise, as defined by the standard deviation.
        number: Determines which peak the function returns. 1 is the most prominent peak, 2 is the second most prominent, etc.
        remove_baseline: If True, the function will remove the baseline from the data before finding the peak (default is False).

    Returns:
        A dataset with the following values:
        - 'amp': Peak amplitude above the base.
        - 'position': Peak location along 'dim'.
        - 'width': Peak full width at half maximum (FWHM).
        - 'baseline': A vector whose dimension is the same as 'dim'. It is the baseline from which the peak is found.
          This is important for fitting resonator spectroscopy measurements.

    Notes:
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

    def _num_peaks(
        arr: np.ndarray, prominence: float, use_smoothed: bool = True
    ) -> np.ndarray:
        """Count the number of peaks in the signal using consistent preprocessing."""
        processed = _preprocess_signal_for_peak_detection(
            arr, apply_baseline_correction=False
        )
        signal_to_check = processed["smoothed"] if use_smoothed else arr
        peaks, _ = _detect_peaks_with_consistent_parameters(
            signal_to_check,
            sp_params.noise_prominence_factor.value * processed["noise"],
        )
        return np.array([len(peaks)])

    def _raw_num_peaks(arr: np.ndarray, prominence: float) -> np.ndarray:
            """Count the number of peaks in the signal using consistent preprocessing."""
            processed = _preprocess_signal_for_peak_detection(arr)
            peaks, _ = _detect_peaks_with_consistent_parameters(
                processed['smoothed'], 
                sp_params.noise_prominence_factor.value * processed['noise']
            )
            return np.array([len(peaks)])

    def _main_peak_snr(arr: np.ndarray, prominence: float) -> np.ndarray:
        """Calculate signal-to-noise ratio of the main peak."""
        processed = _preprocess_signal_for_peak_detection(
            arr, apply_baseline_correction=False
        )
        peaks, _ = _detect_peaks_with_consistent_parameters(
            processed['smoothed'], 
            sp_params.noise_prominence_factor.value * processed['noise']
        )
        
        if len(peaks) == 0:
            return np.array([0.0])
            
        heights = processed['smoothed'][peaks]
        # Main peak: largest amplitude change (height)
        main_idx = np.argmax(np.abs(heights))
        main_height = np.abs(heights[main_idx])
        snr = main_height / (processed['noise'] if processed['noise'] > 0 else sp_params.noise_floor.value)
        return np.array([snr])

    def _main_peak_asymmetry_skew(arr: np.ndarray, prominence: float) -> np.ndarray:
        """Calculate asymmetry and skewness of the main peak."""
        processed = _preprocess_signal_for_peak_detection(
            arr, apply_baseline_correction=False
        )
        peaks, _ = _detect_peaks_with_consistent_parameters(
            processed['smoothed'], 
            sp_params.noise_prominence_factor.value * processed['noise']
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
        window_radius = int(max(left_width, right_width, sp_params.min_window_radius.value))
        start = max(0, int(main_peak - window_radius))
        end = min(len(processed['smoothed']), int(main_peak + window_radius + 1))
        window = processed['smoothed'][start:end]
        skewness = skew(window) if len(window) > sp_params.min_skew_window_size.value else np.nan
        return np.array([asymmetry_ratio, skewness])

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
        kwargs={"use_smoothed": True},
        output_core_dims=[[]],
        vectorize=True,
    )
    raw_num_peaks = xr.apply_ufunc(
        _raw_num_peaks,
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

    return xr.merge(
        [
            peak_position.rename("position"),
            peak_width.rename("width"),
            peak_amp.rename("amplitude"),
            base_line.rename("base_line"),
            num_peaks.rename("num_peaks"),
            raw_num_peaks.rename("raw_num_peaks"),
            snr.rename("snr"),
            xr.DataArray(asymmetry_skew[..., 0], dims=peak_position.dims).rename("asymmetry"),
            xr.DataArray(asymmetry_skew[..., 1], dims=peak_position.dims).rename("skewness"),
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
