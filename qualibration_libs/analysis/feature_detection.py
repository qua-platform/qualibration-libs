import numpy as np
from scipy.fft import fft
import scipy.sparse as sparse
import xarray as xr
from scipy.signal import find_peaks, peak_widths
from scipy.sparse.linalg import spsolve


def peaks_dips(
    da, dim, prominence_factor=5, number=1, remove_baseline=True
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

    return xr.merge(
        [
            peak_position.rename("position"),
            peak_width.rename("width"),
            peak_amp.rename("amplitude"),
            base_line.rename("base_line"),
        ]
    )


def extract_dominant_frequencies(da, dim="idle_time"):
    def extract_dominant_frequency(signal, sample_rate):
        fft_result = fft(signal)
        frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
        positive_freq_idx = np.where(frequencies > 0)
        dominant_idx = np.argmax(np.abs(fft_result[positive_freq_idx]))
        return frequencies[positive_freq_idx][dominant_idx]

    def extract_dominant_frequency_wrapper(signal):
        sample_rate = 1 / (
            da.coords[dim][1].values - da.coords[dim][0].values
        )  # Assuming uniform sampling
        return extract_dominant_frequency(signal, sample_rate)

    dominant_frequencies = xr.apply_ufunc(
        extract_dominant_frequency_wrapper,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
    )

    return dominant_frequencies
