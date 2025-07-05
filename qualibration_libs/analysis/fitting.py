from typing import Dict

import numpy as np
import qiskit_experiments.curve_analysis as ca
import xarray as xr
from lmfit import Model, Parameter
from matplotlib import pyplot as plt
from qualibration_libs.analysis.models import *
from scipy.optimize import curve_fit

__all__ = ["fit_oscillation", "fit_oscillation_decay_exp", "fit_decay_exp"]


def _fix_initial_value(x, da):
    if len(da.dims) == 1:
        return float(x)
    else:
        return x


def fit_decay_exp(da, dim):
    def get_decay(dat):
        def oed(d):
            return ca.guess.exp_decay(da[dim], d)

        return np.apply_along_axis(oed, -1, dat)

    def get_amp(dat):
        max_ = np.max(dat, axis=-1)
        min_ = np.min(dat, axis=-1)
        return (max_ - min_) / 2

    def get_min(dat):
        return np.min(dat, axis=-1)

    decay_guess = xr.apply_ufunc(get_decay, da, input_core_dims=[[dim]]).rename(
        "decay guess"
    )
    amp_guess = xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess")
    min_guess = xr.apply_ufunc(get_min, da, input_core_dims=[[dim]]).rename("min guess")

    def apply_fit(x, y, a, offset, decay):
        try:
            # fit = curve_fit(decay_exp, x, y, p0=[a, offset, decay], bounds=(0, [1, 1., -1]))[0]
            fit, residuals = curve_fit(decay_exp, x, y, p0=[a, offset, decay])
            return np.array(fit.tolist() + np.array(residuals).flatten().tolist())
            # return np.array([fit.values[k] for k in ["a", "offset", "decay"]])
        except RuntimeError:
            print("Fit failed:")
            print(f"{a=}, {offset=}, {decay=}")
            plt.plot(x, decay_exp(x, a, offset, decay))
            plt.plot(x, y)
            plt.show()
            # raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        min_guess,
        decay_guess,
        input_core_dims=[[dim], [dim], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(
        fit_vals=(
            "fit_vals",
            [
                "a",
                "offset",
                "decay",
                "a_a",
                "a_offset",
                "a_decay",
                "offset_a",
                "offset_offset",
                "offset_decay",
                "decay_a",
                "decay_offset",
                "decay_decay",
            ],
        )
    )


def fit_oscillation_decay_exp(da, dim):
    def get_decay(dat):
        def oed(d):
            return ca.guess.oscillation_exp_decay(da[dim], d)

        return np.apply_along_axis(oed, -1, dat)

    def get_freq(dat):
        def f(d):
            return ca.guess.frequency(da[dim], d)

        return np.apply_along_axis(f, -1, dat)

    def get_amp(dat):
        max_ = np.max(dat, axis=-1)
        min_ = np.min(dat, axis=-1)
        return (max_ - min_) / 2

    decay_guess = xr.apply_ufunc(get_decay, da, input_core_dims=[[dim]]).rename(
        "decay guess"
    )
    freq_guess = xr.apply_ufunc(get_freq, da, input_core_dims=[[dim]]).rename(
        "freq guess"
    )
    amp_guess = xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess")

    def apply_fit(x, y, a, f, phi, offset, decay):
        try:
            fit, residuals = curve_fit(
                oscillation_decay_exp, x, y, p0=[a, f, phi, offset, decay]
            )
            return np.array(fit.tolist() + np.array(residuals).flatten().tolist())
        except RuntimeError:
            print(f"{a=}, {f=}, {phi=}, {offset=}, {decay=}")
            plt.plot(x, oscillation_decay_exp(x, a, f, phi, offset, decay))
            plt.plot(x, y)
            plt.show()
            # raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        freq_guess,
        0,
        0.5,
        decay_guess,
        input_core_dims=[[dim], [dim], [], [], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(
        fit_vals=(
            "fit_vals",
            [
                "a",
                "f",
                "phi",
                "offset",
                "decay",
                "a_a",
                "a_f",
                "a_phi",
                "a_offset",
                "a_decay",
                "f_a",
                "f_f",
                "f_phi",
                "f_offset",
                "f_decay",
                "phi_a",
                "phi_f",
                "phi_phi",
                "phi_offset",
                "phi_decay",
                "offset_a",
                "offset_f",
                "offset_phi",
                "offset_offset",
                "offset_decay",
                "decay_a",
                "decay_f",
                "decay_phi",
                "decay_offset",
                "decay_decay",
            ],
        )
    )

    # return a * np.exp(-t * decay) + offset


# def fit_echo_decay_exp(da, dim):
#     def get_decay(dat):
#         def oed(d):
#             return ca.guess.oscillation_exp_decay(da[dim], d)
#
#         return np.apply_along_axis(oed, -1, dat)
#
#     def get_amp(dat):
#         max_ = np.max(dat, axis=-1)
#         min_ = np.min(dat, axis=-1)
#         return (max_ - min_) / 2
#
#     decay_guess = xr.apply_ufunc(get_decay, da, input_core_dims=[[dim]]).rename(
#         "decay guess"
#     )
#     amp_guess = xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess")
#
#     def apply_fit(x, y, a, offset, decay, decay_echo):
#         try:
#             fit = curve_fit(echo_decay_exp, x, y, p0=[a, offset, decay, decay_echo])[0]
#             return fit
#         except RuntimeError:
#             print(f"{a=}, {offset=}, {decay=}, {decay_echo=}")
#             plt.plot(x, echo_decay_exp(x, a, offset, decay, decay_echo))
#             plt.plot(x, y)
#             plt.show()
#             # raise e
#
#     fit_res = xr.apply_ufunc(
#         apply_fit,
#         da[dim],
#         da,
#         amp_guess,
#         -0.0005,
#         decay_guess,
#         decay_guess,
#         input_core_dims=[[dim], [dim], [], [], [], []],
#         output_core_dims=[["fit_vals"]],
#         vectorize=True,
#     )
#     return fit_res.assign_coords(
#         fit_vals=("fit_vals", ["a", "offset", "decay", "decay_echo"])
#     )


def fit_oscillation(da, dim, method="qiskit_curve_analysis"):
    """
    Fits an oscillatory model to data along a specified dimension using selectable parameter estimation methods.
    This function estimates the frequency, amplitude, and phase of an oscillatory signal in the input
    data array `da` along the given dimension `dim` using either curve analysis or FFT-based initial
    parameter guesses. It then fits the data to an oscillatory model of the form:
        y(t) = a * cos(2π * f * t + phi) + offset
    using non-linear least squares optimization with retry mechanism for robustness.
    
    Parameters
    ----------
    da : xarray.DataArray
        The input data array containing the oscillatory signal to be fitted.
    dim : str
        The name of the dimension along which to perform the fit.
    method : str, optional
        Parameter estimation method to use. Options are:
        - "qiskit_curve_analysis": Uses qiskit curve analysis for parameter estimation (default)
        - "fft_based": Uses FFT-based parameter estimation
        
    Returns
    -------
    xarray.DataArray
        An array containing the fitted parameters for each slice along the specified dimension.
        The output has a new dimension 'fit_vals' with coordinates: ['a', 'f', 'phi', 'offset'],
        corresponding to amplitude, frequency, phase, and offset of the fitted oscillation.
        
    Notes
    -----
    - The function supports two parameter estimation methods: curve analysis and FFT-based
    - Includes retry mechanism for improved robustness when initial fit fails (qiskit_curve_analysis only)
    - The fitting is performed using a model function (oscillation) and the lmfit library
    - If the fit fails, diagnostic plots are shown for debugging
    """

    def get_freq_and_amp_and_phase(da, dim):
        """Parameter estimation - matches original new implementation structure"""
        if method == "fft_based":
            def compute_FFT(x, y):
                N = len(x)
                T = x[1] - x[0]
                yf = np.fft.fft(y)
                xf = np.fft.fftfreq(N, T)
                mask = xf > 0.1
                xf, fft_magnitude = xf[mask], np.abs(yf)[mask]
                idx = np.argmax(fft_magnitude)
                peak_freqs = xf
                peak_amps = 2 * fft_magnitude / N
                peak_phases = np.angle(yf[mask])
                return peak_freqs[idx], peak_amps[idx], peak_phases[idx]

            # Apply the FFT computation across the specified dimension
            def get_fft_param(dat, idx):
                return np.apply_along_axis(
                    lambda x: compute_FFT(da[dim].values, x)[idx], -1, dat
                )

            params = [
                xr.apply_ufunc(get_fft_param, da, i, input_core_dims=[[dim], []])
                for i in range(3)
            ]
            params = [_fix_initial_value(p, da) for p in params]
            return [
                p.rename(n)
                for p, n in zip(params, ["freq guess", "amp guess", "phase guess"])
            ]
        
        elif method == "qiskit_curve_analysis":
            def get_freq(dat):
                def f(d):
                    return ca.guess.frequency(da[dim], d)
                return np.apply_along_axis(f, -1, dat)

            def get_amp(dat):
                max_ = np.max(dat, axis=-1)
                min_ = np.min(dat, axis=-1)
                return (max_ - min_) / 2

            da_c = da - da.mean(dim=dim)
            freq_guess = _fix_initial_value(
                xr.apply_ufunc(get_freq, da_c, input_core_dims=[[dim]]).rename("freq guess"),
                da_c,
            )
            amp_guess = _fix_initial_value(
                xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess"), da
            )
            phase_guess = np.pi * (
                da.loc[{dim: np.abs(da.coords[dim]).min()}] < da.mean(dim=dim)
            )
            return freq_guess, amp_guess, phase_guess
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'qiskit_curve_analysis' or 'fft_based'")

    # Handle the different return types from the two methods
    if method == "fft_based":
        params_list = get_freq_and_amp_and_phase(da, dim)
        freq_guess, amp_guess, phase_guess = params_list[0], params_list[1], params_list[2]
    else:
        freq_guess, amp_guess, phase_guess = get_freq_and_amp_and_phase(da, dim)
    
    offset_guess = da.mean(dim=dim)

    def apply_fit(x, y, a, f, phi, offset):
        try:
            model = Model(oscillation, independent_vars=["t"])
            fit = model.fit(
                y,
                t=x,
                a=Parameter("a", value=a, min=0),
                f=Parameter(
                    "f", value=f, min=np.abs(0.5 * f), max=np.abs(f * 3 + 1e-3)
                ),
                phi=Parameter("phi", value=phi),
                offset=offset,
            )
            
            # Retry mechanism with different frequency bounds if R² < 0.9 (only for qiskit_curve_analysis)
            if method == "qiskit_curve_analysis" and fit.rsquared < 0.9:
                fit = model.fit(
                    y,
                    t=x,
                    a=Parameter("a", value=a, min=0),
                    f=Parameter(
                        "f",
                        value=1.0 / (np.max(x) - np.min(x)),
                        min=0,
                        max=np.abs(f * 3 + 1e-3),
                    ),
                    phi=Parameter("phi", value=phi),
                    offset=offset,
                )
            
            # Calculate NRMSE (Normalized Root Mean Square Error)
            y_pred = oscillation(x, fit.values["a"], fit.values["f"], fit.values["phi"], fit.values["offset"])
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            y_range = np.ptp(y)  # Peak-to-peak range
            nrmse = rmse / y_range if y_range > 0 else np.inf
            
            return np.array([fit.values[k] for k in ["a", "f", "phi", "offset"]]), fit.rsquared, nrmse
        except RuntimeError as e:
            print(f"{a=}, {f=}, {phi=}, {offset=}")
            plt.plot(x, oscillation(x, a, f, phi, offset))
            plt.plot(x, y)
            plt.show()
            raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        freq_guess,
        phase_guess,
        offset_guess,
        input_core_dims=[[dim], [dim], [], [], [], []],
        output_core_dims=[["fit_vals"], [], []],
        vectorize=True,
    )
    
    # Extract the fit parameters, R² values, and NRMSE values
    fit_params = fit_res[0].assign_coords(fit_vals=("fit_vals", ["a", "f", "phi", "offset"]))
    r_squared = fit_res[1]
    nrmse = fit_res[2]
    
    # Add R² and NRMSE as attributes to the fit parameters
    if hasattr(r_squared, 'values'):
        r_squared = r_squared.values
    if hasattr(nrmse, 'values'):
        nrmse = nrmse.values
    
    if isinstance(r_squared, np.ndarray):
        r_squared = float(np.mean(r_squared))
    else:
        r_squared = float(r_squared)
        
    if isinstance(nrmse, np.ndarray):
        nrmse = float(np.mean(nrmse))
    else:
        nrmse = float(nrmse)
    
    fit_params.attrs['r_squared'] = r_squared
    fit_params.attrs['nrmse'] = nrmse
    fit_params.attrs['method'] = method
    
    return fit_params


def calculate_quality_metrics(
    raw_data: np.ndarray, fitted_data: np.ndarray
) -> Dict[str, float]:
    """
    Calculate fit quality metrics: RMSE, NRMSE, and R-squared.

    Parameters
    ----------
    raw_data : np.ndarray
        The raw measurement data.
    fitted_data : np.ndarray
        The data from the Lorentzian fit.

    Returns
    -------
    Dict[str, float]
        A dictionary containing 'rmse', 'nrmse', and 'r_squared'.
    """
    residuals = raw_data - fitted_data
    rmse = np.sqrt(np.mean(residuals**2))

    # NRMSE (normalized by peak-to-peak range)
    data_range = np.ptp(raw_data)
    nrmse = rmse / data_range if data_range > 0 else np.inf

    # R-squared (coefficient of determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((raw_data - np.mean(raw_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Ensure R-squared is valid
    if not (0 <= r_squared <= 1):
        r_squared = 0  # Invalid fit

    return {"rmse": rmse, "nrmse": nrmse, "r_squared": r_squared}


# def _truncate_data(transmission, window):
#     ds_diff = np.abs(transmission.diff(dim="freq"))
#     peak_freq = ds_diff.IQ_abs.idxmax(dim="freq")
#     truncated_transmission = transmission.sel(
#         freq=slice(peak_freq - window, peak_freq + window)
#     )
#     return truncated_transmission
#
#
# def _guess_2_resonators(transmission):
#
#     # Function to find the intial guess that consists of the frequencies of two resonators, the coupling
#     # of one of the (the Purcell filter) to a feedline, and the coupling between them. The procudre is to
#     # first find the largest derivative, and assume this is where the dressed resonator is. Then, we
#     # truncate the data in a fixed window around that range, and look for the two frequencies with the
#     # largest dertivatves that are separated from each other by at least 'min_distnace'. This two dressed
#     # frequencies are converted to the bare frequencies using the coupling strength 'J'. Furthermore,
#     # a linear fit is done to the upper envelope to account for the possible linear transmission profile.
#
#     first = peaks_dips(transmission.IQ_abs, dim="freq", number=1)
#     second = peaks_dips(transmission.IQ_abs, dim="freq", number=2)
#     if first.width > second.width:
#         omega_p = first.position.values
#         kappa_p = 2 * first.width.values
#         omega_r = second.position.values
#         J = 2 * second.width.values
#     else:
#         omega_r = first.position.values
#         J = 2 * first.width.values
#         omega_p = second.position.values
#         kappa_p = 2 * second.width.values
#     k, A = (-first.base_line).polyfit(dim="freq", deg=1).polyfit_coefficients.values
#     init_params = Parameters()
#     init_params.add("J", value=J[0], min=0)
#     init_params.add("omega_r", value=omega_r[0])
#     init_params.add("omega_p", value=omega_p[0])
#     init_params.add("k", value=k[0])
#     init_params.add("A", value=A[0])
#     init_params.add("kappa_p", value=kappa_p[0], min=0)
#     init_params.add("phi", value=0)
#
#     return init_params
#
#
# class _two_resonator_model(Model):
#     # A class to fit the S21 model to a data. Accepts an xarray data
#     # that contains I and Q measured as a function of freq.
#
#     def __init__(self, J=0, kappa_p=0, *args, **kwargs):
#         super().__init__(S21_abs, *args, **kwargs)
#
#         # params used in the initial guess generator:
#         # the window one which the fitting is done, around the initial guess for the resonator peak
#         self.window = 100e6
#
#     def make_fit(self, transmission, init_guess=None):
#
#         transmission_trunc = _truncate_data(transmission, self.window)
#
#         if init_guess is None:
#             init_guess = _guess_2_resonators(transmission_trunc)
#
#         data = transmission_trunc.IQ_abs.values
#         f = transmission_trunc.freq.values
#
#         result = self.fit(data, w=f, params=init_guess)
#
#         return result
#
#
# def fit_resonator_purcell(
#     s21_data: xr.Dataset,
#     init_J: float = 15e6,
#     init_kappa_p: float = 10e6,
#     print_report: bool = False,
# ):
#     """Fits the measured complex transmission as a function of frequency
#     and fits it to a model consisting of two resonators coupled to each
#     othewr, as described in arxiv:2307.07765.
#     IMPORTANT: the fit assumes a that within the measureument window there
#     are clear two dips.
#     The transmssion function is:
#     S_{21} = (k * w + A) * [
#              cos(phi) - exp(i phi)  kappa_p  (- 2 * i * Delta_r) /
#              ( 4  J^2 + (kappa_p - 2 i Delta_p) (-2 i Delta_r))  ]
#
#     See the output for a descirption of the model parameters.
#
#     Args:
#         transmission (xarray.DataSet ): DataSet which golds the measured data,
#                                         assumes that it has a DataAray labels
#                                         'IQ_abs' and 'phase' containing the
#                                         absolute value and the phase of the
#                                         signal. The only coordinate is 'freq',
#                                         the frequency for which the signal is
#                                         measured.
#         print_report (bool, optional): If set to True prints the lmfit report.
#                                         Defaults to False.
#
#     Returns:
#         fit [lmfit.ModelResult] : The resulting fit to the data using the two resonator model.
#                             The fitted parameters (accessd through the 'params' object'
#                             are:
#                             params['J'] - coupling betweem the resonator and the Purcell
#                             filter [Hz]
#                             params['omega_r'] - bare frequency of the resonator
#                             params['omega_p'] - bare frequency of the Purcell filter
#                             params['kappa_p'] - coupling strength of the Purcell filter
#                             to the resonator
#                             params['A'] - empirical amplitude of the transmission
#                             params['k']  - an empirical linear slope of the transmission
#                             params['phi'] - a possible phase acquired by the signal due
#                             to unintended capacitance.
#                             Note that the dressed resonator frequency, the location of the
#                             resonator  dip in the signal, is not a fit parameter. It can
#                             be calculated from the model or taken from the initial guess
#                             which looks for that from 'result.init_params['omega_r']'
#         fit_eval [np.array] : A complex numpy array of the fit function evaluated on in the
#                             relevent range
#     """
#
#     resonator_abs = _two_resonator_model(J=init_J, kappa_p=init_kappa_p)
#
#     fit = resonator_abs.make_fit(s21_data)
#     fit_eval = resonator_abs.eval(params=fit.params, w=s21_data.freq.values)
#
#     if print_report:
#         print(fit.fit_report() + "\n")
#         fit.params.pretty_print()
#
#     return fit, fit_eval
#

# def _guess_single(transmission, frequency_LO_IF, rolling_window, window):
#     def find_upper_envelope(transmission, rolling_window):
#         rolling_transmission = transmission.IQ_abs.rolling(
#             freq=rolling_window, center=True
#         ).mean()
#         peak_indices, _ = find_peaks(rolling_transmission)
#         # include the edges of the range in the envelope fit in case there aren't many inside peaks to use
#         peak_indices = np.append([rolling_window, -1], peak_indices)
#         envelope = rolling_transmission.isel(freq=peak_indices)
#         k, A = envelope.polyfit(dim="freq", deg=1).polyfit_coefficients.values
#         return k, A
#
#     k, A = find_upper_envelope(transmission, rolling_window=rolling_window)
#
#     # plt.figure()
#     # transmission.IQ_abs.plot()
#     # plt.plot(transmission.freq,transmission.freq*k + A)
#     # plt.show()
#
#     omega_r = transmission.IQ_abs.idxmin(dim="freq")
#     Q = (
#         frequency_LO_IF
#         / np.abs(
#             (transmission.IQ_abs.diff(dim="freq").idxmin(dim="freq") - omega_r)
#         ).values
#     )
#     Q = Q if Q < 1e4 else 1e4
#     Q = 1e4
#     Qe = Q / (
#         1 - transmission.IQ_abs.min(dim="freq") / transmission.IQ_abs.max(dim="freq")
#     )
#
#     Qe = Qe if Qe > Q else Q
#
#     init_params = Parameters()
#     init_params.add("omega_0", value=frequency_LO_IF, vary=False)
#     init_params.add("omega_r", value=omega_r.values + 0.1e6)
#     init_params.add("k", value=k)
#     init_params.add("A", value=A)
#     init_params.add("Q", value=Q, min=0)
#     init_params.add("Qe_real", value=Qe.values, min=0)
#     init_params.add("Qe_imag", value=0, min=0)
#
#     return init_params
#
#
# class _single_resonator(Model):
#     # A class to fit the S21 model to a data. Accepts an xarray data
#     # that contains I and Q measured as a function of freq.
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(S21_single, *args, **kwargs)
#
#         # params used in the initial guess generator:
#         # used to smooth data to improve peak detection
#         self.rolling_window = 1
#         # the window one which the fitting is done, around the initial guess for the resonator peak
#         self.window = 15e6
#
#     def make_fit(self, transmission, frequency_LO_IF, init_guess=None):
#
#         # transmission_trunc = _truncate_data(transmission,self.window)
#         transmission_trunc = transmission
#
#         if init_guess is None:
#             init_guess = _guess_single(
#                 transmission_trunc,
#                 frequency_LO_IF=frequency_LO_IF,
#                 rolling_window=self.rolling_window,
#                 window=self.window,
#             )
#
#         data = (
#             transmission_trunc.IQ_abs * np.exp(1j * transmission_trunc.phase)
#         ).values
#         f = transmission_trunc.freq.values
#
#         result = self.fit(data, w=f, params=init_guess)
#
#         return result
#
#
# def fit_resonator(
#     s21_data: xr.Dataset, frequency_LO_IF: float, print_report: bool = False
# ):
#     """Fits the measured complex transmission as a function of frequency
#     and fits it to a model consisting of a single resonator, as described in
#     arxiv:1108.3117.
#
#     The transmission function is:
#     S_{21} =(A + k  w)  (
#         1 - ((Q/Qe) / (1 + 2 i Q  (w - omega_r)/(omega_0 + omega_r))))
#
#     See the output for a description of the model parameters.
#
#     Args:
#         transmission (xarray.DataSet ): DataSet which golds the measured data,
#                                         assumes that it has a DataAray labels
#                                         'IQ_abs' and 'phase' containing the
#                                         absolute value and the phase of the
#                                         signal. The only coordinate is 'freq',
#                                         the frequency for which the signal is
#                                         measured.
#         frequency_LO_IF (int): The frequency relative to which the data was taken.
#                                 Should be the sum of the LO and IF.
#         print_report (bool, optional): If set to True prints the lmfit report.
#                                         Defaults to False.
#
#     Returns:
#         fit [lmfit.ModelResult] : The resulting fit to the data using the two resonator model.
#                             The fitted parameters (accessed through the 'params' object'
#                             are:
#                             params['omega_r'] - resonator frequency
#                             params['Qe_imag'] - imaginary part of the external quality
#                             factor
#                             params['Qe_real'] - real part of the external quality
#                             factor
#                             params['Q'] - the total quality factor of the resonator
#                             params['A'] - empirical amplitude of the transmission
#                             params['k']  - an empirical linear slope of the transmission
#         fit_eval [np.array] : A complex numpy array of the fit function evaluated on in the
#                             relevant range
#     """
#
#     resonator = _single_resonator()
#
#     fit = resonator.make_fit(s21_data, frequency_LO_IF=frequency_LO_IF)
#     fit_eval = resonator.eval(params=fit.params, w=s21_data.freq.values)
#
#     if print_report:
#         print(fit.fit_report() + "\n")
#         fit.params.pretty_print()
#
#     return fit, fit_eval


def circle_fit_s21_resonator_model(dataset: xr.Dataset):
    """
    Performs a full S21 circle fit for each qubit in the raw xarray Dataset.
    """
    required_vars = ['full_freq', 'I', 'Q']
    if not all(var in dataset for var in required_vars):
        print(f"Error: Dataset must contain the data variables: {required_vars}")
        return None

    results = {}
    fitters = {}
    qubits = dataset.coords["qubit"].values

    for qubit in qubits:
        # print(f"\n{'='*20}\n--- Fitting qubit: {qubit} ---\n{'='*20}")
        qubit_data = dataset.sel(qubit=qubit)
        frequencies = qubit_data["full_freq"].values
        I_data = qubit_data["I"].values
        Q_data = qubit_data["Q"].values
        valid_indices = ~np.isnan(frequencies) & ~np.isnan(I_data) & ~np.isnan(Q_data)
        frequencies, I_data, Q_data = frequencies[valid_indices], I_data[valid_indices], Q_data[valid_indices]
        s21_complex = I_data + 1j * Q_data

        fitter = S21Resonator(frequencies, s21_complex)
        fit_params = fitter.fit()

        if fit_params:
            results[qubit] = fit_params
            # print("S21 Fit successful.")
            fitters[qubit] = fitter
        else:
            print("S21 Fit failed.")
            results[qubit] = {"fit_parameters": None}

    return results, fitters