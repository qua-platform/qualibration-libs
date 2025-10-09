from matplotlib import pyplot as plt
from qualibration_libs.analysis import guess
from scipy.optimize import curve_fit
import numpy as np
import xarray as xr
from lmfit import Model, Parameter

from qualibration_libs.analysis.models import *


__all__ = ["fit_oscillation", "fit_oscillation_decay_exp", "fit_decay_exp", "unwrap_phase"]


def unwrap_phase(da, dim):
    """
    Unwraps the phase of a DataArray along a specified dimension.
    This is useful for correcting phase jumps in the data.

    Parameters:
    da (xr.DataArray): The input DataArray containing phase data.
    dim (str): The dimension along which to unwrap the phase.

    Returns:
    xr.DataArray: A new DataArray with the unwrapped phase.
    """
    return xr.apply_ufunc(
        np.unwrap, da, input_core_dims=[[dim]], output_core_dims=[[dim]]
    )

def _fix_initial_value(x, da):
    if len(da.dims) == 1:
        return float(x)
    else:
        return x


def fit_decay_exp(da, dim):
    def get_decay(dat):
        def oed(d):
            return guess.exp_decay(da[dim], d)

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
            return guess.oscillation_exp_decay(da[dim], d)

        return np.apply_along_axis(oed, -1, dat)

    def get_freq(dat):
        def f(d):
            return guess.frequency(da[dim], d)

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


def fit_oscillation(da, dim):
    def get_freq(dat):
        def f(d):
            return guess.frequency(da[dim], d)

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
    # phase_guess = np.pi * (da.loc[{dim : da.coords[dim].values[0]}] < da.mean(dim=dim) )
    phase_guess = np.pi * (
        da.loc[{dim: np.abs(da.coords[dim]).min()}] < da.mean(dim=dim)
    )
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
            if fit.rsquared < 0.9:
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
            return np.array([fit.values[k] for k in ["a", "f", "phi", "offset"]])
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
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(fit_vals=("fit_vals", ["a", "f", "phi", "offset"]))


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
