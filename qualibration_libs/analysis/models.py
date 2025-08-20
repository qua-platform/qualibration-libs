import traceback
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq, minimize

__all__ = [
    "oscillation",
    "lorentzian_peak",
    "oscillation_decay_exp",
    "lorentzian_dip",
    "decay_exp",
    "S21Resonator",
]


def lorentzian_peak(x, amplitude, center, width, offset):
    """Computes the Lorentzian peak function.

    Args
        x: The input values at which to evaluate the Lorentzian function.
        amplitude: The amplitude of the Lorentzian peak.
        center: The center position of the Lorentzian peak.
        width: The full width at half maximum (FWHM) of the Lorentzian peak.
        offset: The offset value added to the Lorentzian function.

    Returns
        The evaluated Lorentzian function at the input values `x`.

    Notes:
        The Lorentzian peak function is defined as:
        L(x) = offset + amplitude * (1 / (1 + ((x - center) / width)^2))
        This function is commonly used to model resonance peaks in qubit spectroscopy.
    """
    return offset + amplitude * (1 / (1 + ((x - center) / width) ** 2))


def lorentzian_dip(x, amplitude, center, width, offset):
    """Computes the Lorentzian dip function.

    Args
        x: The input values at which to evaluate the Lorentzian function.
        amplitude: The amplitude of the Lorentzian dip.
        center: The center position of the Lorentzian dip.
        width: The full width at half maximum (FWHM) of the Lorentzian dip.
        offset: The offset value from which the Lorentzian dip subtracts.

    Returns
        The evaluated Lorentzian function at the input values `x`.

    Notes:
        The Lorentzian dip function is defined as:
        L(x) = offset - (amplitude * width^2) / (width^2 + (x - center)^2)
        This function is commonly used to model resonance dips in spectroscopy.
    """
    return offset - amplitude * width**2 / (width**2 + (x - center) ** 2)


def oscillation(t, a, f, phi, offset):
    """Computes a sinusoidal oscillation.

    Args
        t: Time values.
        a: Amplitude of the oscillation.
        f: Frequency of the oscillation.
        phi: Phase offset of the oscillation (in radians).
        offset: Vertical offset of the oscillation.

    Returns
        The computed oscillation values at times `t`.
    """
    return a * np.cos(2 * np.pi * f * t + phi) + offset


# def echo_decay_exp(t, a, offset, decay, decay_echo):
#     """Computes an exponential decay combined with a Gaussian echo decay.
#
#     This model is often used for echo experiments (e.g., Hahn echo).
#
#     Args:
#         t: Time values.
#         a: Initial amplitude.
#         offset: Vertical offset.
#         decay: Exponential decay rate (1/T_exp).
#         decay_echo: Gaussian echo decay rate (1/T_echo).
#
#     Returns:
#         The computed decay values at times `t`.
#     """
#     return a * np.exp(-t * decay - (t * decay_echo) ** 2) + offset


def oscillation_decay_exp(t, a, f, phi, offset, decay):
    """Computes a decaying sinusoidal oscillation (exponential decay).

    Args
        t: Time values.
        a: Initial amplitude of the oscillation.
        f: Frequency of the oscillation.
        phi: Phase offset of the oscillation (in radians).
        offset: Vertical offset of the oscillation.
        decay: Exponential decay rate (1/T_decay).

    Returns
        The computed decaying oscillation values at times `t`.
    """
    return a * np.exp(-t * decay) * np.cos(2 * np.pi * f * t + phi) + offset


def decay_exp(t, a, offset, decay, **kwargs):
    """Computes a simple exponential decay or growth.

    Args
        t: Time values.
        a: Amplitude scaling factor.
        offset: Vertical offset.
        decay: Exponential decay rate (positive for decay, negative for growth).
        **kwargs: Catches unused keyword arguments (e.g., from curve_fit).

    Returns
        The computed exponential decay/growth values at times `t`.
    """
    return a * np.exp(t * decay) + offset


# def S21_abs(w, A, k, phi, kappa_p, omega_p, omega_r, J):
#     """Calculates the absolute value of the S21 transmission for two coupled resonators.
#
#     This model describes two resonators in a shunt geometry, assuming a wide-band resonator
#     (Purcell filter) coupled to a feedline (coupling kappa_p), which in turn is coupled
#     (coupling J) to another resonator. Based on arXiv:2307.07765.
#
#     Args:
#         w: Angular frequency values.
#         A: Base transmission amplitude (offset).
#         k: Linear slope of transmission vs frequency far from resonance.
#         phi: Phase factor related to impedance mismatch or background transmission.
#         kappa_p: Coupling rate of the Purcell filter resonator to the feedline.
#         omega_p: Resonance frequency of the Purcell filter resonator.
#         omega_r: Resonance frequency of the second resonator.
#         J: Coupling strength between the two resonators.
#
#     Returns:
#         The absolute value of the S21 transmission parameter at frequencies `w`.
#     """
#     # real transmission function for two resonator in a shunt geometry, assuming
#     # a wide-band resonator (purcell filter) coupled with a coupling strength kappa_p
#     # to a feedline, which in turn is coupled with strength J to another resonator
#     # based on arxiv:2307.07765
#
#     Delta_p = omega_p - w
#     Delta_r = omega_r - w
#
#     return (A + k * w) * np.abs(
#         np.cos(phi)
#         - np.exp(1j * phi)
#         * kappa_p
#         * (-2 * 1j * Delta_r)
#         / (4 * J**2 + (kappa_p - 2 * 1j * Delta_p) * (-2 * 1j * Delta_r))
#     )
#
#
# def S21_single(w, A, k, omega_0, omega_r, Q, Qe_real, Qe_imag):
#     """Calculates the complex S21 transmission for a single resonator.
#
#     Based on Khalil et al. (arXiv:1108.3117), modified to account for non-normalized
#     transmission and a linear slope far from resonance.
#
#     Args:
#         w: Angular frequency values.
#         A: Base transmission amplitude (offset).
#         k: Linear slope of transmission vs frequency far from resonance.
#         omega_0: Reference frequency (often omega_r, used for normalization in original model).
#         omega_r: Resonance frequency of the resonator.
#         Q: Total quality factor of the resonator (loaded Q).
#         Qe_real: Real part of the external quality factor.
#         Qe_imag: Imaginary part of the external quality factor (accounts for asymmetry).
#
#     Returns:
#         The complex S21 transmission parameter at frequencies `w`.
#     """
#     # complex transmision function for a single resonator based on Khalil et al. (arxiv:1108.3117):
#     #  but slightly modified for convenince, by taking into account a non-normalized transmission
#     # and a linear slope of the transmission far from resonance
#
#     Qe = Qe_real + 1j * Qe_imag
#     return (A + k * w) * (
#         1 - ((Q / Qe) / (1 + 2 * 1j * Q * (w - omega_r) / (omega_0 + omega_r)))
#     )

class S21Resonator:
    """
    A class to fit 1D resonator spectroscopy data using the S21 model.
    Ref: https://doi.org/10.7907/RAT0-VM75
    """
    _DELAY_FIT_PERCENTAGE = 10
    _PHASES_THRESHOLD_PERCENTAGE = 80
    _STD_DEV_GAUSSIAN_KERNEL = 5
    _PHASE_ELEMENTS = 5
    _ResonatorData = namedtuple("ResonatorData", ["freq", "signal", "phase"])

    def __init__(self, frequencies: NDArray, s21_complex: NDArray):
        """Initializes the S21Resonator fitter.

        Args
            frequencies (NDArray): The frequency points of the resonator scan.
            s21_complex (NDArray): The complex S21 data corresponding to the
                frequencies.
        """
        self.frequencies = np.asarray(frequencies)
        self.s21_complex = np.asarray(s21_complex)
        self.fit_params = None
        self.full_s21_model = None
        self.quality_metrics = None

    def fit(self) -> dict:
        """Performs the S21 fit and assesses its quality.

        This method executes the full fitting routine, which includes estimating
        and removing cable delay, performing a circle fit to find the resonance
        circle, and then fitting the phase response to extract the resonator
        parameters. Finally, it assesses the fit quality.

        Returns
            Optional[dict]: A dictionary containing the readable fitted parameters
            and goodness-of-fit metrics (RMSE and R-squared). The parameters include
            'frequency', 'fwhm', 'loaded_q', 'coupling_q', 'internal_q',
            'phi_mismatch_rad', 'amplitude_attenuation', 'phase_shift_rad',
            and 'cable_delay_s'. Returns None if the fitting fails.
        """
        data_to_fit = self._ResonatorData(
            freq=self.frequencies,
            signal=np.abs(self.s21_complex),
            phase=np.unwrap(np.angle(self.s21_complex)),
        )
        try:
            # 1. Perform the core fitting routine to get model parameters
            resonance_freq, model_parameters, errors = self._s21_fit_routine(data_to_fit)
            
            # 2. Generate the full S21 model data based on the fit
            self.full_s21_model = self._s21_model(self.frequencies, *model_parameters)

            # 3. Assess the quality of the fit
            quality_metrics = self.assess_fit_quality()

            # 4. Compile the final results dictionary
            q_loaded, q_coupling = model_parameters[1], model_parameters[2]
            if q_loaded > 0 and np.abs(q_coupling) > 0 and (1.0 / q_loaded - 1.0 / np.abs(q_coupling)) > 0:
                 q_internal = 1.0 / (1.0 / q_loaded - 1.0 / np.abs(q_coupling))
            else:
                 q_internal = 0

            fwhm_hz = resonance_freq / q_loaded if q_loaded > 0 else 0.0
            
            self.fit_params = {
                "frequency": model_parameters[0],
                "fwhm": fwhm_hz, # <-- FWHM is now part of the results
                "loaded_q": q_loaded, "coupling_q": q_coupling, "internal_q": q_internal,
                "phi_mismatch_rad": model_parameters[3], "amplitude_attenuation": model_parameters[4],
                "phase_shift_rad": model_parameters[5], "cable_delay_s": model_parameters[6],
            }
            
            # Add quality metrics to the results
            self.quality_metrics = quality_metrics

            return self.fit_params
            
        except Exception as e:
            print(f"An error occurred during fitting: {e}")
            traceback.print_exc()
            self.fit_params = None
            return None
            
    def assess_fit_quality(self) -> dict:
        """Calculates quantitative metrics for the goodness of fit.

        This method calculates the Root Mean Square Error (RMSE), R-squared,
        and Normalized Root Mean Square Error (NRMSE) for the complex fit.

        Returns
            Optional[dict]: A dictionary containing fit quality metrics: 'rmse',
            'r_squared', and 'nrmse'. Returns None if the fit has not been
            performed.
        """
        if self.full_s21_model is None:
            print("Fit must be performed before assessing quality.")
            return None

        observed = self.s21_complex
        expected = self.full_s21_model

        # 1. Calculate complex residuals (difference between observed and expected)
        residuals = observed - expected

        # 2. Calculate Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean(np.abs(residuals) ** 2))

        # 3. Calculate R-squared (Coefficient of Determination) for complex data
        total_sum_of_squares = np.sum(np.abs(observed - np.mean(observed)) ** 2)
        residual_sum_of_squares = np.sum(np.abs(residuals) ** 2)
        r_squared = 1.0 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else 1.0

        # 4. Normalized RMSE (NRMSE) - normalized by the range of observed data
        observed_range = np.ptp(np.abs(observed))
        nrmse = rmse / observed_range if observed_range > 0 else np.nan

        return {
            "rmse": rmse,
            "r_squared": r_squared,
            "nrmse": nrmse,
        }

    def _s21_model(self, frequencies: NDArray, resonance: float, q_loaded: float, q_coupling: float, phi: float = 0.0, amplitude: float = 1.0, alpha: float = 0.0, tau: float = 0.0) -> NDArray:
        """Complex S21 resonator model.

        Args
            frequencies (NDArray): Frequencies at which to evaluate the model.
            resonance (float): The resonance frequency.
            q_loaded (float): The loaded quality factor.
            q_coupling (float): The coupling quality factor.
            phi (float, optional): The phase mismatch. Defaults to 0.0.
            amplitude (float, optional): The amplitude attenuation. Defaults to 1.0.
            alpha (float, optional): The phase shift. Defaults to 0.0.
            tau (float, optional): The cable delay in seconds. Defaults to 0.0.

        Returns
            NDArray: The complex S21 data calculated from the model.
        """
        return (amplitude * np.exp(1j * alpha) * np.exp(-2 * np.pi * 1j * frequencies * tau) * (1 - ((q_loaded / (np.abs(q_coupling))) * np.exp(1j * phi)) / (1 + 2j * q_loaded * (frequencies / resonance - 1))))
    def _s21_fit_routine(self, data: _ResonatorData) -> tuple[float, list[float], list[float]]:
        """Core S21 fitting routine.

        This private method orchestrates the multi-step fitting process.

        Args
            data (_ResonatorData): A named tuple containing frequency and complex
                S21 data.

        Returns
            Tuple[float, List[float], List[float]]: A tuple containing:
                - The resonance frequency.
                - A list of the seven model parameters.
                - A list of errors (currently placeholders).
        """
        f_data, z_data = data.freq, data.signal * np.exp(1j * data.phase)
        num_points = int(len(f_data) * self._DELAY_FIT_PERCENTAGE / 100)
        tau = self._cable_delay(f_data, data.phase, num_points)
        z_1 = self._remove_cable_delay(f_data, z_data, tau)
        z_c, r_0 = self._circle_fit(z_1)
        phases = np.unwrap(np.angle(z_1 - z_c))
        resonance, q_loaded, theta = self._phase_fit(f_data, phases)
        beta = self._periodic_boundary(theta - np.pi)
        off_resonant_point = z_c + r_0 * np.cos(beta) + 1j * r_0 * np.sin(beta)
        amplitude = np.abs(off_resonant_point); alpha = np.angle(off_resonant_point); phi = self._periodic_boundary(beta - alpha)
        q_coupling = q_loaded / (2 * (r_0 / amplitude)) / np.cos(phi) if np.cos(phi) != 0 else 0
        return resonance, [resonance, q_loaded, q_coupling, phi, amplitude, alpha, tau], [0.0] * 7
    def _cable_delay(self, frequencies: NDArray, phases: NDArray, num_points: int) -> float:
        """Estimates the cable delay from the phase response.

        Args
            frequencies (NDArray): Frequency data.
            phases (NDArray): Unwrapped phase data.
            num_points (int): Number of points from each end of the data to use
                for the linear fit.

        Returns
            float: The estimated cable delay (tau) in seconds.
        """
        freqs, phs = np.concatenate((frequencies[:num_points], frequencies[-num_points:])), np.concatenate((phases[:num_points], phases[-num_points:]))
        return np.polyfit(freqs, phs, 1)[0] / (-2 * np.pi)
    def _remove_cable_delay(self, frequencies: NDArray, z: NDArray, tau: float) -> NDArray:
        """Removes the estimated cable delay from the complex data.

        Args
            frequencies (NDArray): Frequency data.
            z (NDArray): Complex S21 data.
            tau (float): The estimated cable delay in seconds.

        Returns
            NDArray: The complex S21 data with the cable delay removed.
        """
        return z * np.exp(2j * np.pi * frequencies * tau)
    def _circle_fit(self, z: NDArray) -> tuple[complex, float]:
        """Fits the complex data to a circle in the IQ plane.

        Args
            z (NDArray): Complex S21 data (with cable delay removed).

        Returns
            Tuple[complex, float]: A tuple containing:
                - The center of the fitted circle (z_c) as a complex number.
                - The radius of the fitted circle (r_0).
        """
        z_copy = z.copy()
        x_norm, y_norm = 0.5 * (np.max(z_copy.real) + np.min(z_copy.real)), 0.5 * (np.max(z_copy.imag) + np.min(z_copy.imag))
        z_copy -= x_norm + 1j * y_norm
        amp_norm = np.max(np.abs(z_copy)); z_copy /= amp_norm
        coords = np.stack([np.abs(z_copy)**2, z_copy.real, z_copy.imag, np.ones_like(z_copy, dtype=np.float64)])
        m = np.einsum("in,jn->ij", coords, coords)
        b = np.array([[0, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 0], [-2, 0, 0, 0]])
        coeffs = np.linalg.eigvals(np.linalg.inv(b).dot(m))
        eta = np.min(np.real([c for c in coeffs if np.isreal(c) and c > 0]))
        res = minimize(lambda a, m, b, e: a.T @ m @ a - e * (a.T @ b @ a - 1), np.ones(4), args=(m, b, eta), constraints=[{"type": "eq", "fun": lambda a, b: a.T @ b @ a - 1, "args": (b,)}])
        a = res.x
        x_c, y_c = -a[1] / (2 * a[0]), -a[2] / (2 * a[0])
        r_0 = 1 / (2 * np.abs(a[0]) * np.sqrt(a[1]**2 + a[2]**2 - 4 * a[0] * a[3]))
        return (complex(x_c * amp_norm + x_norm, y_c * amp_norm + y_norm), r_0 * amp_norm)
    def _phase_fit(self, frequencies: NDArray, phases: NDArray) -> NDArray:
        """Fits the phase response to extract resonance and Q factor.

        Args
            frequencies (NDArray): Frequency data.
            phases (NDArray): Unwrapped phase data from the circle fit.

        Returns
            NDArray: An array containing the fitted parameters:
                (resonance_frequency, loaded_q, phase_offset_theta).
        """
        roll_off = 2 * np.pi if np.max(phases) - np.min(phases) > self._PHASES_THRESHOLD_PERCENTAGE / 100 * 2 * np.pi else np.max(phases) - np.min(phases)
        phases_smooth = gaussian_filter1d(phases, self._STD_DEV_GAUSSIAN_KERNEL)
        resonance_guess = frequencies[np.argmax(np.abs(np.gradient(phases_smooth)))]
        q_loaded_guess = 2 * resonance_guess / (frequencies[-1] - frequencies[0])
        tau_guess = -(phases[-1] - phases[0] + roll_off) / (2 * np.pi * (frequencies[-1] - frequencies[0]))
        theta_guess = 0.5 * (np.mean(phases[:self._PHASE_ELEMENTS]) + np.mean(phases[-self._PHASE_ELEMENTS:]))
        def res_full(p): return self._phase_dist(phases - self._phase_centered_model(frequencies, *p))
        p_final = leastsq(lambda p: res_full((resonance_guess, p[0], theta_guess, tau_guess)), [q_loaded_guess])[0]
        q_loaded_guess = p_final[0]
        p_final = leastsq(lambda p: res_full((p[0], q_loaded_guess, p[1], tau_guess)), [resonance_guess, theta_guess])[0]
        resonance_guess, theta_guess = p_final
        p_final = leastsq(lambda p: res_full((resonance_guess, q_loaded_guess, theta_guess, p[0])), [tau_guess])[0]
        tau_guess = p_final[0]
        p_final = leastsq(lambda p: res_full((p[0], p[1], theta_guess, tau_guess)), [resonance_guess, q_loaded_guess])[0]
        resonance_guess, q_loaded_guess = p_final
        p_final = leastsq(res_full, [resonance_guess, q_loaded_guess, theta_guess, tau_guess])[0]
        return p_final[:-1]
    def _phase_dist(self, phases: NDArray) -> NDArray:
        """Calculates the distance for periodic phase data.

        This is used as the residual function in the phase fit.

        Args
            phases (NDArray): The phase differences.

        Returns
            NDArray: The wrapped phase distances.
        """
        return np.pi - np.abs(np.pi - np.abs(phases))
    def _phase_centered_model(self, frequencies: NDArray, resonance: float, q_loaded: float, theta: float, tau: float = 0.0) -> NDArray:
        """The model for the phase response of the resonator.

        Args
            frequencies (NDArray): Frequency data.
            resonance (float): The resonance frequency.
            q_loaded (float): The loaded quality factor.
            theta (float): The phase offset.
            tau (float, optional): The cable delay. Defaults to 0.0.

        Returns
            NDArray: The modeled phase response.
        """
        return (theta - 2 * np.pi * tau * (frequencies - resonance) + 2.0 * np.arctan(2.0 * q_loaded * (1.0 - frequencies / resonance)))
    def _periodic_boundary(self, angle: float) -> float:
        """Wraps an angle to the interval [-pi, pi].

        Args
            angle (float): The input angle in radians.

        Returns
            float: The wrapped angle in radians.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi