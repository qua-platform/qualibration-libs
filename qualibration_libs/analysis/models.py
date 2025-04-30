import numpy as np

__all__ = [
    "oscillation",
    "lorentzian_peak",
    "oscillation_decay_exp",
    "lorentzian_dip",
    "decay_exp",
]


def lorentzian_peak(x, amplitude, center, width, offset):
    """Computes the Lorentzian peak function.

    Args:
        x: The input values at which to evaluate the Lorentzian function.
        amplitude: The amplitude of the Lorentzian peak.
        center: The center position of the Lorentzian peak.
        width: The full width at half maximum (FWHM) of the Lorentzian peak.
        offset: The offset value added to the Lorentzian function.

    Returns:
        The evaluated Lorentzian function at the input values `x`.

    Notes:
        The Lorentzian peak function is defined as:
        L(x) = offset + amplitude * (1 / (1 + ((x - center) / width)^2))
        This function is commonly used to model resonance peaks in qubit spectroscopy.
    """
    return offset + amplitude * (1 / (1 + ((x - center) / width) ** 2))


def lorentzian_dip(x, amplitude, center, width, offset):
    """Computes the Lorentzian dip function.

    Args:
        x: The input values at which to evaluate the Lorentzian function.
        amplitude: The amplitude of the Lorentzian dip.
        center: The center position of the Lorentzian dip.
        width: The full width at half maximum (FWHM) of the Lorentzian dip.
        offset: The offset value from which the Lorentzian dip subtracts.

    Returns:
        The evaluated Lorentzian function at the input values `x`.

    Notes:
        The Lorentzian dip function is defined as:
        L(x) = offset - (amplitude * width^2) / (width^2 + (x - center)^2)
        This function is commonly used to model resonance dips in spectroscopy.
    """
    return offset - amplitude * width**2 / (width**2 + (x - center) ** 2)


def oscillation(t, a, f, phi, offset):
    """Computes a sinusoidal oscillation.

    Args:
        t: Time values.
        a: Amplitude of the oscillation.
        f: Frequency of the oscillation.
        phi: Phase offset of the oscillation (in radians).
        offset: Vertical offset of the oscillation.

    Returns:
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

    Args:
        t: Time values.
        a: Initial amplitude of the oscillation.
        f: Frequency of the oscillation.
        phi: Phase offset of the oscillation (in radians).
        offset: Vertical offset of the oscillation.
        decay: Exponential decay rate (1/T_decay).

    Returns:
        The computed decaying oscillation values at times `t`.
    """
    return a * np.exp(-t * decay) * np.cos(2 * np.pi * f * t + phi) + offset


def decay_exp(t, a, offset, decay, **kwargs):
    """Computes a simple exponential decay or growth.

    Args:
        t: Time values.
        a: Amplitude scaling factor.
        offset: Vertical offset.
        decay: Exponential decay rate (positive for decay, negative for growth).
        **kwargs: Catches unused keyword arguments (e.g., from curve_fit).

    Returns:
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
