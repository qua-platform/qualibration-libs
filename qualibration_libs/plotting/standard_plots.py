import plotly
import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np


def plot_active_reset_attempts(ar_data: dict[str, np.typing.NDArray], **hist_kwargs):
    # TODO: is this relevant in the context of the calibration graph?
    """
    plot the distribution of active reset attempts acquired when running `reset_active(save_qua_var=...)`

    ar_data - a dictionary of the form {qubit_name: ar_attempts}
    hist_kwargs - kwargs to pass to matplotlib.pyplot.hist

    Example:
        # in the QUA program, save the number of active reset attempts in a variable as follows:

        q.reset_active(save_qua_var=f'ar_{q.name}')

        # then in the python script, fetch the data and plot it as follows:
        from qualibration_libs.plot_utils import plot_active_reset_attempts
        ar_dat = {}
        for q in machine.active_qubits:
        for q in machine.active_qubits:
            qn = q.name
            ar_dat[qn] = job.result_handles.get(f'ar_{qn}').fetch_all()['value']

        I_ar = {}
        for q in machine.active_qubits:
        for q in machine.active_qubits:
            qn = q.name
            I_ar[qn] = job.result_handles.get(f'I_ar_{qn}').fetch_all()['value']

        plot_active_reset_attempts(ar_dat, bins=100, log=True)
    """
    qubits = list(ar_data.keys())
    fig, axes = plt.subplots(1, len(qubits), sharex="None", figsize=(14, 4), squeeze=False)
    for i, q in enumerate(qubits):
        ax = axes[i]
        ax.hist(ar_data[q], **hist_kwargs)
        ax.set_title(f"q = {q}")
        ax.set_ylabel("prob.")
        ax.set_xlabel("no. of attempts")
    fig.tight_layout()


def plot_spectrum(
    signal: np.ndarray, t_s_usec: float, num_zero_pad: int = 0
) -> tuple[np.ndarray, np.ndarray, plotly.graph_objs.Figure]:
    # TODO: is this relevant in the context of the calibration graph? This could go to the py-qua-tools if useful
    """
    plot the spectrum of a signal

    signal - 1D array to plot
    t_s_usec - sampling time interval in µs
    num_zero_pad - how much to zero pad the signal

    returns: the spectrum (abs^2), the frequency vector and the plotly figure object
    """
    signal_ac = signal - signal.mean()
    signal_pad = np.hstack((signal_ac, np.zeros(num_zero_pad)))
    n = len(signal_pad)
    signal_fft = np.abs(np.fft.fft(signal_pad)) ** 2

    freq_ax = np.fft.fftfreq(len(signal_pad), d=t_s_usec)
    f_s = 0.5 / t_s_usec

    fig = px.line(x=freq_ax[1:], y=signal_fft[1:], labels={"x": "frequency [MHz]", "y": "power"})
    fig.update_layout(xaxis_range=(0, f_s))
    return signal_fft, freq_ax, fig
