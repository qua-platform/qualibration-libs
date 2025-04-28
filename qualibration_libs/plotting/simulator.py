from typing import List
import plotly.graph_objects as go
from qm import QmJob
import numpy as np


def get_simulated_samples_by_element(element_name: str, job: QmJob, config: dict):
    # TODO: this seems very deprecated since it doesn't take into account the OPX1000 --> shall we keep it?
    element = config["elements"][element_name]
    sample_struct = job.get_simulated_samples()
    if "mixInputs" in element:
        port_i = element["mixInputs"]["I"]
        port_q = element["mixInputs"]["Q"]
        samples = (
            sample_struct.__dict__[port_i[0]].analog[str(port_i[1])]
            + 1j * sample_struct.__dict__[port_q[0]].analog[str(port_q[1])]
        )
    else:
        port = element["singleInput"]["port"]
        samples = sample_struct.__dict__[port[0]].analog[str(port[1])]
    return samples


def plot_simulator_output(plot_axes: List[List[str]], job: QmJob, config: dict, duration_nsec: int):
    # TODO: Is this even used or useful?
    """
    generate a plot of simulator output by elements

    :param plot_axes: a list of lists of elements. Will open
    multiple axes, one for each list.
    :param job: The simulated QmJob to plot
    :param config: The config file used to create the job
    :param duration_nsec: the duration to plot in nsec
    """
    time_vec = np.linspace(0, duration_nsec - 1, duration_nsec)
    samples_struct = []
    for plot_axis in plot_axes:
        samples_struct.append([get_simulated_samples_by_element(pa, job, config) for pa in plot_axis])

    fig = go.Figure().set_subplots(rows=len(plot_axes), cols=1, shared_xaxes=True)

    for i, plot_axis in enumerate(plot_axes):
        for j, plotitem in enumerate(plot_axis):
            if samples_struct[i][j].dtype == float:
                fig.add_trace(
                    go.Scatter(x=time_vec, y=samples_struct[i][j], name=plotitem),
                    row=i + 1,
                    col=1,
                )
                print(samples_struct[i][j])
            else:
                fig.add_trace(
                    go.Scatter(x=time_vec, y=samples_struct[i][j].real, name=plotitem + " I"),
                    row=i + 1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(x=time_vec, y=samples_struct[i][j].imag, name=plotitem + " Q"),
                    row=i + 1,
                    col=1,
                )
    fig.update_xaxes(title="time [nsec]")
    return fig
