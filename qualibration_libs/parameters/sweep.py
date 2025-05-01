import numpy as np
from typing import Literal
from qualibrate.parameters import RunnableParameters


class IdleTimeNodeParameters(RunnableParameters):
    """Common parameters for configuring a node in a quantum machine simulation or execution."""

    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 30000
    """Maximum wait time in nanoseconds. Default is 30000."""
    wait_time_num_points: int = 500
    """Number of points for the wait time scan. Default is 500."""
    log_or_linear_sweep: Literal["log", "linear"] = "log"
    """Type of sweep, either "log" (logarithmic) or "linear". Default is "log"."""


def get_idle_times_in_clock_cycles(
    node_parameters: IdleTimeNodeParameters,
) -> np.ndarray:
    """
    Get the idle-times sweep axis according to the sweep type given by ``node.parameters.log_or_linear_sweep``.

    The idle time sweep is in units of clock cycles (4ns).
    The minimum is 4 clock cycles.
    """
    required_attributes = [
        "log_or_linear_sweep",
        "min_wait_time_in_ns",
        "max_wait_time_in_ns",
        "wait_time_num_points",
    ]
    if not all(hasattr(node_parameters, attr) for attr in required_attributes):
        raise ValueError(
            "The provided node parameter must contain the attributes 'log_or_linear_sweep', 'min_wait_time_in_ns', 'max_wait_time_in_ns' and 'wait_time_num_points'."
        )

    if node_parameters.log_or_linear_sweep == "linear":
        idle_times = _get_idle_times_linear_sweep_in_clock_cycles(node_parameters)
    elif node_parameters.log_or_linear_sweep == "log":
        idle_times = _get_idle_times_log_sweep_in_clock_cycles(node_parameters)
    else:
        raise ValueError(
            f"Expected sweep type to be 'log' or 'linear', got {node_parameters.log_or_linear_sweep}"
        )

    return idle_times


def _get_idle_times_linear_sweep_in_clock_cycles(
    node_parameters: IdleTimeNodeParameters,
):
    return (
        np.linspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)


def _get_idle_times_log_sweep_in_clock_cycles(node_parameters: IdleTimeNodeParameters):
    return np.unique(
        np.geomspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)
