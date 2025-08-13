import xarray as xr
import numpy as np

from typing import List

from qm import QmJob

from qualibration_libs.legacy.components import Transmon
from qualibration_libs.legacy.experiments.iq_blobs.parameters import IQBlobsParameters
from qualibration_libs.legacy.lib.qua_datasets import convert_IQ_to_V
from qualibration_libs.legacy.lib.save_utils import fetch_results_as_xarray


def fetch_dataset(job: QmJob, qubits: List[Transmon], node_parameters: IQBlobsParameters) -> xr.Dataset:
    """
    Fetch the data from the OPX and convert it into a xarray with corresponding axes
    """
    ds = fetch_results_as_xarray(
        job.result_handles, qubits,
        measurement_axis={"N": np.linspace(1, node_parameters.num_runs, node_parameters.num_runs)}
    )

    # Fix the structure of ds to avoid tuples
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,  # This ensures the function is applied element-wise
        dask="parallelized",  # This allows for parallel processing
        output_dtypes=[float],  # Specify the output data type
    )

    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, qubits, ["I_g", "Q_g", "I_e", "Q_e"])

    return ds