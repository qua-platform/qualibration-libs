"""
Data preparator for power Rabi plots (node 04b).

Transforms raw power Rabi datasets and fit results into standardized format
for the enhanced PlotConfig system.
"""

from typing import List, Optional
import numpy as np
import xarray as xr
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

# Constants for unit conversion
MV_PER_V = 1e3


def prepare_power_rabi_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None
) -> tuple[xr.Dataset, Optional[xr.Dataset]]:
    """
    Prepare power Rabi data for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing power Rabi data
        qubits: List of qubits to prepare data for
        ds_fit: Optional fit results dataset
        
    Returns:
        Tuple of (prepared_raw_data, prepared_fit_data)
    """
    # Create a copy to avoid modifying original
    ds_prepared = ds_raw.copy()
    
    # Standardize coordinate names and units
    ds_prepared = ds_prepared.assign_coords(
        amp_mV=ds_raw.full_amp * MV_PER_V,
        amp_prefactor=ds_raw.amp_prefactor
    )
    
    # Determine data type (I quadrature or state)
    data_source = None
    data_label = None
    if "I" in ds_raw:
        data_source = "I"
        data_label = "Rotated I quadrature [mV]"
        ds_prepared["I_mV"] = ds_raw.I * MV_PER_V
    elif "state" in ds_raw:
        data_source = "state" 
        data_label = "Qubit state"
        ds_prepared["state"] = ds_raw.state
    else:
        raise RuntimeError("Dataset must contain either 'I' or 'state' for power Rabi plotting")
    
    # Store metadata for plotting
    ds_prepared.attrs.update({
        "data_source": data_source,
        "data_label": data_label
    })
    
    # Prepare fit data if available
    ds_fit_prepared = None
    if ds_fit is not None:
        fit_data_list = []
        
        for qubit in qubits:
            qubit_id = qubit.name if hasattr(qubit, 'name') else str(qubit.grid_location)
            
            if qubit_id not in ds_fit.qubit.values:
                continue
                
            fit_qubit = ds_fit.sel(qubit=qubit_id)
            
            if hasattr(fit_qubit, 'outcome') and fit_qubit.outcome.values == "successful":
                # Generate fitted curve using oscillation function
                fitted_data = oscillation(
                    fit_qubit.amp_prefactor.data,
                    fit_qubit.fit.sel(fit_vals="a").data,
                    fit_qubit.fit.sel(fit_vals="f").data,
                    fit_qubit.fit.sel(fit_vals="phi").data,
                    fit_qubit.fit.sel(fit_vals="offset").data
                )
                
                # Create fitted data arrays with same coordinates as raw data
                if len(ds_raw.nb_of_pulses) == 1:  # 1D case
                    # Get the single pulse slice
                    raw_qubit = ds_raw.sel(qubit=qubit_id)
                    if "nb_of_pulses" in raw_qubit.dims and raw_qubit.sizes["nb_of_pulses"] == 1:
                        raw_qubit = raw_qubit.isel(nb_of_pulses=0)
                    
                    fitted_da = xr.DataArray(
                        fitted_data * MV_PER_V,
                        dims=["amp_prefactor"],
                        coords={
                            "amp_prefactor": fit_qubit.amp_prefactor,
                            "amp_mV": fit_qubit.full_amp * MV_PER_V
                        }
                    )
                    
                    fit_data_list.append(fitted_da.expand_dims("qubit").assign_coords(qubit=[qubit_id]))
        
        if fit_data_list:
            # Combine all fit data
            ds_fit_combined = xr.concat(fit_data_list, dim="qubit")
            ds_prepared["fitted_data_mV"] = ds_fit_combined
            
            # Create a separate fit dataset for metadata
            fit_arrays = {}
            qubit_ids = [qid for qid in ds_fit.qubit.values if qid in [q.name if hasattr(q, 'name') else str(q.grid_location) for q in qubits]]
            
            for qid in qubit_ids:
                fit_qubit = ds_fit.sel(qubit=qid)
                if hasattr(fit_qubit, 'outcome'):
                    if qid == qubit_ids[0]:  # Initialize arrays
                        fit_arrays["outcome"] = [fit_qubit.outcome.values]
                        if len(ds_raw.nb_of_pulses) > 1:  # 2D case
                            fit_arrays["opt_amp_prefactor"] = [float(fit_qubit.opt_amp_prefactor.values)]
                    else:  # Append to arrays
                        fit_arrays["outcome"].append(fit_qubit.outcome.values)
                        if len(ds_raw.nb_of_pulses) > 1:
                            fit_arrays["opt_amp_prefactor"].append(float(fit_qubit.opt_amp_prefactor.values))
            
            # Convert to xarray Dataset
            fit_dataset_arrays = {}
            fit_dataset_arrays["outcome"] = xr.DataArray(
                fit_arrays["outcome"],
                dims=["qubit"],
                coords={"qubit": qubit_ids}
            )
            
            if "opt_amp_prefactor" in fit_arrays:
                fit_dataset_arrays["opt_amp_prefactor"] = xr.DataArray(
                    fit_arrays["opt_amp_prefactor"],
                    dims=["qubit"],
                    coords={"qubit": qubit_ids}
                )
            
            ds_fit_prepared = xr.Dataset(fit_dataset_arrays)
    
    return ds_prepared, ds_fit_prepared


def prepare_power_rabi_raw_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon]
) -> xr.Dataset:
    """
    Prepare power Rabi raw data without fits for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing power Rabi data
        qubits: List of qubits to prepare data for
        
    Returns:
        Prepared dataset with standardized coordinate names and units
    """
    prepared_data, _ = prepare_power_rabi_data(ds_raw, qubits, None)
    return prepared_data


def validate_power_rabi_data(ds: xr.Dataset) -> bool:
    """
    Validate that the dataset has the required structure for power Rabi plotting.
    
    Args:
        ds: Dataset to validate
        
    Returns:
        True if dataset is valid, False otherwise
    """
    required_dims = {"qubit"}
    required_coords = {"full_amp", "amp_prefactor"}
    required_data_vars = set()  # Either I or state required
    
    # Check dimensions
    if not required_dims.issubset(set(ds.dims)):
        return False
        
    # Check coordinates
    if not required_coords.issubset(set(ds.coords)):
        return False
        
    # Check data variables (either I or state)
    if not ("I" in ds.data_vars or "state" in ds.data_vars):
        return False
        
    return True


def determine_plot_type(ds: xr.Dataset) -> str:
    """
    Determine the plot type based on dataset dimensions.
    
    Args:
        ds: Dataset to analyze
        
    Returns:
        "1D" for single pulse plots, "2D" for chevron plots
    """
    if "nb_of_pulses" in ds.dims and ds.sizes["nb_of_pulses"] > 1:
        return "2D"
    else:
        return "1D"


def extract_power_rabi_fit_overlays(ds_fit: xr.Dataset, qubit_id: str) -> dict:
    """
    Extract overlay parameters from power Rabi fit results.
    
    Args:
        ds_fit: Fit results dataset
        qubit_id: ID of the qubit to extract data for
        
    Returns:
        Dictionary containing overlay parameters
    """
    if qubit_id not in ds_fit.qubit.values:
        return {"outcome": "failed"}
        
    fit_qubit = ds_fit.sel(qubit=qubit_id)
    
    if hasattr(fit_qubit, 'outcome') and fit_qubit.outcome.values == "successful":
        overlays = {"outcome": "successful"}
        
        # Add optimal amplitude for 2D plots
        if hasattr(fit_qubit, 'opt_amp_prefactor'):
            overlays["opt_amp_prefactor"] = float(fit_qubit.opt_amp_prefactor.values)
            
        return overlays
    else:
        return {"outcome": "failed"}