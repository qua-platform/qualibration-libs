"""
Data preparator for resonator spectroscopy vs amplitude plots (node 02c).

Transforms raw amplitude sweep datasets and fit results into standardized format
for the enhanced PlotConfig system.
"""

from typing import List, Optional
import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# Constants for unit conversion
GHZ_PER_HZ = 1e-9
MHZ_PER_HZ = 1e-6


def prepare_amplitude_sweep_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None
) -> tuple[xr.Dataset, Optional[xr.Dataset]]:
    """
    Prepare resonator spectroscopy vs amplitude data for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing amplitude sweep data
        qubits: List of qubits to prepare data for
        ds_fit: Optional fit results dataset
        
    Returns:
        Tuple of (prepared_raw_data, prepared_fit_data)
    """
    # Transpose to ensure consistent dimension order (qubit, detuning, power)
    ds_prepared = ds_raw.transpose("qubit", "detuning", "power")
    
    # Standardize coordinate names and add derived coordinates
    freq_coord_name = "full_freq" if "full_freq" in ds_prepared else "freq_full"
    power_coord_name = "power" if "power" in ds_prepared.coords else "power_dbm"
    
    ds_prepared = ds_prepared.assign_coords(
        freq_GHz=ds_prepared[freq_coord_name] * GHZ_PER_HZ,
        detuning_MHz=ds_prepared.detuning * MHZ_PER_HZ,
        power_dbm=ds_prepared[power_coord_name]
    )
    
    # Ensure IQ_abs_norm exists for plotting
    if "IQ_abs_norm" not in ds_prepared:
        if "IQ_abs" in ds_prepared:
            # Create normalized version if not available
            ds_prepared["IQ_abs_norm"] = ds_prepared["IQ_abs"]
    
    # Prepare fit data if available
    ds_fit_prepared = None
    if ds_fit is not None:
        fit_data = {}
        
        for qubit in qubits:
            qubit_id = qubit.name if hasattr(qubit, 'name') else str(qubit.grid_location)
            
            if qubit_id not in ds_fit.qubit.values:
                continue
                
            fit_qubit = ds_fit.sel(qubit=qubit_id)
            
            if hasattr(fit_qubit, 'outcome') and fit_qubit.outcome.values == "successful":
                # Extract fit parameters for overlays
                fit_data[qubit_id] = {
                    "res_freq_GHz": float(fit_qubit.res_freq.values) * GHZ_PER_HZ,
                    "optimal_power": float(fit_qubit.optimal_power.values),
                    "outcome": "successful"
                }
            else:
                fit_data[qubit_id] = {"outcome": "failed"}
        
        if fit_data:
            # Convert to xarray Dataset
            qubit_ids = list(fit_data.keys())
            fit_arrays = {}
            
            for key in ["res_freq_GHz", "optimal_power"]:
                if all(key in fit_data[qid] for qid in qubit_ids):
                    fit_arrays[key] = xr.DataArray(
                        [fit_data[qid].get(key, np.nan) for qid in qubit_ids],
                        dims=["qubit"],
                        coords={"qubit": qubit_ids}
                    )
            
            # Add outcome array
            fit_arrays["outcome"] = xr.DataArray(
                [fit_data[qid]["outcome"] for qid in qubit_ids],
                dims=["qubit"],
                coords={"qubit": qubit_ids}
            )
            
            ds_fit_prepared = xr.Dataset(fit_arrays)
    
    return ds_prepared, ds_fit_prepared


def prepare_amplitude_sweep_raw_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon]
) -> xr.Dataset:
    """
    Prepare amplitude sweep raw data without fits for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing amplitude sweep data
        qubits: List of qubits to prepare data for
        
    Returns:
        Prepared dataset with standardized coordinate names and units
    """
    prepared_data, _ = prepare_amplitude_sweep_data(ds_raw, qubits, None)
    return prepared_data


def validate_amplitude_sweep_data(ds: xr.Dataset) -> bool:
    """
    Validate that the dataset has the required structure for amplitude sweep plotting.
    
    Args:
        ds: Dataset to validate
        
    Returns:
        True if dataset is valid, False otherwise
    """
    required_dims = {"qubit", "detuning", "power"}
    required_coords = {"detuning"}
    required_data_vars = {"IQ_abs_norm"}
    
    # Check dimensions
    if not required_dims.issubset(set(ds.dims)):
        return False
        
    # Check coordinates
    if not required_coords.issubset(set(ds.coords)):
        return False
        
    # Check data variables (allow fallback to IQ_abs)
    if not ("IQ_abs_norm" in ds.data_vars or "IQ_abs" in ds.data_vars):
        return False
        
    # Check for frequency coordinate
    if not ("full_freq" in ds or "freq_full" in ds):
        return False
        
    # Check for power coordinate
    if not ("power" in ds.coords or "power_dbm" in ds.coords):
        return False
        
    return True


def extract_amplitude_fit_overlays(ds_fit: xr.Dataset, qubit_id: str) -> dict:
    """
    Extract overlay parameters from amplitude sweep fit results.
    
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
        return {
            "res_freq": float(fit_qubit.res_freq.values),
            "res_freq_GHz": float(fit_qubit.res_freq.values) * GHZ_PER_HZ,
            "optimal_power": float(fit_qubit.optimal_power.values),
            "outcome": "successful"
        }
    else:
        return {"outcome": "failed"}