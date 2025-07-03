"""
Data preparator for resonator spectroscopy vs flux plots (node 02b).

Transforms raw flux sweep datasets and fit results into standardized format
for the enhanced PlotConfig system.
"""

from typing import List, Optional
import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# Constants for unit conversion
GHZ_PER_HZ = 1e-9
MHZ_PER_HZ = 1e-6


def prepare_flux_sweep_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None
) -> tuple[xr.Dataset, Optional[xr.Dataset]]:
    """
    Prepare resonator spectroscopy vs flux data for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing flux sweep data
        qubits: List of qubits to prepare data for
        ds_fit: Optional fit results dataset
        
    Returns:
        Tuple of (prepared_raw_data, prepared_fit_data)
    """
    # Transpose to ensure consistent dimension order
    ds_prepared = ds_raw.transpose("qubit", "detuning", "flux_bias")
    
    # Standardize coordinate names and add derived coordinates
    freq_coord_name = "full_freq" if "full_freq" in ds_prepared else "freq_full"
    
    ds_prepared = ds_prepared.assign_coords(
        freq_GHz=ds_prepared[freq_coord_name] * GHZ_PER_HZ,
        detuning_MHz=ds_prepared.detuning * MHZ_PER_HZ
    )
    
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
                    "idle_offset": float(fit_qubit.fit_results.idle_offset.values),
                    "flux_min": float(fit_qubit.fit_results.flux_min.values), 
                    "sweet_spot_frequency_GHz": float(fit_qubit.fit_results.sweet_spot_frequency.values) * GHZ_PER_HZ,
                    "outcome": "successful"
                }
            else:
                fit_data[qubit_id] = {"outcome": "failed"}
        
        if fit_data:
            # Convert to xarray Dataset
            qubit_ids = list(fit_data.keys())
            fit_arrays = {}
            
            for key in ["idle_offset", "flux_min", "sweet_spot_frequency_GHz"]:
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


def prepare_flux_sweep_raw_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon]
) -> xr.Dataset:
    """
    Prepare flux sweep raw data without fits for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing flux sweep data
        qubits: List of qubits to prepare data for
        
    Returns:
        Prepared dataset with standardized coordinate names and units
    """
    prepared_data, _ = prepare_flux_sweep_data(ds_raw, qubits, None)
    return prepared_data


def validate_flux_sweep_data(ds: xr.Dataset) -> bool:
    """
    Validate that the dataset has the required structure for flux sweep plotting.
    
    Args:
        ds: Dataset to validate
        
    Returns:
        True if dataset is valid, False otherwise
    """
    required_dims = {"qubit", "detuning", "flux_bias"}
    required_coords = {"flux_bias", "detuning", "attenuated_current"}
    required_data_vars = {"IQ_abs"}
    
    # Check dimensions
    if not required_dims.issubset(set(ds.dims)):
        return False
        
    # Check coordinates  
    if not required_coords.issubset(set(ds.coords)):
        return False
        
    # Check data variables
    if not required_data_vars.issubset(set(ds.data_vars)):
        return False
        
    # Check for frequency coordinate
    if not ("full_freq" in ds or "freq_full" in ds):
        return False
        
    return True