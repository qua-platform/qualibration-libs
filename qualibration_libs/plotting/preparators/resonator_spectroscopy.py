"""
Data preparator for resonator spectroscopy plots (node 02a).

Transforms raw spectroscopy datasets and fit results into standardized format
for the enhanced PlotConfig system.
"""

from typing import List, Optional
import numpy as np
import xarray as xr
from qualang_tools.units import unit
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def prepare_spectroscopy_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon], 
    ds_fit: Optional[xr.Dataset] = None
) -> xr.Dataset:
    """
    Prepare resonator spectroscopy data for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing IQ data
        qubits: List of qubits to prepare data for
        ds_fit: Optional fit results dataset
        
    Returns:
        Prepared dataset with standardized coordinate names and units
    """
    # Create a copy to avoid modifying original
    ds_prepared = ds_raw.copy()
    
    # Standardize coordinate names and units
    ds_prepared = ds_prepared.assign_coords(
        full_freq_GHz=ds_raw.full_freq / u.GHz,
        detuning_MHz=ds_raw.detuning / u.MHz
    )
    
    # Convert IQ_abs to mV
    ds_prepared["IQ_abs_mV"] = ds_raw.IQ_abs / u.mV
    
    # Add fit data if available
    if ds_fit is not None:
        fit_data_list = []
        
        for qubit in qubits:
            qubit_id = qubit.name if hasattr(qubit, 'name') else str(qubit.grid_location)
            
            # Check if qubit exists in fit dataset
            if qubit_id not in ds_fit.qubit.values:
                continue
                
            fit_qubit = ds_fit.sel(qubit=qubit_id)
            
            if hasattr(fit_qubit, 'outcome') and fit_qubit.outcome.values == "successful":
                # Generate fitted curve using lorentzian_dip
                fitted_data = lorentzian_dip(
                    ds_raw.detuning.values,
                    float(fit_qubit.amplitude.values),
                    float(fit_qubit.position.values), 
                    float(fit_qubit.width.values) / 2,
                    float(fit_qubit.base_line.mean().values)
                )
                
                # Create fitted data arrays with same coordinates as raw data
                fitted_da = xr.DataArray(
                    fitted_data / u.mV,
                    dims=["detuning"],
                    coords={
                        "detuning": ds_raw.detuning,
                        "full_freq_GHz": ds_raw.full_freq.sel(qubit=qubit_id) / u.GHz,
                        "detuning_MHz": ds_raw.detuning / u.MHz
                    }
                )
                
                fit_data_list.append(fitted_da.expand_dims("qubit").assign_coords(qubit=[qubit_id]))
        
        if fit_data_list:
            # Combine all fit data
            ds_fit_combined = xr.concat(fit_data_list, dim="qubit")
            ds_prepared["fitted_data_mV"] = ds_fit_combined
    
    return ds_prepared


def prepare_spectroscopy_phase_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon]
) -> xr.Dataset:
    """
    Prepare resonator spectroscopy phase data for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset containing phase data
        qubits: List of qubits to prepare data for
        
    Returns:
        Prepared dataset with standardized coordinate names and units
    """
    # Create a copy to avoid modifying original  
    ds_prepared = ds_raw.copy()
    
    # Standardize coordinate names and units
    ds_prepared = ds_prepared.assign_coords(
        full_freq_GHz=ds_raw.full_freq / u.GHz,
        detuning_MHz=ds_raw.detuning / u.MHz
    )
    
    return ds_prepared