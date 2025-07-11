from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import xarray as xr

from ..analysis import lorentzian_dip

# Constants for unit conversion  
GHZ_PER_HZ = 1e-9
MHZ_PER_HZ = 1e-6
MV_PER_V = 1e3


class DataPreparator(ABC):
    """
    Abstract base class for data preparators.

    The role of a preparator is to transform an experiment-specific xarray.Dataset
    into a standardized pair of datasets that plotting functions can consume.
    This de-couples the plotting from the specific data structures of any given experiment.

    To create a new preparator for an experiment, create a new class that
    inherits from this one and implement the `prepare` method.
    """

    def __init__(self, ds_raw: xr.Dataset, ds_fit: Optional[xr.Dataset] = None, **kwargs: Any):
        """
        Initializes the preparator with raw and optional fitted datasets.

        Args:
            ds_raw: The raw dataset from the experiment.
            ds_fit: The fitted dataset, if available.
            **kwargs: Additional keyword arguments for specific preparators (e.g., 'qubits').
        """
        self.ds_raw = ds_raw
        self.ds_fit = ds_fit
        self._kwargs = kwargs

    @abstractmethod
    def prepare(self) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
        """
        Transforms the dataset(s) into a plot-ready tuple of datasets.

        This method must be implemented by the subclass.

        Returns:
            A tuple containing the processed raw and fit datasets.
        """
        raise NotImplementedError


class ResonatorSpectroscopyPreparator(DataPreparator):
    """Prepares resonator spectroscopy data for plotting."""

    def prepare(self) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
        """
        Prepares resonator spectroscopy datasets for plotting.

        This method enriches the raw and fit datasets with additional, plot-ready
        fields. It uses Lorentzian model fits if they are available.
        """
        # --- Raw Data Preparation ---
        ds_raw_processed = self.ds_raw.copy()
        if "full_freq" in ds_raw_processed:
            ds_raw_processed["full_freq_GHz"] = ds_raw_processed.full_freq / 1e9
        if "detuning" in ds_raw_processed.dims:
            ds_raw_processed.coords["detuning_MHz"] = ("detuning", ds_raw_processed.detuning.values / 1e6)
        if "IQ_abs" in ds_raw_processed:
            ds_raw_processed["IQ_abs_mV"] = ds_raw_processed.IQ_abs * 1e3
        if "phase" in ds_raw_processed:
            ds_raw_processed["phase"] = ds_raw_processed.phase

        # --- Fit Data Preparation ---
        ds_fit_processed = None
        if self.ds_fit is not None:
            ds_fit_processed = self.ds_fit.copy()

            # Only Lorentzian logic, no S21 logic
            if "fitted_curve" not in ds_fit_processed and all(
                    p in self.ds_fit for p in ["amplitude", "position", "width", "base_line"]):
                required_params = ["amplitude", "position", "width", "base_line", "outcome"]
                if all(p in ds_fit_processed for p in required_params):
                    all_curves = xr.DataArray(
                        np.nan,
                        coords=[ds_fit_processed.qubit, self.ds_raw.detuning],
                        dims=["qubit", "detuning"]
                    )
                    for qubit_id in ds_fit_processed.qubit.values:
                        fit_q = ds_fit_processed.sel(qubit=qubit_id)
                        if fit_q.outcome.values == "successful":
                            curve = lorentzian_dip(
                                self.ds_raw.detuning.values,
                                float(fit_q.amplitude.values),
                                float(fit_q.position.values),
                                float(fit_q.width.values) / 2,
                                float(fit_q.base_line.mean().values),
                            )
                            all_curves.loc[dict(qubit=qubit_id)] = curve
                    ds_fit_processed["fitted_curve"] = all_curves

            # Add other plot-ready fields to the fit dataset
            if "full_freq" not in ds_fit_processed and "full_freq" in ds_raw_processed:
                ds_fit_processed["full_freq"] = ds_raw_processed.full_freq
            if "full_freq" in ds_fit_processed:
                ds_fit_processed["full_freq_GHz"] = ds_fit_processed.full_freq / 1e9
            if "fitted_curve" in ds_fit_processed:
                ds_fit_processed["fitted_data_mV"] = ds_fit_processed.fitted_curve * 1e3

        return ds_raw_processed, ds_fit_processed


class PowerRabiPreparator(DataPreparator):
    """Prepares power Rabi data for plotting."""

    def prepare(self) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
        """Prepare power Rabi data for standardized plotting."""
        # The 'qubits' argument from the original function is available in self._kwargs if needed
        # qubits = self._kwargs.get("qubits")

        # Create a copy to avoid modifying original
        ds_prepared = self.ds_raw.copy()

        # Add derived coordinates for plotting
        ds_prepared = ds_prepared.assign_coords(
            amp_mV=self.ds_raw.full_amp * MV_PER_V,
            amp_prefactor=self.ds_raw.amp_prefactor
        )

        # Determine data type and convert if needed
        if "I" in self.ds_raw:
            ds_prepared["I_mV"] = self.ds_raw.I * MV_PER_V
            data_source = "I"
            data_label = "Rotated I quadrature [mV]"
        elif "state" in self.ds_raw:
            data_source = "state"
            data_label = "Qubit state"
        else:
            raise RuntimeError("Dataset must contain either 'I' or 'state' for power Rabi plotting")

        # Store metadata for plotting
        ds_prepared.attrs.update({
            "data_source": data_source,
            "data_label": data_label
        })

        # Process fit data if available
        ds_fit_prepared = None
        if self.ds_fit is not None:
            ds_fit_prepared = self.ds_fit.copy()

            # Add any necessary derived coordinates for fit data
            if "amp_mV" not in ds_fit_prepared.coords and "full_amp" in ds_fit_prepared:
                ds_fit_prepared = ds_fit_prepared.assign_coords(
                    amp_mV=ds_fit_prepared["full_amp"] * MV_PER_V
                )

        return ds_prepared, ds_fit_prepared


class ResonatorSpectroscopyVsAmplitudePreparator(DataPreparator):
    """Prepares resonator spectroscopy vs amplitude data for plotting."""

    def prepare(self) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
        """Prepare resonator spectroscopy vs amplitude data for standardized plotting."""
        # The 'qubits' argument from the original function is available in self._kwargs if needed
        # qubits = self._kwargs.get("qubits")

        # Create a copy to avoid modifying original
        ds_prepared = self.ds_raw.copy()

        # Add derived coordinates for plotting
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
                ds_prepared["IQ_abs_norm"] = ds_prepared["IQ_abs"]

        # Process fit data if available
        ds_fit_prepared = None
        if self.ds_fit is not None:
            ds_fit_prepared = self.ds_fit.copy()

            # Add any necessary derived coordinates for fit data
            if "freq_GHz" not in ds_fit_prepared.coords and freq_coord_name in ds_fit_prepared:
                ds_fit_prepared = ds_fit_prepared.assign_coords(
                    freq_GHz=ds_fit_prepared[freq_coord_name] * GHZ_PER_HZ
                )

        return ds_prepared, ds_fit_prepared


class ResonatorSpectroscopyVsFluxPreparator(DataPreparator):
    """Prepares resonator spectroscopy vs flux data for plotting."""

    def prepare(self) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
        """Prepare resonator spectroscopy vs flux data for standardized plotting."""
        # The 'qubits' argument from the original function is available in self._kwargs if needed
        # qubits = self._kwargs.get("qubits")

        # Create a copy to avoid modifying original
        ds_prepared = self.ds_raw.copy()

        # Add derived coordinates for plotting
        freq_coord_name = "full_freq" if "full_freq" in ds_prepared else "freq_full"

        ds_prepared = ds_prepared.assign_coords(
            freq_GHz=ds_prepared[freq_coord_name] * GHZ_PER_HZ,
            detuning_MHz=ds_prepared.detuning * MHZ_PER_HZ
        )

        # Process fit data if available
        ds_fit_prepared = None
        if self.ds_fit is not None:
            ds_fit_prepared = self.ds_fit.copy()

            # Add any necessary derived coordinates for fit data
            if "freq_GHz" not in ds_fit_prepared.coords and freq_coord_name in ds_fit_prepared:
                ds_fit_prepared = ds_fit_prepared.assign_coords(
                    freq_GHz=ds_fit_prepared[freq_coord_name] * GHZ_PER_HZ
                )

        return ds_prepared, ds_fit_prepared       