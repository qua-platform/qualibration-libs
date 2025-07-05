"""
Data validation utilities for the plotting framework.

This module provides comprehensive validation of input datasets to ensure
they meet the requirements for plotting and provide helpful error messages
when validation fails.
"""

from typing import List, Optional, Dict, Any, Tuple
import xarray as xr
import numpy as np
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from ..configs import TraceConfig, HeatmapTraceConfig


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    
    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


class DataValidator:
    """Comprehensive data validation for plotting datasets."""
    
    def validate_dataset(
        self,
        ds: xr.Dataset,
        qubits: List[AnyTransmon],
        is_fit_data: bool = False
    ) -> None:
        """
        Validate that a dataset meets requirements for plotting.
        
        Args:
            ds: Dataset to validate
            qubits: List of qubits that should be present
            is_fit_data: Whether this is fit data (different requirements)
            
        Raises:
            ValidationError: If validation fails
        """
        self._validate_basic_structure(ds)
        self._validate_qubit_dimension(ds, qubits)
        
        if not is_fit_data:
            self._validate_raw_data_requirements(ds, qubits)
        else:
            self._validate_fit_data_requirements(ds, qubits)
    
    def validate_trace_config(
        self,
        ds: xr.Dataset,
        trace_config: TraceConfig,
        qubit_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a trace configuration can be used with a dataset.
        
        Args:
            ds: Dataset to validate against
            trace_config: Trace configuration to validate
            qubit_id: Specific qubit ID to check
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required data sources
        required_sources = [trace_config.x_source, trace_config.y_source]
        if isinstance(trace_config, HeatmapTraceConfig) and trace_config.z_source:
            required_sources.append(trace_config.z_source)
        
        missing_sources = []
        for source in required_sources:
            if source not in ds:
                missing_sources.append(source)
                errors.append(f"Missing required data source: '{source}'")
        
        # Check custom data sources
        for source in trace_config.custom_data_sources:
            if source not in ds:
                errors.append(f"Missing custom data source: '{source}'")
        
        # Check condition source if specified
        if trace_config.condition_source and trace_config.condition_source not in ds:
            errors.append(f"Missing condition source: '{trace_config.condition_source}'")
        
        # Provide suggestions for missing sources
        if missing_sources:
            available_sources = list(ds.data_vars.keys()) + list(ds.coords.keys())
            for missing in missing_sources:
                suggestions = self._suggest_alternatives(available_sources, missing)
                if suggestions:
                    errors.append(f"  Suggestions for '{missing}': {', '.join(suggestions)}")
        
        return len(errors) == 0, errors
    
    def _validate_basic_structure(self, ds: xr.Dataset) -> None:
        """Validate basic dataset structure."""
        if not isinstance(ds, xr.Dataset):
            raise ValidationError("Input must be an xarray Dataset")
        
        if len(ds.dims) == 0:
            raise ValidationError("Dataset has no dimensions")
        
        if len(ds.data_vars) == 0:
            raise ValidationError("Dataset has no data variables")
    
    def _validate_qubit_dimension(self, ds: xr.Dataset, qubits: List[AnyTransmon]) -> None:
        """Validate qubit dimension and names."""
        if 'qubit' not in ds.dims:
            raise ValidationError(
                "Dataset missing 'qubit' dimension",
                suggestions=["Ensure dataset was created with proper qubit coordinates"]
            )
        
        # Get expected qubit names
        expected_qubits = [q.name for q in qubits]
        dataset_qubits = list(ds.coords['qubit'].values)
        
        # Check for missing qubits
        missing_qubits = set(expected_qubits) - set(dataset_qubits)
        if missing_qubits:
            raise ValidationError(
                f"Missing qubits in dataset: {list(missing_qubits)}",
                suggestions=[f"Available qubits: {dataset_qubits}"]
            )
        
        # Check for extra qubits (warning, not error)
        extra_qubits = set(dataset_qubits) - set(expected_qubits)
        if extra_qubits:
            print(f"Warning: Dataset contains extra qubits: {list(extra_qubits)}")
    
    def _validate_raw_data_requirements(self, ds: xr.Dataset, qubits: List[AnyTransmon]) -> None:
        """Validate requirements specific to raw experimental data."""
        
        # Check for IQ data or state data
        has_iq_data = self._check_iq_data_presence(ds, qubits)
        has_state_data = self._check_state_data_presence(ds, qubits)
        
        if not (has_iq_data or has_state_data):
            raise ValidationError(
                "Dataset missing both IQ data (I1, Q1, ...) and state data (state1, state2, ...)",
                suggestions=[
                    "Ensure QUA program properly saved I/Q or state measurements",
                    "Check stream processing in QUA program"
                ]
            )
        
        # Validate data shapes are consistent
        self._validate_data_shapes(ds, qubits)
    
    def _validate_fit_data_requirements(self, ds: xr.Dataset, qubits: List[AnyTransmon]) -> None:
        """Validate requirements specific to fit data."""
        
        # Check for outcome field
        if 'outcome' not in ds:
            print("Warning: Fit dataset missing 'outcome' field - overlays may not work correctly")
        
        # Check that fit data has reasonable structure
        expected_qubits = [q.name for q in qubits]
        for qubit_name in expected_qubits:
            if qubit_name in ds.coords['qubit'].values:
                qubit_ds = ds.sel(qubit=qubit_name)
                
                # Check for NaN values in critical fields
                for var_name, var_data in qubit_ds.data_vars.items():
                    if np.isscalar(var_data.values) and np.isnan(var_data.values):
                        print(f"Warning: NaN value in fit parameter '{var_name}' for qubit '{qubit_name}'")
    
    def _check_iq_data_presence(self, ds: xr.Dataset, qubits: List[AnyTransmon]) -> bool:
        """Check if dataset contains IQ measurement data."""
        # Pattern 1: Node 02a style - I, Q directly (with qubit dimension)
        if "I" in ds.data_vars and "Q" in ds.data_vars:
            return True
            
        # Pattern 2: Nodes 02b, 02c, 04b style - I1, Q1, I2, Q2, etc.
        for i, qubit in enumerate(qubits, 1):
            i_var = f"I{i}"
            q_var = f"Q{i}"
            if i_var in ds.data_vars and q_var in ds.data_vars:
                return True
                
        # Pattern 3: Look for any IQ-like variables
        data_vars = list(ds.data_vars.keys())
        has_i_like = any(var.startswith('I') or 'I_' in var for var in data_vars)
        has_q_like = any(var.startswith('Q') or 'Q_' in var for var in data_vars)
        
        return has_i_like and has_q_like
    
    def _check_state_data_presence(self, ds: xr.Dataset, qubits: List[AnyTransmon]) -> bool:
        """Check if dataset contains state discrimination data."""
        # Pattern 1: Generic state variable
        if "state" in ds.data_vars:
            return True
            
        # Pattern 2: state1, state2, etc.
        for i, qubit in enumerate(qubits, 1):
            state_var = f"state{i}"
            if state_var in ds.data_vars:
                return True
                
        # Pattern 3: Look for any state-like variables
        data_vars = list(ds.data_vars.keys())
        has_state_like = any('state' in var.lower() for var in data_vars)
        
        return has_state_like
    
    def _validate_data_shapes(self, ds: xr.Dataset, qubits: List[AnyTransmon]) -> None:
        """Validate that data variables have consistent shapes."""
        
        # Get expected shape from dataset dimensions - be flexible about dimension order
        expected_dims = set(ds.dims)
        expected_sizes = ds.sizes
        
        # Check each data variable
        for var_name, var_data in ds.data_vars.items():
            var_dims = set(var_data.dims)
            
            # Check that dimensions are a subset of expected (allows for variables with fewer dims)
            if not var_dims.issubset(expected_dims):
                unknown_dims = var_dims - expected_dims
                raise ValidationError(
                    f"Data variable '{var_name}' has unknown dimensions: {unknown_dims}"
                )
            
            # Check that dimension sizes match where applicable
            for dim in var_dims:
                if dim in expected_sizes:
                    if var_data.sizes[dim] != expected_sizes[dim]:
                        raise ValidationError(
                            f"Data variable '{var_name}' has mismatched size for dimension '{dim}': "
                            f"got {var_data.sizes[dim]}, expected {expected_sizes[dim]}"
                        )
    
    def _suggest_alternatives(self, available_sources: List[str], missing_source: str) -> List[str]:
        """Suggest alternative data sources for missing ones."""
        suggestions = []
        keywords = missing_source.lower().split('_')
        
        for source in available_sources:
            source_lower = source.lower()
            # Simple matching based on shared keywords
            if any(keyword in source_lower for keyword in keywords):
                suggestions.append(source)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def get_dataset_summary(self, ds: xr.Dataset) -> Dict[str, Any]:
        """Get comprehensive summary of dataset for debugging."""
        
        summary = {
            "dimensions": dict(ds.dims),
            "data_variables": list(ds.data_vars.keys()),
            "coordinates": list(ds.coords.keys()),
            "total_size": ds.nbytes,
            "memory_usage_mb": ds.nbytes / (1024 * 1024),
        }
        
        # Add variable-specific information
        var_info = {}
        for var_name, var_data in ds.data_vars.items():
            var_info[var_name] = {
                "dtype": str(var_data.dtype),
                "shape": var_data.shape,
                "has_nan": bool(np.isnan(var_data.values).any()) if var_data.dtype.kind == 'f' else False,
                "range": self._get_value_range(var_data.values) if var_data.dtype.kind in ['f', 'i'] else None,
            }
        
        summary["variable_info"] = var_info
        
        return summary
    
    def _get_value_range(self, values: np.ndarray) -> Optional[Tuple[float, float]]:
        """Get value range for numeric data."""
        try:
            finite_values = values[np.isfinite(values)]
            if len(finite_values) > 0:
                return float(np.min(finite_values)), float(np.max(finite_values))
        except:
            pass
        return None


# ===== CONVENIENCE FUNCTIONS =====

def validate_dataset(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    is_fit_data: bool = False
) -> None:
    """
    Validate dataset using default validator.
    
    Args:
        ds: Dataset to validate
        qubits: List of qubits
        is_fit_data: Whether this is fit data
        
    Raises:
        ValidationError: If validation fails
    """
    validator = DataValidator()
    validator.validate_dataset(ds, qubits, is_fit_data)


def validate_trace_config(
    ds: xr.Dataset,
    trace_config: TraceConfig,
    qubit_id: str
) -> Tuple[bool, List[str]]:
    """
    Validate trace configuration against dataset.
    
    Args:
        ds: Dataset to validate against
        trace_config: Trace configuration
        qubit_id: Qubit ID to check
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = DataValidator()
    return validator.validate_trace_config(ds, trace_config, qubit_id)


def get_dataset_summary(ds: xr.Dataset) -> Dict[str, Any]:
    """Get comprehensive dataset summary for debugging."""
    validator = DataValidator()
    return validator.get_dataset_summary(ds)