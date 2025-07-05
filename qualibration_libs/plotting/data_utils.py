"""Data manipulation utilities for the plotting module.

This module provides centralized data extraction, transformation, and validation
utilities used across all plotting engines to ensure consistent data handling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import xarray as xr

from .configs.constants import CoordinateNames, PlotConstants
from .exceptions import DataSourceError, ValidationError

logger = logging.getLogger(__name__)


class DataExtractor:
    """Utilities for extracting data from xarray datasets."""
    
    @staticmethod
    def extract_qubit_data(
        ds: xr.Dataset,
        qubit_id: str,
        coordinate_name: str = CoordinateNames.QUBIT
    ) -> xr.Dataset:
        """Extract data for a specific qubit from dataset.
        
        Args:
            ds: Dataset containing multi-qubit data
            qubit_id: ID of the qubit to extract
            coordinate_name: Name of the qubit coordinate (default: "qubit")
            
        Returns:
            Dataset containing only data for the specified qubit
            
        Raises:
            DataSourceError: If qubit coordinate not found or qubit_id not in dataset
        """
        if coordinate_name not in ds.coords:
            raise DataSourceError(
                f"Coordinate '{coordinate_name}' not found in dataset",
                context={"available_coords": list(ds.coords)},
                suggestions=["Check if dataset has qubit dimension", "Verify coordinate name"]
            )
            
        if qubit_id not in ds[coordinate_name].values:
            raise DataSourceError(
                f"Qubit '{qubit_id}' not found in dataset",
                context={
                    "available_qubits": list(ds[coordinate_name].values),
                    "requested_qubit": qubit_id
                },
                suggestions=["Check qubit ID spelling", "Verify qubit exists in dataset"]
            )
            
        return ds.sel({coordinate_name: qubit_id})
    
    @staticmethod
    def get_coordinate_values(
        ds: xr.Dataset,
        coord_name: str,
        as_numpy: bool = True
    ) -> Union[np.ndarray, xr.DataArray]:
        """Safely extract coordinate values from dataset.
        
        Args:
            ds: Dataset to extract from
            coord_name: Name of the coordinate
            as_numpy: If True, return numpy array; else return DataArray
            
        Returns:
            Coordinate values as numpy array or DataArray
            
        Raises:
            DataSourceError: If coordinate not found
        """
        if coord_name not in ds.coords:
            raise DataSourceError(
                f"Coordinate '{coord_name}' not found",
                context={"available_coords": list(ds.coords)},
                suggestions=[f"Use one of: {', '.join(ds.coords)}"]
            )
            
        values = ds[coord_name]
        return values.values if as_numpy else values
    
    @staticmethod
    def check_data_source_exists(
        ds: xr.Dataset,
        source_name: str,
        source_type: str = "any"
    ) -> bool:
        """Check if a data source exists in dataset.
        
        Args:
            ds: Dataset to check
            source_name: Name of the source to check
            source_type: Type to check - "coord", "data_var", or "any"
            
        Returns:
            True if source exists, False otherwise
        """
        if source_type == "coord":
            return source_name in ds.coords
        elif source_type == "data_var":
            return source_name in ds.data_vars
        else:  # "any"
            return source_name in ds.coords or source_name in ds.data_vars
    
    @staticmethod
    def get_data_array_safe(
        ds: xr.Dataset,
        var_name: str,
        default: Optional[Any] = None
    ) -> Optional[xr.DataArray]:
        """Safely get a data array from dataset.
        
        Args:
            ds: Dataset to extract from
            var_name: Variable name to extract
            default: Default value if not found
            
        Returns:
            DataArray if found, default value otherwise
        """
        if var_name in ds.data_vars:
            return ds[var_name]
        elif var_name in ds.coords:
            return ds[var_name]
        else:
            return default


class UnitConverter:
    """Centralized unit conversion utilities."""
    
    @staticmethod
    def hz_to_ghz(freq_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert frequency from Hz to GHz.
        
        Args:
            freq_hz: Frequency in Hz
            
        Returns:
            Frequency in GHz
        """
        return freq_hz * PlotConstants.GHZ_PER_HZ
    
    @staticmethod
    def hz_to_mhz(freq_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert frequency from Hz to MHz.
        
        Args:
            freq_hz: Frequency in Hz
            
        Returns:
            Frequency in MHz
        """
        return freq_hz * PlotConstants.MHZ_PER_HZ
    
    @staticmethod
    def v_to_mv(voltage_v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert voltage from V to mV.
        
        Args:
            voltage_v: Voltage in V
            
        Returns:
            Voltage in mV
        """
        return voltage_v * PlotConstants.MV_PER_V
    
    @staticmethod
    def mv_to_v(voltage_mv: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert voltage from mV to V.
        
        Args:
            voltage_mv: Voltage in mV
            
        Returns:
            Voltage in V
        """
        return voltage_mv / PlotConstants.MV_PER_V
    
    @staticmethod
    def linear_to_dbm(
        power_linear: Union[float, np.ndarray],
        impedance: float = 50.0
    ) -> Union[float, np.ndarray]:
        """Convert linear power to dBm.
        
        Args:
            power_linear: Power in linear units (W)
            impedance: Reference impedance (default: 50 ohms)
            
        Returns:
            Power in dBm
        """
        # Convert to mW then to dBm
        power_mw = power_linear * 1000
        return 10 * np.log10(power_mw)
    
    @staticmethod
    def apply_unit_conversion(
        data: Union[xr.DataArray, np.ndarray],
        conversion_type: str,
        **kwargs
    ) -> Union[xr.DataArray, np.ndarray]:
        """Apply a unit conversion to data.
        
        Args:
            data: Data to convert
            conversion_type: Type of conversion ("hz_to_ghz", "v_to_mv", etc.)
            **kwargs: Additional arguments for specific conversions
            
        Returns:
            Converted data (same type as input)
            
        Raises:
            ValueError: If conversion type not recognized
        """
        conversions = {
            "hz_to_ghz": UnitConverter.hz_to_ghz,
            "hz_to_mhz": UnitConverter.hz_to_mhz,
            "v_to_mv": UnitConverter.v_to_mv,
            "linear_to_dbm": UnitConverter.linear_to_dbm,
        }
        
        if conversion_type not in conversions:
            raise ValueError(
                f"Unknown conversion type: {conversion_type}. "
                f"Available: {', '.join(conversions.keys())}"
            )
            
        converter = conversions[conversion_type]
        
        if isinstance(data, xr.DataArray):
            return data.copy(data=converter(data.values, **kwargs))
        else:
            return converter(data, **kwargs)


class DataValidator:
    """Data validation and checking utilities."""
    
    @staticmethod
    def validate_array_shape(
        arr: np.ndarray,
        expected_ndim: Optional[int] = None,
        expected_shape: Optional[Tuple[int, ...]] = None,
        name: str = "array"
    ) -> None:
        """Validate array dimensions and shape.
        
        Args:
            arr: Array to validate
            expected_ndim: Expected number of dimensions
            expected_shape: Expected shape (None values are ignored)
            name: Name for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        if expected_ndim is not None and arr.ndim != expected_ndim:
            raise ValidationError(
                f"{name} has wrong number of dimensions",
                context={
                    "expected_ndim": expected_ndim,
                    "actual_ndim": arr.ndim,
                    "shape": arr.shape
                }
            )
            
        if expected_shape is not None:
            for i, (expected, actual) in enumerate(zip(expected_shape, arr.shape)):
                if expected is not None and expected != actual:
                    raise ValidationError(
                        f"{name} has wrong shape in dimension {i}",
                        context={
                            "expected_shape": expected_shape,
                            "actual_shape": arr.shape,
                            "dimension": i
                        }
                    )
    
    @staticmethod
    def check_for_nans(
        data: Union[np.ndarray, xr.DataArray],
        raise_on_all_nan: bool = True
    ) -> Dict[str, Any]:
        """Check for NaN values in data.
        
        Args:
            data: Data to check
            raise_on_all_nan: If True, raise error if all values are NaN
            
        Returns:
            Dictionary with NaN statistics
            
        Raises:
            ValidationError: If all values are NaN and raise_on_all_nan is True
        """
        values = data.values if isinstance(data, xr.DataArray) else data
        
        has_nans = np.any(np.isnan(values))
        all_nans = np.all(np.isnan(values))
        nan_count = np.sum(np.isnan(values))
        nan_fraction = nan_count / values.size if values.size > 0 else 0
        
        if all_nans and raise_on_all_nan:
            raise ValidationError(
                "All values are NaN",
                context={"shape": values.shape, "dtype": values.dtype},
                suggestions=["Check data processing pipeline", "Verify input data validity"]
            )
            
        return {
            "has_nans": has_nans,
            "all_nans": all_nans,
            "nan_count": nan_count,
            "nan_fraction": nan_fraction
        }
    
    @staticmethod
    def validate_fit_success(
        ds_fit: Optional[xr.Dataset],
        qubit_id: Optional[str] = None
    ) -> bool:
        """Check if fit was successful.
        
        Args:
            ds_fit: Fit results dataset
            qubit_id: Optional qubit ID for more specific checking
            
        Returns:
            True if fit was successful
        """
        if ds_fit is None:
            return False
            
        # Check for outcome attribute
        if hasattr(ds_fit, CoordinateNames.OUTCOME):
            outcome = getattr(ds_fit, CoordinateNames.OUTCOME)
            return getattr(outcome, "values", None) == CoordinateNames.SUCCESSFUL
            
        # Check for outcome variable
        if CoordinateNames.OUTCOME in ds_fit.data_vars:
            return ds_fit[CoordinateNames.OUTCOME].values == CoordinateNames.SUCCESSFUL
            
        return False
    
    @staticmethod
    def get_finite_values(
        data: Union[np.ndarray, xr.DataArray]
    ) -> np.ndarray:
        """Get only finite (non-NaN, non-Inf) values from data.
        
        Args:
            data: Data to filter
            
        Returns:
            1D array of finite values
        """
        values = data.values if isinstance(data, xr.DataArray) else data
        flat_values = values.flatten()
        return flat_values[np.isfinite(flat_values)]


class RobustStatistics:
    """Robust statistical calculations."""
    
    @staticmethod
    def calculate_percentile_limits(
        data: Union[np.ndarray, xr.DataArray],
        min_percentile: float = PlotConstants.DEFAULT_MIN_PERCENTILE,
        max_percentile: float = PlotConstants.DEFAULT_MAX_PERCENTILE,
        fallback_range: Tuple[float, float] = (0.0, 1.0)
    ) -> Tuple[float, float]:
        """Calculate robust min/max using percentiles.
        
        Args:
            data: Data to calculate limits for
            min_percentile: Lower percentile (default: 2)
            max_percentile: Upper percentile (default: 98)
            fallback_range: Range to use if calculation fails
            
        Returns:
            Tuple of (min_value, max_value)
        """
        finite_values = DataValidator.get_finite_values(data)
        
        if len(finite_values) == 0:
            logger.warning("No finite values found, using fallback range")
            return fallback_range
            
        vmin = float(np.percentile(finite_values, min_percentile))
        vmax = float(np.percentile(finite_values, max_percentile))
        
        # Ensure vmin < vmax
        if vmin >= vmax:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
            
        # Final check
        if vmin >= vmax:
            logger.warning("Could not determine valid range, using fallback")
            return fallback_range
            
        return vmin, vmax
    
    @staticmethod
    def get_robust_range(
        data: Union[np.ndarray, xr.DataArray],
        method: str = "percentile",
        **kwargs
    ) -> Tuple[float, float]:
        """Get data range using robust methods.
        
        Args:
            data: Data to analyze
            method: Method to use ("percentile", "iqr", "std")
            **kwargs: Method-specific arguments
            
        Returns:
            Tuple of (min_value, max_value)
        """
        if method == "percentile":
            return RobustStatistics.calculate_percentile_limits(data, **kwargs)
        elif method == "iqr":
            finite_values = DataValidator.get_finite_values(data)
            q1 = np.percentile(finite_values, 25)
            q3 = np.percentile(finite_values, 75)
            iqr = q3 - q1
            factor = kwargs.get("factor", 1.5)
            return (q1 - factor * iqr, q3 + factor * iqr)
        elif method == "std":
            finite_values = DataValidator.get_finite_values(data)
            mean = np.mean(finite_values)
            std = np.std(finite_values)
            n_std = kwargs.get("n_std", 3)
            return (mean - n_std * std, mean + n_std * std)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def calculate_data_bounds(
        data: Union[np.ndarray, xr.DataArray],
        padding_fraction: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate data bounds with optional padding.
        
        Args:
            data: Data to calculate bounds for
            padding_fraction: Fraction of range to add as padding
            
        Returns:
            Tuple of (min_bound, max_bound)
        """
        finite_values = DataValidator.get_finite_values(data)
        
        if len(finite_values) == 0:
            return (0.0, 1.0)
            
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        
        # Add padding
        if padding_fraction > 0:
            vrange = vmax - vmin
            padding = vrange * padding_fraction
            vmin -= padding
            vmax += padding
            
        return vmin, vmax


class ArrayManipulator:
    """Array reshaping and manipulation utilities."""
    
    @staticmethod
    def prepare_heatmap_data(
        z_data: Union[np.ndarray, xr.DataArray],
        orientation: str = "horizontal",
        transpose_axes: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Prepare 2D data for heatmap plotting.
        
        Args:
            z_data: 2D data for heatmap
            orientation: Plot orientation ("horizontal" or "vertical")
            transpose_axes: Specific axes to transpose
            
        Returns:
            Properly oriented 2D array
        """
        values = z_data.values if isinstance(z_data, xr.DataArray) else z_data
        
        if values.ndim != 2:
            raise ValidationError(
                f"Heatmap data must be 2D",
                context={"shape": values.shape, "ndim": values.ndim}
            )
            
        if transpose_axes is not None:
            values = np.transpose(values, transpose_axes)
        elif orientation == "vertical":
            values = values.T
            
        return values
    
    @staticmethod
    def tile_for_hover_data(
        data_1d: np.ndarray,
        target_shape: Tuple[int, int],
        axis: int = 0
    ) -> np.ndarray:
        """Tile 1D data to match 2D shape for hover tooltips.
        
        Args:
            data_1d: 1D array to tile
            target_shape: Target 2D shape
            axis: Axis along which to tile (0 for rows, 1 for columns)
            
        Returns:
            Tiled 2D array
        """
        if data_1d.ndim != 1:
            raise ValidationError(
                f"Input must be 1D",
                context={"shape": data_1d.shape, "ndim": data_1d.ndim}
            )
            
        if axis == 0:
            # Tile along rows
            return np.tile(data_1d.reshape(1, -1), (target_shape[0], 1))
        else:
            # Tile along columns
            return np.tile(data_1d.reshape(-1, 1), (1, target_shape[1]))
    
    @staticmethod
    def stack_custom_data(
        arrays: List[np.ndarray],
        axis: int = -1
    ) -> np.ndarray:
        """Stack multiple arrays along a new axis.
        
        Args:
            arrays: List of arrays to stack
            axis: Axis along which to stack
            
        Returns:
            Stacked array
        """
        if not arrays:
            raise ValidationError("No arrays provided to stack")
            
        # Validate all arrays have same shape
        base_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != base_shape:
                raise ValidationError(
                    f"Array {i} has different shape",
                    context={
                        "expected_shape": base_shape,
                        "actual_shape": arr.shape
                    }
                )
                
        return np.stack(arrays, axis=axis)


class CoordinateTransformer:
    """Coordinate transformation utilities."""
    
    @staticmethod
    def assign_derived_coordinates(
        ds: xr.Dataset,
        transformations: Dict[str, Tuple[str, callable]]
    ) -> xr.Dataset:
        """Create new coordinates derived from existing ones.
        
        Args:
            ds: Dataset to transform
            transformations: Dict mapping new_coord_name -> (source_coord, transform_func)
            
        Returns:
            Dataset with new coordinates added
            
        Example:
            transformations = {
                "freq_GHz": ("frequency", lambda f: f * 1e-9),
                "amp_mV": ("amplitude", lambda a: a * 1e3)
            }
        """
        ds_new = ds.copy()
        
        for new_name, (source_name, transform_func) in transformations.items():
            if source_name not in ds.coords and source_name not in ds.data_vars:
                logger.warning(
                    f"Source '{source_name}' not found for transformation to '{new_name}'"
                )
                continue
                
            source_data = ds[source_name]
            transformed_data = transform_func(source_data)
            
            # Preserve attributes if possible
            if hasattr(transformed_data, "attrs"):
                transformed_data.attrs = source_data.attrs.copy()
                
            ds_new = ds_new.assign_coords({new_name: transformed_data})
            
        return ds_new
    
    @staticmethod
    def create_meshgrid_coordinates(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        indexing: str = "xy"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2D meshgrid from 1D coordinates.
        
        Args:
            x_coords: X coordinate values
            y_coords: Y coordinate values
            indexing: Meshgrid indexing ("xy" or "ij")
            
        Returns:
            Tuple of (X_grid, Y_grid)
        """
        return np.meshgrid(x_coords, y_coords, indexing=indexing)
    
    @staticmethod
    def transform_coordinate_units(
        ds: xr.Dataset,
        coord_name: str,
        unit_conversion: str,
        new_coord_name: Optional[str] = None
    ) -> xr.Dataset:
        """Transform coordinate units using standard conversions.
        
        Args:
            ds: Dataset containing the coordinate
            coord_name: Name of coordinate to transform
            unit_conversion: Type of conversion (e.g., "hz_to_ghz")
            new_coord_name: Name for new coordinate (default: append unit suffix)
            
        Returns:
            Dataset with transformed coordinate
        """
        if coord_name not in ds.coords:
            raise DataSourceError(
                f"Coordinate '{coord_name}' not found",
                context={"available_coords": list(ds.coords)}
            )
            
        # Apply conversion
        converted_values = UnitConverter.apply_unit_conversion(
            ds[coord_name], unit_conversion
        )
        
        # Determine new coordinate name
        if new_coord_name is None:
            unit_suffixes = {
                "hz_to_ghz": "_GHz",
                "hz_to_mhz": "_MHz", 
                "v_to_mv": "_mV",
                "linear_to_dbm": "_dBm"
            }
            suffix = unit_suffixes.get(unit_conversion, "_converted")
            new_coord_name = coord_name + suffix
            
        return ds.assign_coords({new_coord_name: converted_values})


# Convenience functions for common operations

def extract_trace_data(
    ds: xr.Dataset,
    x_source: str,
    y_source: str,
    z_source: Optional[str] = None,
    qubit_id: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract x, y, and optionally z data for plotting.
    
    Args:
        ds: Dataset to extract from
        x_source: Name of x data source
        y_source: Name of y data source  
        z_source: Optional name of z data source (for heatmaps)
        qubit_id: Optional qubit ID to extract data for
        
    Returns:
        Tuple of (x_data, y_data, z_data) where z_data may be None
    """
    # Extract qubit-specific data if needed
    if qubit_id is not None and CoordinateNames.QUBIT in ds.coords:
        ds = DataExtractor.extract_qubit_data(ds, qubit_id)
        
    # Get x and y data
    x_data = ds[x_source].values if x_source in ds else None
    y_data = ds[y_source].values if y_source in ds else None
    
    if x_data is None or y_data is None:
        missing = []
        if x_data is None:
            missing.append(f"x_source '{x_source}'")
        if y_data is None:
            missing.append(f"y_source '{y_source}'")
        raise DataSourceError(
            f"Missing data sources: {', '.join(missing)}",
            context={"available_sources": list(ds.data_vars) + list(ds.coords)}
        )
        
    # Get optional z data
    z_data = None
    if z_source is not None:
        if z_source in ds:
            z_data = ds[z_source].values
        else:
            logger.warning(f"z_source '{z_source}' not found, ignoring")
            
    return x_data, y_data, z_data


def prepare_hover_data_2d(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_values: np.ndarray,
    additional_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """Prepare properly shaped arrays for 2D hover tooltips.
    
    Args:
        x_coords: 1D array of x coordinates
        y_coords: 1D array of y coordinates
        z_values: 2D array of z values
        additional_data: Optional dict of additional arrays to include
        
    Returns:
        Dictionary with all arrays shaped for hover display
    """
    ny, nx = z_values.shape
    
    hover_data = {
        "x": ArrayManipulator.tile_for_hover_data(x_coords, (ny, nx), axis=0),
        "y": ArrayManipulator.tile_for_hover_data(y_coords, (ny, nx), axis=1),
        "z": z_values
    }
    
    if additional_data:
        for name, data in additional_data.items():
            if data.ndim == 1:
                # Determine which axis to tile along based on size
                if len(data) == nx:
                    hover_data[name] = ArrayManipulator.tile_for_hover_data(
                        data, (ny, nx), axis=0
                    )
                elif len(data) == ny:
                    hover_data[name] = ArrayManipulator.tile_for_hover_data(
                        data, (ny, nx), axis=1
                    )
            elif data.shape == z_values.shape:
                hover_data[name] = data
                
    return hover_data