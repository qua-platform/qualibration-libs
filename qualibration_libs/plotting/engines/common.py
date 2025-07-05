"""
Common utilities and helper functions for plotting engines.

This module provides shared functionality used by both Plotly and Matplotlib
rendering engines to ensure consistent behavior and reduce code duplication.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from ..configs import (ColorbarConfig, Colors, HeatmapTraceConfig,
                       LineOverlayConfig, MarkerOverlayConfig, OverlayConfig,
                       TraceConfig)
from ..configs.constants import CoordinateNames, PlotConstants
from ..grids import QubitGrid
from ..data_utils import DataExtractor


class PlotlyEngineUtils:
    """Utility functions for Plotly rendering engine."""
    
    @staticmethod
    def calculate_robust_zlimits(z_data: np.ndarray, zmin_percentile: float = PlotConstants.DEFAULT_MIN_PERCENTILE, zmax_percentile: float = PlotConstants.DEFAULT_MAX_PERCENTILE) -> Tuple[float, float]:
        """Calculate robust z-axis limits using percentiles."""
        flat_data = z_data.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        if len(valid_data) == 0:
            return 0.0, 1.0
        
        zmin = float(np.percentile(valid_data, zmin_percentile))
        zmax = float(np.percentile(valid_data, zmax_percentile))
        
        # Ensure zmin < zmax
        if zmin >= zmax:
            zmin = float(np.min(valid_data))
            zmax = float(np.max(valid_data))
            
        return zmin, zmax
    
    @staticmethod
    def build_custom_data(ds: xr.Dataset, custom_data_sources: List[str]) -> Optional[np.ndarray]:
        """Build custom data array from specified sources."""
        if not custom_data_sources:
            return None
        
        custom_arrays = []
        for source in custom_data_sources:
            if DataExtractor.check_data_source_exists(ds, source):
                data_array = DataExtractor.get_data_array_safe(ds, source)
                if data_array is not None:
                    custom_arrays.append(data_array.values)
        
        if not custom_arrays:
            return None
        
        return np.stack(custom_arrays, axis=-1)
    
    @staticmethod
    def check_trace_visibility(ds: xr.Dataset, trace_config: TraceConfig, qubit_id: str) -> bool:
        """Check if a trace should be visible based on conditions."""
        if not trace_config.visible:
            return False
        
        if trace_config.condition_source and DataExtractor.check_data_source_exists(ds, trace_config.condition_source):
            data_array = DataExtractor.get_data_array_safe(ds, trace_config.condition_source)
            if data_array is not None:
                condition_value = data_array.values
                # Handle scalar and array conditions
                if np.isscalar(condition_value):
                    return condition_value == trace_config.condition_value
                else:
                    # For array conditions, check if any values match
                    return np.any(condition_value == trace_config.condition_value)
        
        return True
    
    @staticmethod
    def validate_trace_sources(ds: xr.Dataset, trace_config: TraceConfig) -> bool:
        """Validate that all required data sources exist in dataset."""
        required_sources = [trace_config.x_source, trace_config.y_source]
        
        if isinstance(trace_config, HeatmapTraceConfig) and trace_config.z_source:
            required_sources.append(trace_config.z_source)
        
        return all(source in ds for source in required_sources)
    
    @staticmethod
    def create_subplot_title(qubit_id: str, template: str = "Qubit {qubit}") -> str:
        """Create standardized subplot title."""
        return template.format(qubit=qubit_id)


class MatplotlibEngineUtils:
    """Utility functions for Matplotlib rendering engine."""
    
    @staticmethod
    def translate_plotly_linestyle(plotly_dash: str) -> str:
        """Convert Plotly dash style to Matplotlib linestyle."""
        linestyle_map = {
            "solid": "-",
            "dot": ":",
            "dash": "--", 
            "longdash": "-.",
            "dashdot": "-."
        }
        return linestyle_map.get(plotly_dash, "--")
    
    @staticmethod
    def extract_matplotlib_color(style_dict: Dict[str, Any]) -> str:
        """Extract color from style dictionary with fallback."""
        return style_dict.get("color", Colors.RAW_DATA)
    
    @staticmethod
    def apply_matplotlib_styling(ax, layout_config, trace_config: Optional[TraceConfig] = None):
        """Apply standardized matplotlib styling to an axis."""
        ax.set_xlabel(layout_config.x_axis_title)
        ax.set_ylabel(layout_config.y_axis_title)
        
        if trace_config and hasattr(trace_config, 'style'):
            # Apply any matplotlib-specific styling from trace config
            pass


class OverlayRenderer:
    """Handles rendering of overlays for both Plotly and Matplotlib."""
    
    @staticmethod
    def _normalize_fit_dataset(ds_fit: xr.Dataset, qubit_id: str) -> xr.Dataset:
        """Normalize fit dataset to single-qubit format for overlay rendering."""
        # Check if dataset has qubit as a dimension (not just coordinate)
        if CoordinateNames.QUBIT in ds_fit.dims:
            # Dataset still has qubit dimension - need to select
            if qubit_id not in ds_fit.coords.get(CoordinateNames.QUBIT, []):
                raise ValueError(f"Qubit {qubit_id} not found in dataset")
            return ds_fit.sel(**{CoordinateNames.QUBIT: qubit_id})
        else:
            # Dataset is already a single-qubit slice - use directly
            return ds_fit
    
    @staticmethod
    def should_render_overlay(ds_fit: xr.Dataset, overlay_config: OverlayConfig, qubit_id: str) -> bool:
        """Check if overlay should be rendered based on conditions."""
        try:
            fit_qubit = OverlayRenderer._normalize_fit_dataset(ds_fit, qubit_id)
        except ValueError:
            return False
        
        # Check if condition source exists using DataExtractor
        if not DataExtractor.check_data_source_exists(fit_qubit, overlay_config.condition_source):
            return False
            
        # Get condition value using DataExtractor
        data_array = DataExtractor.get_data_array_safe(fit_qubit, overlay_config.condition_source)
        if data_array is None:
            return False
            
        condition_value = data_array.values
        
        # Check if condition matches - handle both scalar and array values
        if np.isscalar(condition_value):
            result = condition_value == overlay_config.condition_value
        else:
            # For arrays or DataArrays, extract the scalar value
            try:
                scalar_value = condition_value.item() if hasattr(condition_value, 'item') else condition_value
                result = scalar_value == overlay_config.condition_value
            except (ValueError, AttributeError):
                result = False
        
        return result
    
    @staticmethod
    def get_overlay_position(ds_fit: xr.Dataset, source: str, qubit_id: str) -> Optional[float]:
        """Get position value for overlay from fit dataset."""
        try:
            fit_qubit = OverlayRenderer._normalize_fit_dataset(ds_fit, qubit_id)
        except ValueError:
            return None
        
        # Get data using DataExtractor
        data_array = DataExtractor.get_data_array_safe(fit_qubit, source)
        if data_array is None:
            return None
            
        value = data_array.values
            
        if np.isscalar(value):
            return float(value)
        elif hasattr(value, 'size') and value.size == 1:
            return float(value.item())
        elif hasattr(value, '__len__') and len(value) == 1:
            return float(value[0])
        else:
            # For arrays with multiple values, take the first one
            return float(np.array(value).flatten()[0])


class GridManager:
    """Manages qubit grid layout and iteration for plotting."""
    
    @staticmethod
    def create_grid(ds_raw: xr.Dataset, qubits: List[AnyTransmon], create_figure: bool = True) -> QubitGrid:
        """Create QubitGrid with appropriate settings."""
        grid_locations = [q.grid_location for q in qubits]
        return QubitGrid(ds_raw, grid_locations, create_figure=create_figure)
    
    @staticmethod
    def get_grid_dimensions(grid: QubitGrid) -> Tuple[int, int]:
        """Get grid dimensions (rows, cols)."""
        return grid.n_rows, grid.n_cols
    
    @staticmethod
    def get_subplot_titles(grid: QubitGrid) -> List[str]:
        """Get standardized subplot titles."""
        return grid.get_subplot_titles()


class DataSourceValidator:
    """Validates data sources and provides fallbacks."""
    
    @staticmethod
    def validate_required_sources(ds: xr.Dataset, required_sources: List[str]) -> Tuple[bool, List[str]]:
        """Validate that required data sources exist."""
        missing_sources = [source for source in required_sources if source not in ds]
        return len(missing_sources) == 0, missing_sources
    
    @staticmethod
    def get_available_sources(ds: xr.Dataset) -> List[str]:
        """Get list of available data sources in dataset."""
        return list(ds.data_vars.keys()) + list(ds.coords.keys())
    
    @staticmethod
    def suggest_alternatives(ds: xr.Dataset, missing_source: str) -> List[str]:
        """Suggest alternative data sources for missing ones."""
        available = DataSourceValidator.get_available_sources(ds)
        
        # Simple matching based on keywords
        suggestions = []
        keywords = missing_source.lower().split('_')
        
        for source in available:
            source_lower = source.lower()
            if any(keyword in source_lower for keyword in keywords):
                suggestions.append(source)
        
        return suggestions[:3]  # Return top 3 suggestions


class LegendManager:
    """Manages legend display and formatting."""
    
    @staticmethod
    def should_show_legend(trace_configs: List[TraceConfig]) -> bool:
        """Determine if legend should be shown based on trace configurations."""
        named_traces = [trace for trace in trace_configs if trace.name and trace.visible]
        return len(named_traces) > 1
    
    @staticmethod
    def get_legend_names(trace_configs: List[TraceConfig]) -> List[str]:
        """Get list of legend names from trace configurations."""
        return [trace.name for trace in trace_configs if trace.name and trace.visible]