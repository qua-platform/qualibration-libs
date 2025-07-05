"""Base rendering engine for all plotting backends.

This module provides the abstract base class that all plotting engines must inherit from.
It contains shared functionality like experiment detection, data validation, and routing logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Tuple
import logging
import numpy as np
import xarray as xr

from ..configs import (
    PlotConfig, SpectroscopyConfig, HeatmapConfig, HeatmapTraceConfig, 
    TraceConfig
)
from .common import (
    GridManager, OverlayRenderer, QubitGrid
)
from .data_validators import DataValidator
from ..exceptions import (
    ConfigurationError, DataSourceError, EngineError, QubitError
)

logger = logging.getLogger(__name__)


class BaseRenderingEngine(ABC):
    """Abstract base class for all plotting engines.
    
    This class provides shared functionality for experiment detection, data validation,
    and figure creation routing. Concrete implementations must provide backend-specific
    rendering methods.
    
    Attributes:
        data_validator: Validator for input data.
        grid_manager: Manager for grid layout calculations.
        overlay_renderer: Renderer for plot overlays like fit results.
    """
    
    def __init__(self) -> None:
        """Initialize base engine with common utilities."""
        self.data_validator = DataValidator()
        self.grid_manager = GridManager()
        self.overlay_renderer = OverlayRenderer()
        # Engine-specific utils should be initialized in subclasses
        
    def create_figure(
        self, 
        ds_raw: xr.Dataset, 
        qubits: List[str], 
        plot_configs: List[PlotConfig], 
        ds_fit: Optional[xr.Dataset] = None
    ) -> Any:
        """Create a figure based on the provided configuration.
        
        This is the main entry point that routes to specialized figure creation methods
        based on the configuration type.
        
        Args:
            ds_raw: Raw measurement data
            qubits: List of qubit IDs to plot
            plot_configs: List of plot configurations
            ds_fit: Optional fit results dataset
            
        Returns:
            Figure object (type depends on backend)
            
        Raises:
            TypeError: If inputs are not of expected types
            ConfigurationError: If plot configuration is invalid
            EngineError: If figure creation fails
        """
        # Input validation
        if not isinstance(ds_raw, xr.Dataset):
            raise TypeError(f"ds_raw must be an xarray Dataset, got {type(ds_raw).__name__}")
        
        if not isinstance(qubits, list):
            raise TypeError(f"qubits must be a list, got {type(qubits).__name__}")
            
        if not plot_configs:
            logger.debug("No plot configurations provided, creating empty figure")
            return self._create_empty_figure()
            
        # Validate all configs are the same type
        config_types = {type(config) for config in plot_configs}
        if len(config_types) > 1:
            raise ConfigurationError(
                "All plot configurations must be of the same type",
                context={"config_types": [t.__name__ for t in config_types]},
                suggestions=["Use separate create_figure calls for different plot types"]
            )
            
        # Use first config to determine plot type
        config = plot_configs[0]
        
        try:
            # Route to appropriate specialized method
            if isinstance(config, SpectroscopyConfig):
                return self.create_spectroscopy_figure(ds_raw, qubits, config, ds_fit)
            elif isinstance(config, HeatmapConfig):
                return self.create_heatmap_figure(ds_raw, qubits, config, ds_fit)
            else:
                return self._create_generic_figure(ds_raw, qubits, config, ds_fit)
        except Exception as e:
            # Re-raise our custom exceptions as-is
            if isinstance(e, (ConfigurationError, DataSourceError, QubitError)):
                raise
            # Wrap unexpected exceptions
            raise EngineError(
                f"Failed to create figure: {str(e)}",
                context={"config_type": type(config).__name__, "engine": self.__class__.__name__},
                suggestions=["Check the data format", "Verify configuration settings"]
            ) from e
    
    # ==================== Experiment Detection Methods ====================
    
    def _is_flux_spectroscopy(self, ds_raw: xr.Dataset) -> bool:
        """Detect if dataset is for flux spectroscopy experiment.
        
        Flux spectroscopy experiments have flux-related coordinates but no power coordinates.
        
        Args:
            ds_raw: Raw dataset to check
            
        Returns:
            True if this is a flux spectroscopy experiment
        """
        flux_indicators = ["flux_bias", "attenuated_current"]
        power_indicators = ["power", "power_dbm"]
        
        has_flux = any(coord in ds_raw.coords for coord in flux_indicators)
        has_power = any(coord in ds_raw.coords for coord in power_indicators)
        
        return has_flux and not has_power
    
    def _is_power_rabi(self, ds_raw: xr.Dataset) -> bool:
        """Detect if dataset is for power rabi experiment.
        
        Power Rabi experiments have amplitude and pulse-related coordinates.
        
        Args:
            ds_raw: Raw dataset to check
            
        Returns:
            True if this is a power rabi experiment
        """
        power_rabi_indicators = ["amp_prefactor", "full_amp", "nb_of_pulses"]
        
        return all(
            coord in ds_raw.coords or coord in ds_raw.dims 
            for coord in power_rabi_indicators
        )
    
    def _is_fit_successful(self, ds_fit: Optional[xr.Dataset]) -> bool:
        """Check if fit was successful.
        
        Args:
            ds_fit: Fit results dataset
            
        Returns:
            True if fit was successful
        """
        if ds_fit is None:
            return False
            
        return (
            hasattr(ds_fit, "outcome") and 
            getattr(ds_fit.outcome, "values", None) == "successful"
        )
    
    # ==================== Validation Methods ====================
    
    def _check_trace_visibility(
        self, 
        ds: xr.Dataset, 
        trace_config: TraceConfig, 
        qubit_id: str
    ) -> bool:
        """Check if a trace should be visible based on conditions.
        
        Args:
            ds: Dataset containing the data
            trace_config: Trace configuration with visibility conditions
            qubit_id: ID of the qubit being checked
            
        Returns:
            True if trace should be visible
            
        Raises:
            DataSourceError: If condition source is specified but not found
        """
        if not trace_config.visible:
            return False
        
        if trace_config.condition_source:
            if trace_config.condition_source not in ds:
                raise DataSourceError(
                    f"Condition source '{trace_config.condition_source}' not found in dataset",
                    context={
                        "available_variables": list(ds.data_vars),
                        "qubit_id": qubit_id
                    },
                    suggestions=[
                        "Check the condition_source name in configuration",
                        "Verify the dataset contains the expected variable"
                    ]
                )
            
            try:
                condition_value = ds[trace_config.condition_source].values
                if np.isscalar(condition_value):
                    return condition_value == trace_config.condition_value
                else:
                    return np.any(condition_value == trace_config.condition_value)
            except Exception as e:
                logger.warning(f"Error checking trace visibility condition: {e}")
                return True  # Default to visible on error
        
        return True
    
    def _validate_trace_sources(
        self, 
        ds: xr.Dataset, 
        trace_config: TraceConfig
    ) -> bool:
        """Validate that all required data sources exist in dataset.
        
        Args:
            ds: Dataset to validate against
            trace_config: Trace configuration with data sources
            
        Returns:
            True if all required sources exist
            
        Raises:
            DataSourceError: If required sources are missing
        """
        required_sources = [trace_config.x_source, trace_config.y_source]
        
        if isinstance(trace_config, HeatmapTraceConfig) and trace_config.z_source:
            required_sources.append(trace_config.z_source)
        
        missing_sources = [
            source for source in required_sources 
            if source and source not in ds
        ]
        
        if missing_sources:
            raise DataSourceError(
                f"Required data sources not found in dataset",
                context={
                    "missing_sources": missing_sources,
                    "available_sources": list(ds.data_vars) + list(ds.coords),
                    "trace_name": getattr(trace_config, 'name', 'unnamed')
                },
                suggestions=[
                    f"Check if '{source}' exists with a different name" for source in missing_sources
                ] + ["Verify data preparation step completed successfully"]
            )
        
        return True
    
    def _calculate_robust_zlimits(
        self, 
        z_data: np.ndarray, 
        zmin_percentile: float = 2.0, 
        zmax_percentile: float = 98.0
    ) -> Tuple[float, float]:
        """Calculate robust z-axis limits using percentiles.
        
        Args:
            z_data: 2D array of z values
            zmin_percentile: Lower percentile for minimum
            zmax_percentile: Upper percentile for maximum
            
        Returns:
            Tuple of (zmin, zmax)
        """
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
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    def create_spectroscopy_figure(
        self, 
        ds_raw: xr.Dataset, 
        qubits: List[str], 
        config: SpectroscopyConfig, 
        ds_fit: Optional[xr.Dataset] = None
    ) -> Any:
        """Create a spectroscopy (1D) figure.
        
        Args:
            ds_raw: Raw measurement data
            qubits: List of qubit IDs to plot
            config: Spectroscopy configuration
            ds_fit: Optional fit results
            
        Returns:
            Backend-specific figure object
        """
        pass
    
    @abstractmethod
    def create_heatmap_figure(
        self, 
        ds_raw: xr.Dataset, 
        qubits: List[str], 
        config: HeatmapConfig, 
        ds_fit: Optional[xr.Dataset] = None
    ) -> Any:
        """Create a heatmap (2D) figure.
        
        Args:
            ds_raw: Raw measurement data
            qubits: List of qubit IDs to plot
            config: Heatmap configuration
            ds_fit: Optional fit results
            
        Returns:
            Backend-specific figure object
        """
        pass
    
    @abstractmethod
    def _create_generic_figure(
        self, 
        ds_raw: xr.Dataset, 
        qubits: List[str], 
        config: PlotConfig, 
        ds_fit: Optional[xr.Dataset] = None
    ) -> Any:
        """Create a generic figure for custom plot types.
        
        Args:
            ds_raw: Raw measurement data
            qubits: List of qubit IDs to plot
            config: Generic plot configuration
            ds_fit: Optional fit results
            
        Returns:
            Backend-specific figure object
        """
        pass
    
    @abstractmethod
    def _create_empty_figure(self) -> Any:
        """Create an empty figure when no configs provided.
        
        Subclasses must implement this method to create an empty figure/canvas
        appropriate for their backend (e.g., matplotlib Figure, plotly Figure).
        
        Returns:
            Backend-specific empty figure object
        """
        pass
    
    @abstractmethod
    def _add_spectroscopy_traces(
        self,
        figure: Any,
        ds_raw: xr.Dataset,
        qubits: List[str],
        config: SpectroscopyConfig,
        ds_fit: Optional[xr.Dataset] = None
    ) -> None:
        """Add spectroscopy traces to an existing figure.
        
        Subclasses must implement this method to add 1D traces (lines) to
        the figure using their backend-specific plotting functions.
        
        Args:
            figure: Backend-specific figure object to add traces to.
            ds_raw: Raw measurement data.
            qubits: List of qubit IDs to plot.
            config: Spectroscopy configuration with trace definitions.
            ds_fit: Optional fit results dataset.
        """
        pass
    
    @abstractmethod
    def _add_heatmap_trace(
        self,
        figure: Any,
        ds_raw: xr.Dataset,
        qubit_id: str,
        trace_config: HeatmapTraceConfig,
        subplot_position: Tuple[int, int],
        ds_fit: Optional[xr.Dataset] = None
    ) -> None:
        """Add a single heatmap trace to a figure.
        
        Subclasses must implement this method to add 2D heatmap data to
        the figure at the specified subplot position.
        
        Args:
            figure: Backend-specific figure object to add trace to.
            ds_raw: Raw measurement data.
            qubit_id: ID of the qubit being plotted.
            trace_config: Configuration for this specific heatmap trace.
            subplot_position: Grid position (row, col) for this subplot.
            ds_fit: Optional fit results dataset.
        """
        pass
    
    @abstractmethod
    def _add_generic_trace(
        self,
        figure: Any,
        ds: xr.Dataset,
        qubit_id: str,
        trace_config: TraceConfig,
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a generic trace to a figure.
        
        Subclasses must implement this method to add generic traces
        (not spectroscopy or heatmap specific) to the figure.
        
        Args:
            figure: Backend-specific figure object to add trace to.
            ds: Dataset containing the data.
            qubit_id: ID of the qubit being plotted.
            trace_config: Configuration for this trace.
            subplot_position: Grid position (row, col) for this subplot.
        """
        pass
    
    @abstractmethod
    def _add_overlays(
        self,
        figure: Any,
        overlays: List[Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add overlay elements to a subplot.
        
        Subclasses must implement this method to add overlay elements
        like fit lines, annotations, or markers to a specific subplot.
        
        Args:
            figure: Backend-specific figure object.
            overlays: List of overlay configurations to add.
            subplot_position: Grid position (row, col) for the target subplot.
        """
        pass
    
    @abstractmethod
    def _add_dual_axis(
        self,
        figure: Any,
        ds: xr.Dataset,
        qubit_id: str,
        primary_config: TraceConfig,
        secondary_config: TraceConfig,
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add dual y-axis traces to a subplot.
        
        Subclasses must implement this method to create plots with two
        y-axes sharing the same x-axis (e.g., for plotting different
        quantities with different scales).
        
        Args:
            figure: Backend-specific figure object.
            ds: Dataset containing the data.
            qubit_id: ID of the qubit being plotted.
            primary_config: Configuration for the primary (left) y-axis.
            secondary_config: Configuration for the secondary (right) y-axis.
            subplot_position: Grid position (row, col) for this subplot.
        """
        pass
    
    # ==================== Helper Methods for Subclasses ====================
    
    def _get_experiment_type(self, ds_raw: xr.Dataset) -> str:
        """Determine the experiment type from dataset structure.
        
        Args:
            ds_raw: Raw dataset
            
        Returns:
            String identifier for experiment type
        """
        if self._is_power_rabi(ds_raw):
            return "power_rabi"
        elif self._is_flux_spectroscopy(ds_raw):
            return "flux_spectroscopy"
        else:
            return "generic"
    
    def _should_add_overlays(
        self, 
        config: PlotConfig, 
        ds_fit: Optional[xr.Dataset]
    ) -> bool:
        """Check if overlays should be added to the plot.
        
        Args:
            config: Plot configuration
            ds_fit: Fit results dataset
            
        Returns:
            True if overlays should be added
        """
        if not hasattr(config, 'overlays') or not config.overlays:
            return False
            
        if ds_fit is None:
            return False
            
        return True