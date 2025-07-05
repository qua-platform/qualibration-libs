"""Overlay abstraction system for unified plotting framework.

This module provides a flexible overlay system that abstracts the rendering
of overlays (lines, markers, annotations) across different plotting backends
and experiment types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import logging

import numpy as np
import xarray as xr

from .configs import LineOverlayConfig, MarkerOverlayConfig, OverlayConfig
from .configs.constants import CoordinateNames, PlotConstants, ExperimentTypes
from .data_utils import DataExtractor
from .exceptions import OverlayError

logger = logging.getLogger(__name__)


class PlotBackend(Protocol):
    """Protocol defining the interface that plot backends must implement."""
    
    def add_vertical_line(
        self, 
        x: float, 
        y_range: Tuple[float, float],
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a vertical line to the plot."""
        ...
    
    def add_horizontal_line(
        self,
        y: float,
        x_range: Tuple[float, float], 
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a horizontal line to the plot."""
        ...
    
    def add_marker(
        self,
        x: float,
        y: float,
        style: Dict[str, Any],
        subplot_position: Tuple[int, int]
    ) -> None:
        """Add a marker to the plot."""
        ...


class BaseOverlay(ABC):
    """Abstract base class for all overlay types.
    
    This class defines the interface that all overlay implementations must follow
    and provides common functionality for extracting data and validating conditions.
    """
    
    def __init__(self, config: OverlayConfig):
        """Initialize the overlay with its configuration.
        
        Args:
            config: Configuration object for the overlay
        """
        self.config = config
    
    @abstractmethod
    def render(
        self,
        backend: PlotBackend,
        ds_fit: xr.Dataset,
        qubit_id: str,
        subplot_position: Tuple[int, int],
        ds_raw: Optional[xr.Dataset] = None
    ) -> None:
        """Render the overlay using the provided backend.
        
        Args:
            backend: The plotting backend to use for rendering
            ds_fit: Fit dataset containing overlay data
            qubit_id: ID of the qubit being plotted
            subplot_position: Grid position (row, col) for the subplot
            ds_raw: Optional raw dataset for additional context
        """
        pass
    
    def should_render(self, ds_fit: xr.Dataset, qubit_id: str) -> bool:
        """Check if this overlay should be rendered based on conditions.
        
        Args:
            ds_fit: Fit dataset to check conditions against
            qubit_id: ID of the qubit being checked
            
        Returns:
            True if the overlay should be rendered
        """
        if not self.config.condition_source:
            return True
        
        try:
            fit_qubit = DataExtractor.extract_qubit_data(ds_fit, qubit_id)
        except (KeyError, ValueError):
            return False
        
        # Check for condition in data variables or coordinates
        if self.config.condition_source in fit_qubit.data_vars:
            condition_value = fit_qubit[self.config.condition_source].values
        elif self.config.condition_source in fit_qubit.coords:
            condition_value = fit_qubit.coords[self.config.condition_source].values
        else:
            return False
        
        # Check condition
        if np.isscalar(condition_value):
            return condition_value == self.config.condition_value
        else:
            # For arrays, check if any value matches
            return np.any(condition_value == self.config.condition_value)
    
    def get_position_value(
        self,
        ds_fit: xr.Dataset,
        source: str,
        qubit_id: str
    ) -> Optional[float]:
        """Extract position value from dataset.
        
        Args:
            ds_fit: Fit dataset containing position data
            source: Name of the data source
            qubit_id: ID of the qubit
            
        Returns:
            Position value or None if not found
        """
        try:
            fit_qubit = DataExtractor.extract_qubit_data(ds_fit, qubit_id)
            
            if source in fit_qubit.data_vars:
                value = fit_qubit[source].values
            elif source in fit_qubit.coords:
                value = fit_qubit.coords[source].values
            elif hasattr(fit_qubit, source):
                value = getattr(fit_qubit, source).values
            else:
                return None
            
            # Convert to scalar
            if np.isscalar(value):
                return float(value)
            else:
                return float(value.item()) if value.size == 1 else float(value[0])
                
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Could not extract position from {source}: {e}")
            return None


class LineOverlay(BaseOverlay):
    """Overlay for rendering lines (vertical or horizontal)."""
    
    def __init__(self, config: LineOverlayConfig):
        """Initialize line overlay.
        
        Args:
            config: Line overlay configuration
        """
        super().__init__(config)
        self.line_config: LineOverlayConfig = config
    
    def render(
        self,
        backend: PlotBackend,
        ds_fit: xr.Dataset,
        qubit_id: str,
        subplot_position: Tuple[int, int],
        ds_raw: Optional[xr.Dataset] = None
    ) -> None:
        """Render the line overlay.
        
        Args:
            backend: The plotting backend to use
            ds_fit: Fit dataset containing line position
            qubit_id: ID of the qubit being plotted
            subplot_position: Grid position for the subplot
            ds_raw: Optional raw dataset for range calculation
        """
        position = self.get_position_value(ds_fit, self.line_config.position_source, qubit_id)
        if position is None:
            return
        
        if self.line_config.orientation == "vertical":
            # Need y-range for vertical line
            y_range = self._get_axis_range(ds_raw, ds_fit, qubit_id, axis='y')
            backend.add_vertical_line(
                x=position,
                y_range=y_range,
                style=self.line_config.line_style,
                subplot_position=subplot_position
            )
        else:  # horizontal
            # Need x-range for horizontal line
            x_range = self._get_axis_range(ds_raw, ds_fit, qubit_id, axis='x')
            backend.add_horizontal_line(
                y=position,
                x_range=x_range,
                style=self.line_config.line_style,
                subplot_position=subplot_position
            )
    
    def _get_axis_range(
        self,
        ds_raw: Optional[xr.Dataset],
        ds_fit: xr.Dataset,
        qubit_id: str,
        axis: str
    ) -> Tuple[float, float]:
        """Get the range for an axis from available data.
        
        Args:
            ds_raw: Raw dataset (preferred source)
            ds_fit: Fit dataset (fallback source)
            qubit_id: ID of the qubit
            axis: 'x' or 'y'
            
        Returns:
            Tuple of (min, max) for the axis
        """
        # Default ranges
        if axis == 'x':
            default_range = (0.0, 1.0)
        else:
            default_range = (-50.0, -25.0)  # Common for power in dBm
        
        # Try to get range from raw data first
        if ds_raw is not None:
            try:
                return self._extract_range_from_dataset(ds_raw, qubit_id, axis)
            except Exception:
                pass
        
        # Try fit data
        try:
            return self._extract_range_from_dataset(ds_fit, qubit_id, axis)
        except Exception:
            return default_range
    
    def _extract_range_from_dataset(
        self,
        ds: xr.Dataset,
        qubit_id: str,
        axis: str
    ) -> Tuple[float, float]:
        """Extract range from a specific dataset."""
        # This is a simplified version - real implementation would
        # need to handle different coordinate names based on experiment type
        if axis == 'x':
            coords = [CoordinateNames.FREQUENCY, CoordinateNames.FLUX_BIAS, 'freq_GHz']
        else:
            coords = [CoordinateNames.POWER, CoordinateNames.AMP_MV, 'power']
        
        for coord in coords:
            if coord in ds.coords:
                values = ds.coords[coord].values
                return (float(np.min(values)), float(np.max(values)))
        
        raise ValueError(f"No suitable coordinate found for {axis} axis")


class MarkerOverlay(BaseOverlay):
    """Overlay for rendering markers at specific positions."""
    
    def __init__(self, config: MarkerOverlayConfig):
        """Initialize marker overlay.
        
        Args:
            config: Marker overlay configuration
        """
        super().__init__(config)
        self.marker_config: MarkerOverlayConfig = config
    
    def render(
        self,
        backend: PlotBackend,
        ds_fit: xr.Dataset,
        qubit_id: str,
        subplot_position: Tuple[int, int],
        ds_raw: Optional[xr.Dataset] = None
    ) -> None:
        """Render the marker overlay.
        
        Args:
            backend: The plotting backend to use
            ds_fit: Fit dataset containing marker position
            qubit_id: ID of the qubit being plotted
            subplot_position: Grid position for the subplot
            ds_raw: Optional raw dataset (unused for markers)
        """
        x_pos = self.get_position_value(ds_fit, self.marker_config.x_source, qubit_id)
        y_pos = self.get_position_value(ds_fit, self.marker_config.y_source, qubit_id)
        
        if x_pos is None or y_pos is None:
            return
        
        backend.add_marker(
            x=x_pos,
            y=y_pos,
            style=self.marker_config.marker_style,
            subplot_position=subplot_position
        )


class ExperimentSpecificOverlay(BaseOverlay):
    """Base class for experiment-specific overlays.
    
    This allows for custom overlay logic based on the experiment type,
    maintaining backward compatibility with existing specialized overlay methods.
    """
    
    def __init__(self, experiment_type: str, overlays: List[OverlayConfig]):
        """Initialize experiment-specific overlay.
        
        Args:
            experiment_type: Type of experiment
            overlays: List of overlay configurations
        """
        # Don't call super().__init__ as we handle multiple configs
        self.experiment_type = experiment_type
        self.overlays = overlays
    
    def should_render(self, ds_fit: xr.Dataset, qubit_id: str) -> bool:
        """Check if fit was successful for this experiment type."""
        try:
            fit_qubit = DataExtractor.extract_qubit_data(ds_fit, qubit_id)
            
            # Check for successful outcome
            if CoordinateNames.OUTCOME in fit_qubit.coords:
                return fit_qubit[CoordinateNames.OUTCOME] == CoordinateNames.SUCCESSFUL
            elif hasattr(fit_qubit, CoordinateNames.OUTCOME):
                return getattr(fit_qubit, CoordinateNames.OUTCOME).values == CoordinateNames.SUCCESSFUL
            
            return True  # Default to rendering if no outcome info
            
        except Exception:
            return False


class FluxSpectroscopyOverlay(ExperimentSpecificOverlay):
    """Specialized overlay renderer for flux spectroscopy experiments."""
    
    def __init__(self, overlays: List[OverlayConfig]):
        """Initialize flux spectroscopy overlay."""
        super().__init__(ExperimentTypes.FLUX_SPECTROSCOPY, overlays)
    
    def render(
        self,
        backend: PlotBackend,
        ds_fit: xr.Dataset,
        qubit_id: str,
        subplot_position: Tuple[int, int],
        ds_raw: Optional[xr.Dataset] = None
    ) -> None:
        """Render flux spectroscopy specific overlays."""
        if not self.should_render(ds_fit, qubit_id):
            return
        
        try:
            fit_qubit = DataExtractor.extract_qubit_data(ds_fit, qubit_id)
            
            # Extract parameters
            if hasattr(fit_qubit, 'fit_results'):
                idle_offset = float(fit_qubit.fit_results.idle_offset.values)
                flux_min = float(fit_qubit.fit_results.flux_min.values)
                sweet_spot_freq = float(fit_qubit.fit_results.sweet_spot_frequency.values) * PlotConstants.GHZ_PER_HZ
            else:
                # Fallback for different structure
                idle_offset = float(fit_qubit.idle_offset.values) * 1e-3  # mV to V
                flux_min = float(fit_qubit.flux_min.values) * 1e-3
                sweet_spot_freq = float(fit_qubit.sweet_spot_frequency.values) * PlotConstants.GHZ_PER_HZ
            
            # Get frequency range from raw data
            freq_range = self._get_frequency_range(ds_raw, qubit_id) if ds_raw else (7.0, 7.6)
            
            # Red dashed vertical line at idle offset
            backend.add_vertical_line(
                x=idle_offset,
                y_range=freq_range,
                style={"color": "#FF0000", "width": 2.5, "dash": "dash"},
                subplot_position=subplot_position
            )
            
            # Purple dashed vertical line at flux min
            backend.add_vertical_line(
                x=flux_min,
                y_range=freq_range,
                style={"color": "#800080", "width": 2.5, "dash": "dash"},
                subplot_position=subplot_position
            )
            
            # Magenta marker at sweet spot
            backend.add_marker(
                x=idle_offset,
                y=sweet_spot_freq,
                style={"symbol": "x", "color": "#FF00FF", "size": 15},
                subplot_position=subplot_position
            )
            
        except Exception as e:
            logger.warning(f"Could not render flux spectroscopy overlay for {qubit_id}: {e}")
    
    def _get_frequency_range(self, ds_raw: xr.Dataset, qubit_id: str) -> Tuple[float, float]:
        """Extract frequency range from raw dataset."""
        try:
            ds_transposed = ds_raw.transpose(CoordinateNames.QUBIT, CoordinateNames.DETUNING, CoordinateNames.FLUX_BIAS)
            freq_coord = CoordinateNames.FULL_FREQ if CoordinateNames.FULL_FREQ in ds_transposed else "freq_full"
            
            if freq_coord in ds_transposed:
                freq_array = ds_transposed[freq_coord].values
                q_labels = list(ds_transposed[CoordinateNames.QUBIT].values)
                q_idx = q_labels.index(qubit_id)
                freq_vals = freq_array[q_idx] * PlotConstants.GHZ_PER_HZ
                return (float(freq_vals.min()), float(freq_vals.max()))
        except Exception:
            pass
        
        return (7.0, 7.6)  # Default GHz range


class OverlayManager:
    """Manages the creation and rendering of overlays.
    
    This class acts as a factory and coordinator for different overlay types,
    handling the routing to appropriate overlay implementations based on
    configuration and experiment type.
    """
    
    def __init__(self):
        """Initialize the overlay manager."""
        self._experiment_handlers = {
            ExperimentTypes.FLUX_SPECTROSCOPY: FluxSpectroscopyOverlay,
            # Add more experiment-specific handlers as needed
        }
    
    def create_overlays(
        self,
        overlay_configs: List[Union[LineOverlayConfig, MarkerOverlayConfig]],
        experiment_type: Optional[str] = None
    ) -> List[BaseOverlay]:
        """Create overlay instances from configurations.
        
        Args:
            overlay_configs: List of overlay configurations
            experiment_type: Optional experiment type for specialized handling
            
        Returns:
            List of overlay instances ready for rendering
        """
        # Check if we should use experiment-specific handler
        if experiment_type and experiment_type in self._experiment_handlers:
            handler_class = self._experiment_handlers[experiment_type]
            return [handler_class(overlay_configs)]
        
        # Otherwise create individual overlays
        overlays = []
        for config in overlay_configs:
            if isinstance(config, LineOverlayConfig):
                overlays.append(LineOverlay(config))
            elif isinstance(config, MarkerOverlayConfig):
                overlays.append(MarkerOverlay(config))
            else:
                logger.warning(f"Unknown overlay config type: {type(config)}")
        
        return overlays
    
    def render_overlays(
        self,
        overlays: List[BaseOverlay],
        backend: PlotBackend,
        ds_fit: xr.Dataset,
        qubit_id: str,
        subplot_position: Tuple[int, int],
        ds_raw: Optional[xr.Dataset] = None
    ) -> None:
        """Render all overlays using the provided backend.
        
        Args:
            overlays: List of overlay instances to render
            backend: The plotting backend to use
            ds_fit: Fit dataset containing overlay data
            qubit_id: ID of the qubit being plotted
            subplot_position: Grid position for the subplot
            ds_raw: Optional raw dataset for additional context
        """
        for overlay in overlays:
            try:
                if overlay.should_render(ds_fit, qubit_id):
                    overlay.render(backend, ds_fit, qubit_id, subplot_position, ds_raw)
            except Exception as e:
                logger.warning(
                    f"Failed to render overlay {type(overlay).__name__} "
                    f"for qubit {qubit_id}: {e}"
                )