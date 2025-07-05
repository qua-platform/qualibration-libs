"""
Enhanced Matplotlib rendering engine for the unified plotting framework.

This engine provides specialized Matplotlib rendering with support for complex
plot types and ensures visual compatibility with the Plotly engine outputs.
"""

from typing import List, Optional, Union
import logging

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure as MatplotlibFigure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from ..configs import (Colors, HeatmapConfig, HeatmapTraceConfig,
                       LineOverlayConfig, MarkerOverlayConfig,
                       PlotConfig, SpectroscopyConfig, TraceConfig)
from ..configs.constants import CoordinateNames, PlotConstants, ExperimentTypes
from ..grids import QubitGrid, grid_iter
from .base_engine import BaseRenderingEngine
from .common import GridManager, MatplotlibEngineUtils, OverlayRenderer
from .data_validators import DataValidator
from .experiment_detector import ExperimentDetector
from ..exceptions import (
    PlottingError
)
from ..data_utils import DataValidator as DataUtilsValidator, extract_trace_data

logger = logging.getLogger(__name__)


class MatplotlibEngine(BaseRenderingEngine):
    """Enhanced Matplotlib rendering engine with specialized plot type support.
    
    This engine provides specialized Matplotlib rendering capabilities for quantum
    calibration experiments, with support for spectroscopy, heatmap, and generic
    plot types while maintaining visual compatibility with Plotly outputs.
    
    Attributes:
        utils: Utilities for Matplotlib-specific operations
        overlay_renderer: Renderer for overlay elements
        grid_manager: Manager for subplot grid layouts
        data_validator: Validator for data integrity
        experiment_detector: Detector for experiment types
    """
    
    def __init__(self) -> None:
        """Initialize the Matplotlib rendering engine.
        
        Sets up all necessary components including utilities, overlay renderer,
        grid manager, data validator, and experiment detector.
        """
        super().__init__()
        self.utils = MatplotlibEngineUtils()
        self.overlay_renderer = OverlayRenderer()
        self.grid_manager = GridManager()
        self.data_validator = DataValidator()
        self.experiment_detector = ExperimentDetector()
    
    def _create_empty_figure(self) -> MatplotlibFigure:
        """Create an empty figure when no configs provided.
        
        Returns:
            MatplotlibFigure: An empty matplotlib figure with default subplot
        """
        fig, _ = plt.subplots()
        return fig
    
    def create_spectroscopy_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: SpectroscopyConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> MatplotlibFigure:
        """Create specialized figure for 1D spectroscopy plots."""
        
        # Setup figure
        grid = self._setup_figure_grid(ds_raw, qubits, config.layout.title)
        
        # Track which legend entries have been created
        legend_entries_created = set()
        
        # Plot data for each qubit
        for i, (ax, name_dict) in enumerate(grid_iter(grid)):
            qubit_id = list(name_dict.values())[0]
            self._plot_spectroscopy_subplot(
                ax, ds_raw, ds_fit, qubit_id, config, i, legend_entries_created
            )
        
        # Add combined legend
        self._add_combined_legend(grid)
        
        grid.fig.tight_layout(rect=[
            PlotConstants.TIGHT_LAYOUT_RECT_LEFT,
            PlotConstants.TIGHT_LAYOUT_RECT_BOTTOM,
            PlotConstants.TIGHT_LAYOUT_RECT_RIGHT,
            PlotConstants.TIGHT_LAYOUT_RECT_TOP
        ])
        return grid.fig
    
    def _setup_figure_grid(self, ds_raw: xr.Dataset, qubits: List[AnyTransmon], title: str) -> QubitGrid:
        """Set up the figure grid with standard settings."""
        grid = self._create_grid_layout(ds_raw, qubits, create_figure=True)
        
        # Use base class dimensions calculation
        dimensions = self._calculate_figure_dimensions(grid.n_rows, grid.n_cols, "standard")
        grid.fig.set_size_inches(dimensions["width_inches"], dimensions["height_inches"])
        grid.fig.suptitle(title)
        return grid
    
    def _plot_spectroscopy_subplot(
        self,
        ax: plt.Axes,
        ds_raw: xr.Dataset,
        ds_fit: Optional[xr.Dataset],
        qubit_id: str,
        config: SpectroscopyConfig,
        subplot_index: int,
        legend_entries_created: set
    ) -> None:
        """Plot spectroscopy data for a single subplot."""
        # Extract datasets for this qubit
        ds_qubit_raw, ds_qubit_fit = self._extract_qubit_datasets(ds_raw, ds_fit, qubit_id)
        
        # Plot raw data traces
        self._add_spectroscopy_traces(ax, ds_qubit_raw, config.traces, subplot_index, legend_entries_created)
        
        # Plot fit traces if available and successful
        if self._should_plot_fit_traces(ds_qubit_fit):
            self._add_spectroscopy_traces(ax, ds_qubit_fit, config.fit_traces, subplot_index, legend_entries_created, is_fit=True)
        
        # Configure axis labels and title
        self._configure_subplot_axes(ax, config.layout, qubit_id)
        
        # Add dual axis if configured
        if config.dual_axis and config.dual_axis.enabled:
            self._add_dual_axis(ax, ds_qubit_raw, config.dual_axis)
    
    def _should_plot_fit_traces(self, ds_qubit_fit: Optional[xr.Dataset]) -> bool:
        """Check if fit traces should be plotted."""
        return ds_qubit_fit is not None and DataUtilsValidator.validate_fit_success(ds_qubit_fit)
    
    def _configure_subplot_axes(self, ax: plt.Axes, layout: any, qubit_id: str) -> None:
        """Configure axis labels and title for a subplot."""
        ax.set_xlabel(layout.x_axis_title)
        ax.set_ylabel(layout.y_axis_title)
        ax.set_title(f"{CoordinateNames.QUBIT.capitalize()} {qubit_id}")
    
    def _add_combined_legend(self, grid: QubitGrid) -> None:
        """Add combined legend from all subplots."""
        all_handles = []
        all_labels = []
        
        # Collect unique legend entries from all subplots
        for axis_row in grid.axes:
            for ax in axis_row:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label and label not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(label)
        
        if all_handles:
            grid.fig.legend(all_handles, all_labels, loc='upper right')
    
    def create_heatmap_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: HeatmapConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> MatplotlibFigure:
        """Create specialized figure for 2D heatmap plots."""
        
        # Use base class methods for grid creation
        grid = self._create_grid_layout(ds_raw, qubits, create_figure=True)
        
        # Determine plot type and calculate dimensions
        plot_type = "flux" if self._is_flux_spectroscopy(ds_raw) else "heatmap"
        dimensions = self._calculate_figure_dimensions(grid.n_rows, grid.n_cols, plot_type)
        grid.fig.set_size_inches(dimensions["width_inches"], dimensions["height_inches"])
        grid.fig.suptitle(config.layout.title)
        
        for i, (ax, name_dict) in enumerate(grid_iter(grid)):
            qubit_id = list(name_dict.values())[0]
            ds_qubit_raw, ds_qubit_fit = self._extract_qubit_datasets(ds_raw, ds_fit, qubit_id)
            
            # Plot heatmap traces
            for trace_config in config.traces:
                if self._check_trace_visibility(ds_qubit_raw, trace_config, qubit_id):
                    self._add_heatmap_trace(ax, ds_qubit_raw, trace_config, i)
            
            # Add overlays if fit data available
            if ds_qubit_fit is not None and config.overlays:
                # Route to appropriate overlay handler based on experiment type
                experiment_type = self.experiment_detector.detect_experiment_type(ds_raw)
                if experiment_type == ExperimentTypes.FLUX_SPECTROSCOPY:
                    self._add_overlays_flux_spectroscopy_matplotlib(ax, ds_qubit_fit, qubit_id, config.overlays, ds_raw)
                else:
                    self._add_overlays(ax, ds_qubit_fit, qubit_id, config.overlays)
            
            # Set labels and title
            ax.set_xlabel(config.layout.x_axis_title)
            ax.set_ylabel(config.layout.y_axis_title)
            ax.set_title(f"{CoordinateNames.QUBIT.capitalize()} {qubit_id}")
        
        grid.fig.tight_layout(rect=[
            PlotConstants.TIGHT_LAYOUT_RECT_LEFT,
            PlotConstants.TIGHT_LAYOUT_RECT_BOTTOM,
            PlotConstants.TIGHT_LAYOUT_RECT_RIGHT,
            PlotConstants.TIGHT_LAYOUT_RECT_TOP
        ])
        return grid.fig
    
    def _create_generic_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: PlotConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> MatplotlibFigure:
        """Create generic figure for basic plot configurations."""
        
        # Setup figure
        grid = self._setup_figure_grid(ds_raw, qubits, config.layout.title)
        
        # Track which legend entries have been created
        legend_entries_created = set()
        
        # Plot data for each qubit
        for i, (ax, name_dict) in enumerate(grid_iter(grid)):
            qubit_id = list(name_dict.values())[0]
            self._plot_generic_subplot(
                ax, ds_raw, ds_fit, qubit_id, config, i, legend_entries_created
            )
        
        # Add combined legend
        self._add_combined_legend_generic(grid)
        
        grid.fig.tight_layout(rect=[
            PlotConstants.TIGHT_LAYOUT_RECT_LEFT,
            PlotConstants.TIGHT_LAYOUT_RECT_BOTTOM,
            PlotConstants.TIGHT_LAYOUT_RECT_RIGHT,
            PlotConstants.TIGHT_LAYOUT_RECT_TOP
        ])
        return grid.fig
    
    def _plot_generic_subplot(
        self,
        ax: plt.Axes,
        ds_raw: xr.Dataset,
        ds_fit: Optional[xr.Dataset],
        qubit_id: str,
        config: PlotConfig,
        subplot_index: int,
        legend_entries_created: set
    ) -> None:
        """Plot generic data for a single subplot."""
        # Extract datasets for this qubit
        ds_qubit_raw, ds_qubit_fit = self._extract_qubit_datasets(ds_raw, ds_fit, qubit_id)
        
        # Plot raw traces
        self._plot_raw_traces(ax, ds_qubit_raw, config.traces, subplot_index, legend_entries_created)
        
        # Plot fit traces if available and successful
        if self._should_plot_fit_traces(ds_qubit_fit):
            self._plot_fit_traces(ax, ds_qubit_fit, config.fit_traces, subplot_index, legend_entries_created)
        
        # Configure axis labels and title
        self._configure_subplot_axes(ax, config.layout, qubit_id)
    
    def _plot_raw_traces(
        self,
        ax: plt.Axes,
        ds: xr.Dataset,
        traces: List[TraceConfig],
        subplot_index: int,
        legend_entries_created: set
    ) -> None:
        """Plot raw data traces on the axis."""
        for trace_config in traces:
            if self._check_trace_visibility(ds, trace_config, None):
                self._add_generic_trace(ax, ds, trace_config, subplot_index, legend_entries_created)
    
    def _plot_fit_traces(
        self,
        ax: plt.Axes,
        ds: xr.Dataset,
        traces: List[TraceConfig],
        subplot_index: int,
        legend_entries_created: set
    ) -> None:
        """Plot fit traces on the axis."""
        for trace_config in traces:
            if self._check_trace_visibility(ds, trace_config, None):
                self._add_generic_trace(ax, ds, trace_config, subplot_index, legend_entries_created, is_fit=True)
    
    def _add_combined_legend_generic(self, grid: QubitGrid) -> None:
        """Add combined legend from all subplots (without location specification)."""
        all_handles = []
        all_labels = []
        
        # Collect unique legend entries from all subplots
        for axis_row in grid.axes:
            for ax in axis_row:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label and label not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(label)
        
        if all_handles:
            grid.fig.legend(all_handles, all_labels)
    
    def _add_spectroscopy_traces(
        self,
        ax: plt.Axes,
        ds: xr.Dataset,
        traces: List[TraceConfig],
        subplot_index: int,
        legend_entries_created: set,
        is_fit: bool = False
    ) -> None:
        """Add spectroscopy-specific traces to axis."""
        
        for trace_config in traces:
            if not self._validate_trace_sources(ds, trace_config):
                continue
            
            x_data, y_data, _ = extract_trace_data(ds, trace_config.x_source, trace_config.y_source)
            
            # Extract styling
            color = self.utils.extract_matplotlib_color(trace_config.style)
            linestyle = self.utils.translate_plotly_linestyle(
                trace_config.style.get("dash", "solid")
            )
            linewidth = trace_config.style.get("width", PlotConstants.DEFAULT_LINE_WIDTH)
            
            # Only set label the first time this trace type appears
            label = ""
            if trace_config.name and trace_config.name not in legend_entries_created:
                label = trace_config.name
                legend_entries_created.add(trace_config.name)
            
            # Plot based on type
            if trace_config.plot_type in ["scatter", "line"]:
                if is_fit:
                    ax.plot(
                        x_data, y_data,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=label
                    )
                else:
                    ax.plot(
                        x_data, y_data,
                        marker='.',
                        linestyle='-',
                        color=color,
                        linewidth=linewidth,
                        label=label
                    )
    
    def _add_heatmap_trace(
        self,
        ax: plt.Axes,
        ds: xr.Dataset,
        trace_config: HeatmapTraceConfig,
        subplot_index: int
    ) -> None:
        """Add heatmap trace to axis."""
        
        if not self._validate_trace_sources(ds, trace_config):
            return
        
        # Use xarray's built-in plotting for proper 2D handling
        try:
            im = ds[trace_config.z_source].plot(
                ax=ax,
                x=trace_config.x_source,
                y=trace_config.y_source,
                add_colorbar=False,  # No colorbar for 2D heatmaps like original 02b
                cmap=self.translate_plotly_colorscale(trace_config.colorscale)
            )
            
            # Apply robust color scaling
            if hasattr(im, 'set_clim'):
                _, _, z_data = extract_trace_data(ds, trace_config.x_source, trace_config.y_source, trace_config.z_source)
                zmin, zmax = self._calculate_robust_zlimits(z_data, trace_config.zmin_percentile, trace_config.zmax_percentile)
                im.set_clim(zmin, zmax)
                
        except Exception as e:
            logger.warning(f"Failed to plot heatmap for {trace_config.name}: {e}")
    
    def _add_generic_trace(
        self,
        ax: plt.Axes,
        ds: xr.Dataset,
        trace_config: TraceConfig,
        subplot_index: int,
        legend_entries_created: set,
        is_fit: bool = False
    ) -> None:
        """Add generic trace to axis."""
        
        if not self._validate_trace_sources(ds, trace_config):
            return
        
        if trace_config.plot_type == "heatmap" and isinstance(trace_config, HeatmapTraceConfig):
            self._add_heatmap_trace(ax, ds, trace_config, subplot_index)
        else:
            # Handle as line/scatter plot
            x_data, y_data, _ = extract_trace_data(ds, trace_config.x_source, trace_config.y_source)
            
            color = self.utils.extract_matplotlib_color(trace_config.style)
            linestyle = self.utils.translate_plotly_linestyle(
                trace_config.style.get("dash", "solid")
            )
            linewidth = trace_config.style.get("width", PlotConstants.DEFAULT_LINE_WIDTH)
            
            # Only set label the first time this trace type appears
            label = ""
            if trace_config.name and trace_config.name not in legend_entries_created:
                label = trace_config.name
                legend_entries_created.add(trace_config.name)
            
            if is_fit or trace_config.plot_type == "line":
                ax.plot(
                    x_data, y_data,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label
                )
            else:
                ax.plot(
                    x_data, y_data,
                    marker='.',
                    linestyle='-',
                    color=color,
                    linewidth=linewidth,
                    label=label
                )
    
    def _add_overlays(
        self,
        ax: plt.Axes,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]]
    ) -> None:
        """Add overlay traces to matplotlib axis."""
        
        for overlay in overlays:
            if not self.overlay_renderer.should_render_overlay(ds_fit, overlay, qubit_id):
                continue
            
            if overlay.type == "line":
                self._add_line_overlay_matplotlib(ax, ds_fit, qubit_id, overlay)
            elif overlay.type == "marker":
                self._add_marker_overlay_matplotlib(ax, ds_fit, qubit_id, overlay)
    
    def _add_line_overlay_matplotlib(
        self,
        ax: plt.Axes,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlay: LineOverlayConfig
    ) -> None:
        """Add line overlay to matplotlib axis."""
        
        position = self.overlay_renderer.get_overlay_position(ds_fit, overlay.position_source, qubit_id)
        if position is None:
            return
        
        color = overlay.line_style.get("color", Colors.FIT_LINE)
        linestyle = self.utils.translate_plotly_linestyle(
            overlay.line_style.get("dash", "dash")
        )
        linewidth = overlay.line_style.get("width", PlotConstants.OVERLAY_LINE_WIDTH)
        
        if overlay.orientation == "vertical":
            ax.axvline(
                x=position,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth
            )
        elif overlay.orientation == "horizontal":
            ax.axhline(
                y=position,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth
            )
    
    def _add_marker_overlay_matplotlib(
        self,
        ax: plt.Axes,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlay: MarkerOverlayConfig
    ) -> None:
        """Add marker overlay to matplotlib axis."""
        
        x_pos = self.overlay_renderer.get_overlay_position(ds_fit, overlay.x_source, qubit_id)
        y_pos = self.overlay_renderer.get_overlay_position(ds_fit, overlay.y_source, qubit_id)
        
        if x_pos is None or y_pos is None:
            return
        
        # Translate Plotly marker style to matplotlib
        color = overlay.marker_style.get("color", Colors.OPTIMAL_MARKER)
        size = overlay.marker_style.get("size", PlotConstants.OVERLAY_MARKER_SIZE)
        symbol = overlay.marker_style.get("symbol", "x")
        
        # Map Plotly symbols to matplotlib markers
        marker_map = {
            "x": "x",
            "circle": "o",
            "square": "s",
            "diamond": "D",
            "cross": "+",
            "triangle-up": "^",
            "triangle-down": "v"
        }
        marker = marker_map.get(symbol, "x")
        
        ax.plot(
            x_pos, y_pos,
            marker=marker,
            color=color,
            markersize=size,
            linestyle='None'
        )
    
    def _add_dual_axis(self, ax: plt.Axes, ds: xr.Dataset, dual_config) -> None:
        """Add dual axis (top x-axis) to matplotlib axis."""
        
        if not self._validate_dual_axis_source(ds, dual_config):
            return
        
        # Create twin axis
        ax_top = ax.twiny()
        
        # Get coordinate and data variable
        main_x_coord = self._get_main_coordinate(ds)
        data_var = self._get_data_variable(ds)
        
        if main_x_coord is None or data_var is None:
            return
        
        # Create and plot dual axis
        self._create_dual_axis_plot(ax_top, ds, dual_config, main_x_coord, data_var)
    
    def _validate_dual_axis_source(self, ds: xr.Dataset, dual_config) -> bool:
        """Validate that the dual axis source exists in the dataset."""
        return dual_config.top_axis_source in ds
    
    def _get_main_coordinate(self, ds: xr.Dataset) -> Optional[str]:
        """Find the main x-coordinate that was used for the main plot."""
        if CoordinateNames.AMP_MV in ds.coords:
            return CoordinateNames.AMP_MV
        elif CoordinateNames.FREQUENCY in ds.coords:
            return CoordinateNames.FREQUENCY
        return None
    
    def _get_data_variable(self, ds: xr.Dataset) -> Optional[str]:
        """Find the data variable that was plotted."""
        if CoordinateNames.I in ds.data_vars:
            return CoordinateNames.I
        elif CoordinateNames.STATE in ds.data_vars:
            return CoordinateNames.STATE
        return None
    
    def _create_dual_axis_plot(
        self,
        ax_top: plt.Axes,
        ds: xr.Dataset,
        dual_config,
        main_x_coord: str,
        data_var: str
    ) -> None:
        """Create the dual axis plot."""
        try:
            # Create dataset with reassigned coordinates
            ds_twin = ds.assign_coords({main_x_coord: ds[dual_config.top_axis_source]})
            
            # Plot based on dimensionality
            if self._is_2d_plot(ds):
                self._plot_2d_dual_axis(ax_top, ds_twin, data_var, main_x_coord)
            else:
                self._plot_1d_dual_axis(ax_top, ds_twin, data_var, main_x_coord)
            
            # Configure the dual axis
            self._configure_dual_axis(ax_top, dual_config)
            
        except Exception as e:
            logger.warning(f"Could not create twin axis plot using assign_coords approach: {e}")
    
    def _is_2d_plot(self, ds: xr.Dataset) -> bool:
        """Check if this is a 2D plot based on number of pulses dimension."""
        return (CoordinateNames.NB_OF_PULSES in ds.dims and 
                ds.sizes.get(CoordinateNames.NB_OF_PULSES, 1) > 1)
    
    def _plot_2d_dual_axis(
        self,
        ax_top: plt.Axes,
        ds_twin: xr.Dataset,
        data_var: str,
        main_x_coord: str
    ) -> None:
        """Plot 2D heatmap on dual axis."""
        # Use pcolormesh for 2D plotting to avoid add_labels issue
        x_data = ds_twin[main_x_coord].values
        y_data = ds_twin[CoordinateNames.NB_OF_PULSES].values
        z_data = ds_twin[data_var].values
        ax_top.pcolormesh(x_data, y_data, z_data, alpha=0)  # Make invisible, just for axis scaling
    
    def _plot_1d_dual_axis(
        self,
        ax_top: plt.Axes,
        ds_twin: xr.Dataset,
        data_var: str,
        main_x_coord: str
    ) -> None:
        """Plot 1D line on dual axis."""
        # Extract data values for manual plotting to avoid add_labels issue
        x_data = ds_twin[main_x_coord].values
        y_data = ds_twin[data_var].values
        ax_top.plot(x_data, y_data, alpha=0)  # Make invisible, just for axis scaling
    
    def _configure_dual_axis(self, ax_top: plt.Axes, dual_config) -> None:
        """Configure the dual axis properties."""
        ax_top.set_xlabel(dual_config.top_axis_title)
        ax_top.grid(False)  # Disable grid on secondary axis
    
    
    def _add_overlays_flux_spectroscopy_matplotlib(
        self,
        ax: plt.Axes,
        ds_qubit_fit: xr.Dataset,
        qubit_id: str,
        overlays: List,
        ds_raw: xr.Dataset
    ) -> None:
        """Add flux spectroscopy overlays to matplotlib axis (EXACT copy of original logic)."""
        
        # Use base class validation method
        if not self._validate_overlay_fit(ds_qubit_fit, qubit_id):
            return
            
        # Extract parameters using base class method
        parameter_map = {
            'idle_offset': 'idle_offset',
            'flux_min': 'flux_min',
            'sweet_spot_frequency': 'sweet_spot_frequency'
        }
        unit_conversions = {
            'sweet_spot_frequency': PlotConstants.GHZ_PER_HZ,  # Convert Hz to GHz
            'idle_offset': 1e-3,  # Convert mV to V
            'flux_min': 1e-3  # Convert mV to V
        }
        
        params = self._extract_overlay_parameters(ds_qubit_fit, parameter_map, unit_conversions)
        
        if not all(key in params for key in ['idle_offset', 'flux_min', 'sweet_spot_frequency']):
            logger.warning(f"Missing required parameters for flux spectroscopy overlays in qubit {qubit_id}")
            return
            
        # Get frequency range using base class method
        freq_range = self._get_frequency_range(ds_raw, qubit_id)
        if freq_range is None:
            logger.warning(f"Could not get frequency range for flux spectroscopy overlays")
            return
            
        freq_min, freq_max = freq_range
        
        try:
            
            # Red dashed vertical line at idle_offset
            ax.axvline(
                params['idle_offset'],
                linestyle="dashed",
                linewidth=2.5,
                color="#FF0000",
                label="idle offset",
            )
            
            # Purple dashed vertical line at flux_min
            ax.axvline(
                params['flux_min'],
                linestyle="dashed",
                linewidth=2.5,
                color="#800080",
                label="min offset",
            )
            
            # Magenta star marker at sweet spot
            ax.plot(
                params['idle_offset'],
                params['sweet_spot_frequency'],
                marker="*",
                color="#FF00FF",
                markersize=18,
                linestyle="None",
            )
                
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Could not add flux spectroscopy overlays to matplotlib for {qubit_id}: {e}")
            return