"""
Enhanced Matplotlib rendering engine for the unified plotting framework.

This engine provides specialized Matplotlib rendering with support for complex
plot types and ensures visual compatibility with the Plotly engine outputs.
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure as MatplotlibFigure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from ..configs import (Colors, HeatmapConfig, HeatmapTraceConfig,
                       LineOverlayConfig, LineStyles, MarkerOverlayConfig,
                       PlotConfig, SpectroscopyConfig, TraceConfig,
                       get_standard_matplotlib_size)
from ..grids import QubitGrid, grid_iter
from .common import GridManager, MatplotlibEngineUtils, OverlayRenderer
from .data_validators import DataValidator


class MatplotlibEngine:
    """Enhanced Matplotlib rendering engine with specialized plot type support."""
    
    def __init__(self):
        self.utils = MatplotlibEngineUtils()
        self.overlay_renderer = OverlayRenderer()
        self.grid_manager = GridManager()
        self.data_validator = DataValidator()
    
    def create_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        plot_configs: List[PlotConfig],
        ds_fit: Optional[xr.Dataset] = None,
    ) -> MatplotlibFigure:
        """
        Create a Matplotlib figure using standard configuration.
        
        Args:
            ds_raw: Raw experimental dataset
            qubits: List of qubits to plot
            plot_configs: List of plot configurations
            ds_fit: Optional fitted dataset
            
        Returns:
            Matplotlib figure object
        """
        if not plot_configs:
            fig, _ = plt.subplots()
            return fig
        
        config = plot_configs[0]  # Use first config for now
        
        # Route to specialized handlers based on config type
        if isinstance(config, SpectroscopyConfig):
            return self.create_spectroscopy_figure(ds_raw, qubits, config, ds_fit)
        elif isinstance(config, HeatmapConfig):
            return self.create_heatmap_figure(ds_raw, qubits, config, ds_fit)
        else:
            # Generic fallback
            return self._create_generic_figure(ds_raw, qubits, config, ds_fit)
    
    def create_spectroscopy_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: SpectroscopyConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> MatplotlibFigure:
        """Create specialized figure for 1D spectroscopy plots."""
        
        grid = self.grid_manager.create_grid(ds_raw, qubits, create_figure=True)
        grid.fig.set_size_inches(*get_standard_matplotlib_size())
        grid.fig.suptitle(config.layout.title)
        
        # Track which legend entries have been created
        legend_entries_created = set()
        
        for i, (ax, name_dict) in enumerate(grid_iter(grid)):
            qubit_id = list(name_dict.values())[0]
            ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
            ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None
            
            # Plot raw data traces
            self._add_spectroscopy_traces(ax, ds_qubit_raw, config.traces, i, legend_entries_created)
            
            # Plot fit traces - only if fit was successful
            if ds_qubit_fit is not None and self._is_fit_successful(ds_qubit_fit):
                self._add_spectroscopy_traces(ax, ds_qubit_fit, config.fit_traces, i, legend_entries_created, is_fit=True)
            
            # Set labels and title
            ax.set_xlabel(config.layout.x_axis_title)
            ax.set_ylabel(config.layout.y_axis_title)
            ax.set_title(f"Qubit {qubit_id}")
            
            # Add dual axis if configured
            if config.dual_axis and config.dual_axis.enabled:
                self._add_dual_axis_matplotlib(ax, ds_qubit_raw, config.dual_axis)
        
        # Add legend by collecting handles and labels from all subplots
        all_handles = []
        all_labels = []
        # grid.axes is a list of lists (2D structure), need to flatten it
        for axis_row in grid.axes:
            for ax in axis_row:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label and label not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(label)
        
        if all_handles:
            grid.fig.legend(all_handles, all_labels, loc='upper right')
        
        grid.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return grid.fig
    
    def create_heatmap_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: HeatmapConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> MatplotlibFigure:
        """Create specialized figure for 2D heatmap plots."""
        
        grid = self.grid_manager.create_grid(ds_raw, qubits, create_figure=True)
        grid.fig.set_size_inches(*get_standard_matplotlib_size())
        grid.fig.suptitle(config.layout.title)
        
        for i, (ax, name_dict) in enumerate(grid_iter(grid)):
            qubit_id = list(name_dict.values())[0]
            ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
            ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None
            
            # Plot heatmap traces
            for trace_config in config.traces:
                if self._check_trace_visibility(ds_qubit_raw, trace_config, qubit_id):
                    self._add_heatmap_trace(ax, ds_qubit_raw, trace_config, i)
            
            # Add overlays if fit data available
            if ds_qubit_fit is not None and config.overlays:
                # Route to appropriate overlay handler based on experiment type
                if self._is_flux_spectroscopy(ds_raw):
                    self._add_overlays_flux_spectroscopy_matplotlib(ax, ds_qubit_fit, qubit_id, config.overlays, ds_raw)
                else:
                    self._add_overlays_matplotlib(ax, ds_qubit_fit, qubit_id, config.overlays)
            
            # Set labels and title
            ax.set_xlabel(config.layout.x_axis_title)
            ax.set_ylabel(config.layout.y_axis_title)
            ax.set_title(f"Qubit {qubit_id}")
        
        grid.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return grid.fig
    
    def _create_generic_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: PlotConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> MatplotlibFigure:
        """Create generic figure for basic plot configurations."""
        
        grid = self.grid_manager.create_grid(ds_raw, qubits, create_figure=True)
        grid.fig.set_size_inches(*get_standard_matplotlib_size())
        grid.fig.suptitle(config.layout.title)
        
        # Track which legend entries have been created
        legend_entries_created = set()
        
        for i, (ax, name_dict) in enumerate(grid_iter(grid)):
            qubit_id = list(name_dict.values())[0]
            ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
            ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None
            
            # Plot raw traces
            for trace_config in config.traces:
                if self._check_trace_visibility(ds_qubit_raw, trace_config, qubit_id):
                    self._add_generic_trace(ax, ds_qubit_raw, trace_config, i, legend_entries_created)
            
            # Plot fit traces - only if fit was successful
            if ds_qubit_fit is not None and self._is_fit_successful(ds_qubit_fit):
                for trace_config in config.fit_traces:
                    if self._check_trace_visibility(ds_qubit_fit, trace_config, qubit_id):
                        self._add_generic_trace(ax, ds_qubit_fit, trace_config, i, legend_entries_created, is_fit=True)
            
            # Set labels and title
            ax.set_xlabel(config.layout.x_axis_title)
            ax.set_ylabel(config.layout.y_axis_title)
            ax.set_title(f"Qubit {qubit_id}")
        
        # Add legend by collecting handles and labels from all subplots
        all_handles = []
        all_labels = []
        # grid.axes is a list of lists (2D structure), need to flatten it
        for axis_row in grid.axes:
            for ax in axis_row:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label and label not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(label)
        
        if all_handles:
            grid.fig.legend(all_handles, all_labels)
        
        grid.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return grid.fig
    
    def _add_spectroscopy_traces(
        self,
        ax,
        ds: xr.Dataset,
        traces: List[TraceConfig],
        subplot_index: int,
        legend_entries_created: set,
        is_fit: bool = False
    ):
        """Add spectroscopy-specific traces to axis."""
        
        for trace_config in traces:
            if not self._validate_trace_sources(ds, trace_config):
                continue
            
            x_data = ds[trace_config.x_source].values
            y_data = ds[trace_config.y_source].values
            
            # Extract styling
            color = self.utils.extract_matplotlib_color(trace_config.style)
            linestyle = self.utils.translate_plotly_linestyle(
                trace_config.style.get("dash", "solid")
            )
            linewidth = trace_config.style.get("width", LineStyles.RAW_LINE_WIDTH)
            
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
        ax,
        ds: xr.Dataset,
        trace_config: HeatmapTraceConfig,
        subplot_index: int
    ):
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
                cmap=self._translate_plotly_colorscale(trace_config.colorscale)
            )
            
            # Apply robust color scaling
            if hasattr(im, 'set_clim'):
                z_data = ds[trace_config.z_source].values
                zmin, zmax = self._calculate_robust_zlimits(z_data, trace_config.zmin_percentile, trace_config.zmax_percentile)
                im.set_clim(zmin, zmax)
                
        except Exception as e:
            print(f"Warning: Failed to plot heatmap for {trace_config.name}: {e}")
    
    def _add_generic_trace(
        self,
        ax,
        ds: xr.Dataset,
        trace_config: TraceConfig,
        subplot_index: int,
        legend_entries_created: set,
        is_fit: bool = False
    ):
        """Add generic trace to axis."""
        
        if not self._validate_trace_sources(ds, trace_config):
            return
        
        if trace_config.plot_type == "heatmap" and isinstance(trace_config, HeatmapTraceConfig):
            self._add_heatmap_trace(ax, ds, trace_config, subplot_index)
        else:
            # Handle as line/scatter plot
            x_data = ds[trace_config.x_source].values
            y_data = ds[trace_config.y_source].values
            
            color = self.utils.extract_matplotlib_color(trace_config.style)
            linestyle = self.utils.translate_plotly_linestyle(
                trace_config.style.get("dash", "solid")
            )
            linewidth = trace_config.style.get("width", LineStyles.RAW_LINE_WIDTH)
            
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
    
    def _add_overlays_matplotlib(
        self,
        ax,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]]
    ):
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
        ax,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlay: LineOverlayConfig
    ):
        """Add line overlay to matplotlib axis."""
        
        position = self.overlay_renderer.get_overlay_position(ds_fit, overlay.position_source, qubit_id)
        if position is None:
            return
        
        color = overlay.line_style.get("color", Colors.FIT_LINE)
        linestyle = self.utils.translate_plotly_linestyle(
            overlay.line_style.get("dash", "dash")
        )
        linewidth = overlay.line_style.get("width", LineStyles.OVERLAY_LINE_WIDTH)
        
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
        ax,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlay: MarkerOverlayConfig
    ):
        """Add marker overlay to matplotlib axis."""
        
        x_pos = self.overlay_renderer.get_overlay_position(ds_fit, overlay.x_source, qubit_id)
        y_pos = self.overlay_renderer.get_overlay_position(ds_fit, overlay.y_source, qubit_id)
        
        if x_pos is None or y_pos is None:
            return
        
        # Translate Plotly marker style to matplotlib
        color = overlay.marker_style.get("color", Colors.OPTIMAL_MARKER)
        size = overlay.marker_style.get("size", LineStyles.MARKER_SIZE)
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
    
    def _add_dual_axis_matplotlib(self, ax, ds: xr.Dataset, dual_config):
        """Add dual axis (top x-axis) to matplotlib axis."""
        
        if dual_config.top_axis_source not in ds:
            return
        
        # Create twin axis (exactly like original: ax2 = ax.twiny())
        ax_top = ax.twiny()
        
        # Use EXACT same approach as original plotting.py:
        # ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit][data].plot(ax=ax2, x="amp_mV", alpha=RAW_DATA_ALPHA)
        
        # Find the main x-coordinate that was used for the main plot
        main_x_coord = None
        if 'amp_mV' in ds.coords:
            main_x_coord = 'amp_mV'
        elif 'frequency' in ds.coords:
            main_x_coord = 'frequency'
        
        if main_x_coord is None:
            return
        
        # Find the data variable that was plotted
        data_var = None
        if 'I' in ds.data_vars:
            data_var = 'I'
        elif 'state' in ds.data_vars:
            data_var = 'state'
        
        if data_var is None:
            return
        
        try:
            ds_twin = ds.assign_coords({main_x_coord: ds[dual_config.top_axis_source]})
            
            # Plot on twin axis with alpha=0 to make it invisible (just for axis scaling)
            if 'nb_of_pulses' in ds.dims and ds.sizes.get('nb_of_pulses', 1) > 1:
                                     # 2D case - plot heatmap
                     ds_twin[data_var].plot(
                         ax=ax_top,
                         x=main_x_coord,
                         y='nb_of_pulses',
                         add_colorbar=False,
                         alpha=0,  # Make invisible, just for axis scaling
                         add_labels=False  # Suppress automatic title generation
                     )
            else:
                                     # 1D case - plot line
                     ds_twin[data_var].plot(
                         ax=ax_top,
                         x=main_x_coord,
                         alpha=0,  # Make invisible, just for axis scaling
                         add_labels=False  # Suppress automatic title generation
                     )
            
            # Set the axis title
            ax_top.set_xlabel(dual_config.top_axis_title)
            ax_top.grid(False)  # Disable grid on secondary axis
            
        except Exception as e:
            print(f"Warning: Could not create twin axis plot using assign_coords approach: {e}")
            # Don't fallback, just skip the dual axis if it fails
    
    def _check_trace_visibility(self, ds: xr.Dataset, trace_config: TraceConfig, qubit_id: str) -> bool:
        """Check if a trace should be visible based on conditions."""
        if not trace_config.visible:
            return False
        
        if trace_config.condition_source and trace_config.condition_source in ds:
            condition_value = ds[trace_config.condition_source].values
            # Handle scalar and array conditions
            if np.isscalar(condition_value):
                return condition_value == trace_config.condition_value
            else:
                # For array conditions, check if any values match
                return np.any(condition_value == trace_config.condition_value)
        
        return True
    
    def _validate_trace_sources(self, ds: xr.Dataset, trace_config: TraceConfig) -> bool:
        """Validate that all required data sources exist in dataset."""
        required_sources = [trace_config.x_source, trace_config.y_source]
        
        if isinstance(trace_config, HeatmapTraceConfig) and trace_config.z_source:
            required_sources.append(trace_config.z_source)
        
        return all(source in ds for source in required_sources)
    
    def _translate_plotly_colorscale(self, plotly_colorscale: str) -> str:
        """Translate Plotly colorscale to matplotlib colormap."""
        colorscale_map = {
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Magma": "magma",
            "Blues": "Blues",
            "Reds": "Reds",
            "binary": "binary",
            "RdBu": "RdBu",
        }
        return colorscale_map.get(plotly_colorscale, "viridis")
    
    def _calculate_robust_zlimits(self, z_data: np.ndarray, zmin_percentile: float = 2.0, zmax_percentile: float = 98.0):
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
    
    def _is_flux_spectroscopy(self, ds_raw: xr.Dataset) -> bool:
        """Detect if dataset is for flux spectroscopy experiment."""
        # Check for characteristic flux spectroscopy coordinates
        flux_indicators = ["flux_bias", "attenuated_current"]
        power_indicators = ["power", "power_dbm"]
        
        has_flux = any(coord in ds_raw.coords for coord in flux_indicators)
        has_power = any(coord in ds_raw.coords for coord in power_indicators)
        
        # Flux spectroscopy has flux_bias but no power coordinates
        return has_flux and not has_power
    
    def _is_fit_successful(self, ds_fit: xr.Dataset) -> bool:
        """Check if fit was successful - matches original plotting.py logic."""
        return hasattr(ds_fit, "outcome") and getattr(ds_fit.outcome, "values", None) == "successful"
    
    def _add_overlays_flux_spectroscopy_matplotlib(
        self,
        ax,
        ds_qubit_fit: xr.Dataset,
        qubit_id: str,
        overlays: List,
        ds_raw: xr.Dataset
    ):
        """Add flux spectroscopy overlays to matplotlib axis (EXACT copy of original logic)."""
        
        # Check if fit was successful (exact same check as original)
        if not (hasattr(ds_qubit_fit, "outcome") and ds_qubit_fit.outcome.values == "successful"):
            return
            
        try:
            # Extract values exactly like original matplotlib code in plotting.py
            if hasattr(ds_qubit_fit, 'fit_results'):
                # Use fit_results like original code  
                idle_offset = ds_qubit_fit.fit_results.idle_offset
                flux_min = ds_qubit_fit.fit_results.flux_min
                sweet_spot_frequency = ds_qubit_fit.fit_results.sweet_spot_frequency.values * 1e-9  # Convert Hz to GHz
                
                # Red dashed vertical line at idle_offset (EXACT copy of original)
                ax.axvline(
                    idle_offset,
                    linestyle="dashed",      # IDLE_OFFSET_LINESTYLE from original
                    linewidth=2.5,           # IDLE_OFFSET_LINEWIDTH from original  
                    color="#FF0000",         # IDLE_OFFSET_COLOR from original
                    label="idle offset",
                )
                
                # Purple dashed vertical line at flux_min (EXACT copy of original)
                ax.axvline(
                    flux_min,
                    linestyle="dashed",      # IDLE_OFFSET_LINESTYLE from original
                    linewidth=2.5,           # IDLE_OFFSET_LINEWIDTH from original
                    color="#800080",         # MIN_OFFSET_COLOR from original
                    label="min offset",
                )
                
                # Magenta star marker at sweet spot (EXACT copy of original)
                ax.plot(
                    idle_offset.values,
                    sweet_spot_frequency,
                    marker="*",              # SWEET_SPOT_MARKER from original
                    color="#FF00FF",         # SWEET_SPOT_COLOR from original  
                    markersize=18,           # SWEET_SPOT_MARKERSIZE from original
                    linestyle="None",
                )
            else:
                # Fallback for different fit dataset structure
                idle_offset = ds_qubit_fit.idle_offset.values * 1e-3  # Convert mV to V
                flux_min = ds_qubit_fit.flux_min.values * 1e-3  # Convert mV to V
                sweet_spot_frequency = ds_qubit_fit.sweet_spot_frequency.values * 1e-9  # Convert Hz to GHz
                
                ax.axvline(idle_offset, linestyle="dashed", linewidth=2.5, color="#FF0000")
                ax.axvline(flux_min, linestyle="dashed", linewidth=2.5, color="#800080")
                ax.plot(idle_offset, sweet_spot_frequency, marker="*", color="#FF00FF", markersize=18, linestyle="None")
                
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Warning: Could not add flux spectroscopy overlays to matplotlib for {qubit_id}: {e}")
            return