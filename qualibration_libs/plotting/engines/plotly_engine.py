"""
Enhanced Plotly rendering engine for the unified plotting framework.

This engine provides specialized Plotly rendering with support for complex
plot types including adaptive 1D/2D plots, heatmaps with overlays, and
dual-axis configurations.
"""

from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from ..configs import (DualAxisConfig, HeatmapConfig,
                       HeatmapTraceConfig, LineOverlayConfig,
                       MarkerOverlayConfig, PlotConfig, SpectroscopyConfig,
                       TraceConfig, get_standard_plotly_style)
from ..configs.constants import CoordinateNames, PlotConstants, ExperimentTypes, ColorScales
from .common import GridManager, OverlayRenderer, PlotlyEngineUtils
from .data_validators import DataValidator
from .base_engine import BaseRenderingEngine
from .experiment_detector import ExperimentDetector
from ..exceptions import (
    PlottingError
)
from ..data_utils import DataExtractor, DataValidator as DataValidatorUtils, ArrayManipulator, UnitConverter

logger = logging.getLogger(__name__)


class PlotlyEngine(BaseRenderingEngine):
    """Enhanced Plotly rendering engine with specialized plot type support.
    
    This engine provides specialized Plotly rendering capabilities for creating
    complex scientific plots including spectroscopy plots, heatmaps with overlays,
    and dual-axis configurations. It extends the base rendering engine with
    Plotly-specific implementations.
    
    Attributes:
        utils: Utility functions for Plotly-specific operations
        overlay_renderer: Handles rendering of overlay elements on plots
        grid_manager: Manages grid layouts for multi-qubit visualizations
        data_validator: Validates data compatibility with plot configurations
        experiment_detector: Detects experiment types from datasets
    """
    
    def __init__(self) -> None:
        """Initialize the Plotly rendering engine.
        
        Sets up the engine with necessary utilities and managers for rendering
        complex scientific plots using Plotly.
        """
        super().__init__()
        self.utils = PlotlyEngineUtils()
        self.overlay_renderer = OverlayRenderer()
        self.grid_manager = GridManager()
        self.data_validator = DataValidator()
        self.experiment_detector = ExperimentDetector()
    
    def _create_empty_figure(self) -> go.Figure:
        """Create an empty figure when no configs provided.
        
        Returns:
            go.Figure: An empty Plotly figure with standard styling applied.
        """
        fig = go.Figure()
        fig.update_layout(**get_standard_plotly_style())
        return fig
    
    def create_spectroscopy_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: SpectroscopyConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> go.Figure:
        """Create specialized figure for 1D spectroscopy plots.
        
        Args:
            ds_raw: Raw dataset containing spectroscopy measurements
            qubits: List of qubit objects to plot
            config: Configuration for spectroscopy plot styling
            ds_fit: Optional fit dataset containing analysis results
            
        Returns:
            go.Figure: Plotly figure with spectroscopy plots for each qubit
        """
        
        # Use base class methods for grid creation
        grid = self._create_grid_layout(ds_raw, qubits, create_figure=False)
        spacing = self._get_subplot_spacing("standard")
        titles = self._generate_subplot_titles(grid)
        
        fig = make_subplots(
            rows=grid.n_rows,
            cols=grid.n_cols,
            subplot_titles=titles,
            horizontal_spacing=spacing["horizontal"],
            vertical_spacing=spacing["vertical"],
        )
        
        # Add traces for each qubit
        for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
            row = grid_row + 1
            col = grid_col + 1
            qubit_id = list(name_dict.values())[0]
            
            ds_qubit_raw, ds_qubit_fit = self._extract_qubit_datasets(ds_raw, ds_fit, qubit_id)
            
            # Add raw data traces
            self._add_spectroscopy_traces(fig, ds_qubit_raw, config.traces, row, col, i)
            
            # Add fit traces
            if ds_qubit_fit is not None:
                self._add_spectroscopy_traces(fig, ds_qubit_fit, config.fit_traces, row, col, i, is_fit=True)
            
            # Update axes
            fig.update_xaxes(title_text=config.layout.x_axis_title, row=row, col=col)
            fig.update_yaxes(title_text=config.layout.y_axis_title, row=row, col=col)
            
            # Add dual axis if configured
            if config.dual_axis and config.dual_axis.enabled:
                self._add_dual_axis(fig, ds_qubit_raw, config.dual_axis, row, col, grid.n_cols)
        
        # Apply layout settings using base class dimensions
        dimensions = self._calculate_figure_dimensions(grid.n_rows, grid.n_cols, "standard")
        layout_settings = get_standard_plotly_style()
        layout_settings.update({
            "title_text": config.layout.title,
            "height": dimensions["height"],
            "width": dimensions["width"]
        })
        
        fig.update_layout(**layout_settings)
        return fig
    
    def create_heatmap_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: HeatmapConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> go.Figure:
        """Create specialized figure for 2D heatmap plots.
        
        Args:
            ds_raw: Raw dataset containing 2D measurement data
            qubits: List of qubit objects to plot
            config: Configuration for heatmap plot styling
            ds_fit: Optional fit dataset containing analysis results and overlays
            
        Returns:
            go.Figure: Plotly figure with heatmap plots for each qubit
        """
        
        # Use base class methods for grid creation
        grid = self._create_grid_layout(ds_raw, qubits, create_figure=False)
        # Determine plot type for spacing
        plot_type = "flux" if self._is_flux_spectroscopy(ds_raw) else "heatmap"
        spacing = self._get_subplot_spacing(plot_type, config.subplot_spacing if hasattr(config, 'subplot_spacing') else None)
        titles = self._generate_subplot_titles(grid)
        
        fig = make_subplots(
            rows=grid.n_rows,
            cols=grid.n_cols,
            subplot_titles=titles,
            horizontal_spacing=spacing["horizontal"],
            vertical_spacing=spacing["vertical"],
        )
        
        heatmap_info = []
        
        # Add traces for each qubit
        for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
            row = grid_row + 1
            col = grid_col + 1
            qubit_id = list(name_dict.values())[0]
            
            # For multi-qubit datasets with 2D frequency arrays, handle differently
            ds_qubit_raw, ds_qubit_fit = self._extract_qubit_datasets(ds_raw, ds_fit, qubit_id)
            
            # Add heatmap traces  
            for trace_config in config.traces:
                if self._check_trace_visibility(ds_qubit_raw, trace_config, qubit_id):
                    # For 2D heatmaps, route to specialized handlers
                    if hasattr(config, 'traces') and len(config.traces) > 0 and config.traces[0].plot_type == "heatmap":
                        # Detect experiment type to choose correct handler
                        if self.experiment_detector.detect_experiment_type(ds_raw) == "flux_spectroscopy":
                            self._add_heatmap_trace_flux_spectroscopy(fig, ds_raw, trace_config, row, col, qubit_id, i)
                        elif self.experiment_detector.detect_experiment_type(ds_raw) == "power_rabi":
                            self._add_heatmap_trace_power_rabi(fig, ds_qubit_raw, trace_config, row, col)
                        else:
                            # Default to amplitude/power spectroscopy (02b)
                            self._add_heatmap_trace_multi_qubit(fig, ds_raw, trace_config, row, col, qubit_id, i)
                    else:
                        self._add_heatmap_trace(fig, ds_qubit_raw, trace_config, row, col)
                    heatmap_info.append((len(fig.data) - 1, row, col))
            
            # Add overlays if fit data available  
            if ds_qubit_fit is not None and config.overlays:
                # For multi-qubit heatmaps, handle overlays specially
                if hasattr(config, 'traces') and len(config.traces) > 0 and config.traces[0].plot_type == "heatmap":
                    # Route overlays to correct experiment type
                    if self.experiment_detector.detect_experiment_type(ds_raw) == "flux_spectroscopy":
                        self._add_overlays_flux_spectroscopy(fig, ds_fit, qubit_id, config.overlays, row, col, ds_raw)
                    elif self.experiment_detector.detect_experiment_type(ds_raw) == "power_rabi":
                        self._add_overlays_power_rabi(fig, ds_qubit_fit, qubit_id, config.overlays, row, col, ds_raw)
                    else:
                        # Default to amplitude/power spectroscopy (02b)
                        self._add_overlays_multi_qubit(fig, ds_fit, qubit_id, config.overlays, row, col)
                else:
                    self._add_overlays(fig, ds_qubit_fit, qubit_id, config.overlays, row, col)
            
            # Update axes
            fig.update_xaxes(title_text=config.layout.x_axis_title, row=row, col=col)
            fig.update_yaxes(title_text=config.layout.y_axis_title, row=row, col=col)
            
            # Add dual axis if configured
            if config.dual_axis and config.dual_axis.enabled:
                self._add_dual_axis(fig, ds_qubit_raw, config.dual_axis, row, col, grid.n_cols)
        
        # Position colorbars
        self._position_colorbars(fig, heatmap_info, grid.n_cols, config.traces)
        
        # Apply layout settings using base class dimensions
        dimensions = self._calculate_figure_dimensions(grid.n_rows, grid.n_cols, "heatmap")
        layout_settings = get_standard_plotly_style()
        layout_settings.update({
            "title_text": config.layout.title,
            "showlegend": False,  # Heatmaps typically don't show legends
            "height": dimensions["height"],
            "width": dimensions["width"]
        })
        
        fig.update_layout(**layout_settings)
        return fig
    
    
    def _create_generic_figure(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        config: PlotConfig,
        ds_fit: Optional[xr.Dataset] = None,
    ) -> go.Figure:
        """Create generic figure for basic plot configurations.
        
        Args:
            ds_raw: Raw dataset containing measurement data
            qubits: List of qubit objects to plot
            config: Generic plot configuration
            ds_fit: Optional fit dataset containing analysis results
            
        Returns:
            go.Figure: Plotly figure with plots for each qubit
        """
        
        # Use base class methods for grid creation
        grid = self._create_grid_layout(ds_raw, qubits, create_figure=False)
        spacing = self._get_subplot_spacing("standard")
        titles = self._generate_subplot_titles(grid)
        
        fig = make_subplots(
            rows=grid.n_rows,
            cols=grid.n_cols,
            subplot_titles=titles,
            horizontal_spacing=spacing["horizontal"],
            vertical_spacing=spacing["vertical"],
        )
        
        # Add traces for each qubit
        for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
            row = grid_row + 1
            col = grid_col + 1
            qubit_id = list(name_dict.values())[0]
            
            ds_qubit_raw, ds_qubit_fit = self._extract_qubit_datasets(ds_raw, ds_fit, qubit_id)
            
            # Add traces based on type
            for trace_config in config.traces + config.fit_traces:
                if not self._check_trace_visibility(ds_qubit_raw, trace_config, qubit_id):
                    continue
                
                is_fit_trace = trace_config in config.fit_traces
                ds_source = ds_qubit_fit if is_fit_trace else ds_qubit_raw
                
                if ds_source is None or not self._validate_trace_sources(ds_source, trace_config):
                    continue
                
                self._add_generic_trace(fig, ds_source, trace_config, row, col, i)
            
            # Update axes
            fig.update_xaxes(title_text=config.layout.x_axis_title, row=row, col=col)
            fig.update_yaxes(title_text=config.layout.y_axis_title, row=row, col=col)
        
        # Apply layout settings
        fig.update_layout(**get_standard_plotly_style(), title_text=config.layout.title)
        return fig
    
    def _add_spectroscopy_traces(
        self,
        fig: go.Figure,
        ds: xr.Dataset,
        traces: List[TraceConfig],
        row: int,
        col: int,
        subplot_index: int,
        is_fit: bool = False
    ) -> None:
        """Add spectroscopy-specific traces to figure."""
        
        for trace_config in traces:
            # For fit traces, ensure the fit was successful before plotting
            if is_fit and (CoordinateNames.OUTCOME not in ds.coords or ds.outcome != CoordinateNames.SUCCESSFUL):
                continue

            if not self._validate_trace_sources(ds, trace_config):
                continue
            
            # Build custom data
            custom_data = self._build_custom_data(ds, trace_config.custom_data_sources)
            
            # Create trace properties
            trace_props = {
                "name": trace_config.name,
                "hovertemplate": trace_config.hover_template,
                "customdata": custom_data,
                "legendgroup": trace_config.name,
                "showlegend": subplot_index == 0,  # Only show legend for first subplot
            }
            
            # Create appropriate trace type
            if trace_config.plot_type == "scatter" or trace_config.plot_type == "line":
                trace = go.Scatter(
                    x=ds[trace_config.x_source].values,
                    y=ds[trace_config.y_source].values,
                    mode=trace_config.mode,
                    line=trace_config.style,
                    **trace_props,
                )
            else:
                continue  # Skip unsupported types for spectroscopy
            
            fig.add_trace(trace, row=row, col=col)
    
    def _add_heatmap_trace(
        self,
        fig: go.Figure,
        ds: xr.Dataset,
        trace_config: HeatmapTraceConfig,
        row: int,
        col: int
    ) -> None:
        """Add heatmap trace to figure."""
        
        if not self._validate_trace_sources(ds, trace_config):
            return
        
        # Get data arrays
        x_data = ds[trace_config.x_source].values
        y_data = ds[trace_config.y_source].values
        z_data = ds[trace_config.z_source].values
        
        # Ensure proper shape for heatmap - handle 2D heatmaps properly
        if z_data.ndim == 2:
            # For 2D heatmaps, we need to ensure z_data is shaped correctly
            # The original 02b code transposes and reshapes as needed
            z_data = z_data.T
            if z_data.shape[0] != len(y_data):
                z_data = z_data.T
            nan_info = DataValidatorUtils.check_for_nans(z_data, raise_on_all_nan=False)
            if nan_info['all_nans']:
                z_data = np.zeros_like(z_data)
        elif z_data.ndim == 1:
            z_data = z_data[np.newaxis, :]
        
        # Calculate robust z-limits
        zmin, zmax = self._calculate_robust_zlimits(
            z_data, trace_config.zmin_percentile, trace_config.zmax_percentile
        )
        
        # Build custom data - for 2D heatmaps need to reshape to match z_data
        custom_data = self._build_custom_data(ds, trace_config.custom_data_sources)
        if custom_data is not None and z_data.ndim == 2:
            # For 2D heatmaps, custom data needs to be tiled to match heatmap shape
            if len(custom_data.shape) == 2 and custom_data.shape[1] == 1:
                # Remove extra dimension first: (150, 1) -> (150,)
                custom_data = custom_data.squeeze()
            if custom_data.shape[0] == x_data.shape[0]:  # matches frequency dimension
                # Tile to (n_powers, n_freqs) like the original det2d
                custom_data = ArrayManipulator.tile_for_hover_data(custom_data, z_data.shape, axis=0)
        
        # Create heatmap trace
        
        fig.add_trace(
            go.Heatmap(
                x=x_data,
                y=y_data,
                z=z_data,
                customdata=custom_data,
                colorscale=trace_config.colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=False,  # Try original setting
                colorbar=dict(
                    x=1.0,  # placeholder like original
                    y=0.5,
                    len=1.0,
                    thickness=trace_config.colorbar.thickness if trace_config.colorbar else 14,
                    xanchor="left",
                    yanchor="middle",
                    ticks="outside",
                    ticklabelposition="outside",
                    title=trace_config.colorbar.title if trace_config.colorbar else "|IQ|",
                ),
                hovertemplate=trace_config.hover_template,
                name=trace_config.name
            ),
            row=row,
            col=col
        )
    
    def _add_heatmap_trace_multi_qubit(
        self,
        fig: go.Figure,
        ds_full: xr.Dataset,
        trace_config: HeatmapTraceConfig,
        row: int,
        col: int,
        qubit_id: str,
        qubit_index: int
    ) -> None:
        """Add heatmap trace for multi-qubit experiments using centralized data extraction."""
        
        # Determine coordinate names
        freq_coord = CoordinateNames.FULL_FREQ if CoordinateNames.FULL_FREQ in ds_full else CoordinateNames.FREQ_FULL
        power_coord = CoordinateNames.POWER if CoordinateNames.POWER in ds_full.coords else CoordinateNames.POWER_DBM
        
        # Use base class method to prepare data
        try:
            freq_vals, power_vals, z_mat, detuning_mhz = self._prepare_multi_qubit_heatmap_data(
                ds_full, qubit_id, freq_coord, power_coord, CoordinateNames.IQ_ABS_NORM
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not prepare heatmap data for {qubit_id}: {e}")
            return
        
        # Build custom data for hover
        n_powers, n_freqs = z_mat.shape
        det2d = ArrayManipulator.tile_for_hover_data(detuning_mhz, (n_powers, n_freqs), axis=0)
        
        # Calculate z-limits using base class method
        zmin, zmax = self._calculate_robust_zlimits(z_mat, 2, 98)
        
        # Create heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=freq_vals,
                y=power_vals,
                customdata=det2d,
                colorscale=ColorScales.VIRIDIS,
                zmin=zmin,
                zmax=zmax,
                showscale=False,
                colorbar=dict(
                    x=1.0,
                    y=0.5,
                    len=1.0,
                    thickness=14,
                    xanchor="left",
                    yanchor="middle", 
                    ticks="outside",
                    ticklabelposition="outside",
                    title="|IQ|",
                ),
                hovertemplate=(
                    "Freq [GHz]: %{x:.3f}<br>"
                    "Power [dBm]: %{y:.2f}<br>"
                    "Detuning [MHz]: %{customdata:.2f}<br>"
                    "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
                ),
                name=f"Qubit {qubit_id}",
            ),
            row=row,
            col=col
        )
        
        # Set axis ranges
        fig.update_xaxes(
            range=[freq_vals.min(), freq_vals.max()],
            row=row, col=col
        )
        fig.update_yaxes(
            range=[power_vals.min(), power_vals.max()],
            row=row, col=col
        )
    
    def _add_heatmap_trace_power_rabi(
        self,
        fig: go.Figure,
        ds: xr.Dataset,
        trace_config: HeatmapTraceConfig,
        row: int,
        col: int
    ) -> None:
        """Add heatmap trace for power rabi using centralized data extraction."""
        
        # Use base class method to prepare data
        try:
            data_dict = self._prepare_power_rabi_heatmap_data(ds)
        except (KeyError, ValueError, DataSourceError) as e:
            logger.warning(f"Could not prepare power rabi heatmap data: {e}")
            return
        
        # Extract prepared data
        amp_mv = data_dict['amp_mv']
        amp_prefactor = data_dict['amp_prefactor']
        nb_pulses = data_dict['nb_pulses']
        z_plot = data_dict['z_data']
        
        # Customdata for hover: each column is the prefactor value
        customdata = ArrayManipulator.tile_for_hover_data(
            amp_prefactor, (len(nb_pulses), len(amp_prefactor)), axis=1
        )
        
        hm_trace = go.Heatmap(
            z=z_plot,
            x=amp_mv,
            y=nb_pulses,
            colorscale="Viridis",
            showscale=False,
            colorbar=dict(
                title="|IQ|",
                titleside="right",
                ticks="outside",
            ),
            customdata=customdata,
            hovertemplate="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata:.3f}<br>Pulses: %{y}<br>Value: %{z:.3f}<extra></extra>",
        )
        
        fig.add_trace(hm_trace, row=row, col=col)

    def _add_heatmap_trace_flux_spectroscopy(
        self,
        fig: go.Figure,
        ds_full: xr.Dataset,
        trace_config: HeatmapTraceConfig,
        row: int,
        col: int,
        qubit_id: str,
        qubit_index: int
    ) -> None:
        """Add heatmap trace for flux spectroscopy using centralized data extraction."""
        
        # Use base class method to prepare data
        try:
            data_dict = self._prepare_flux_spectroscopy_heatmap_data(ds_full, qubit_id)
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not prepare flux spectroscopy data for {qubit_id}: {e}")
            return
            
        # Extract prepared data
        freq_vals = data_dict['freq_ghz']
        flux_vals = data_dict['flux_v']
        current_vals = data_dict['current_a']
        detuning_mhz = data_dict['detuning_mhz']
        z_mat = data_dict['z_data']
        
        # Build custom data for hover
        n_freqs, n_flux = z_mat.shape
        det2d = ArrayManipulator.tile_for_hover_data(detuning_mhz, (n_freqs, n_flux), axis=1)
        current2d = ArrayManipulator.tile_for_hover_data(current_vals, (n_freqs, n_flux), axis=0)
        customdata = ArrayManipulator.stack_custom_data([det2d, current2d], axis=-1)
        
        # Calculate z-limits
        finite_values = DataValidator.get_finite_values(z_mat)
        if len(finite_values) > 0:
            zmin = float(np.min(finite_values))
            zmax = float(np.max(finite_values))
        else:
            zmin, zmax = 0.0, 1.0
        
        # Create heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=flux_vals,
                y=freq_vals,
                customdata=customdata,
                colorscale=ColorScales.VIRIDIS,
                zmin=zmin,
                zmax=zmax,
                showscale=False,
                colorbar=dict(
                    x=1.0,
                    y=0.5,
                    len=1.0,
                    thickness=14,
                    xanchor="left",
                    yanchor="middle", 
                    ticks="outside",
                    ticklabelposition="outside",
                    title="|IQ|",
                ),
                hovertemplate=(
                    "Flux [V]: %{x:.3f}<br>"
                    "Current [A]: %{customdata[1]:.6f}<br>"
                    "Freq [GHz]: %{y:.3f}<br>"
                    "Detuning [MHz]: %{customdata[0]:.2f}<br>"
                    "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
                ),
                name=f"Qubit {qubit_id}",
            ),
            row=row,
            col=col
        )
        
        # Set axis ranges
        fig.update_xaxes(
            range=[flux_vals.min(), flux_vals.max()],
            row=row, col=col
        )
        fig.update_yaxes(
            range=[freq_vals.min(), freq_vals.max()],
            row=row, col=col
        )

    def _add_overlays_multi_qubit(
        self,
        fig: go.Figure,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlays: List,
        row: int,
        col: int
    ) -> None:
        """Add overlays for multi-qubit heatmap using centralized methods."""
        
        # Use base class validation method
        if not self._validate_overlay_fit(ds_fit, qubit_id):
            return
            
        # Extract parameters using base class method
        parameter_map = {
            'res_freq_ghz': 'res_freq',
            'optimal_power': 'optimal_power'
        }
        unit_conversions = {
            'res_freq_ghz': PlotConstants.GHZ_PER_HZ  # Convert Hz to GHz
        }
        
        params = self._extract_overlay_parameters(ds_fit, parameter_map, unit_conversions)
        
        if 'res_freq_ghz' not in params or 'optimal_power' not in params:
            logger.warning(f"Missing required parameters for overlays in qubit {qubit_id}")
            return
            
        try:
            # Get power range for vertical line (from -50 to -25 dBm)
            power_min, power_max = -50.0, -25.0
            
            # Add red dashed vertical line at resonator frequency
            fig.add_trace(
                go.Scatter(
                    x=[params['res_freq_ghz'], params['res_freq_ghz']],
                    y=[power_min, power_max],
                    mode="lines",
                    line=dict(
                        color="#FF0000",  # Red
                        width=2,
                        dash="dash"
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            
            # Get frequency range using centralized method
            freq_range = self._get_frequency_range(ds_fit, qubit_id)
            if freq_range:
                freq_min, freq_max = freq_range
                
                # Add magenta horizontal line at optimal power
                fig.add_trace(
                    go.Scatter(
                        x=[freq_min, freq_max],
                        y=[params['optimal_power'], params['optimal_power']],
                        mode="lines",
                        line=dict(
                            color="#FF00FF",  # Magenta
                            width=2
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Could not add overlays for {qubit_id}: {e}")
            return
    
    def _add_overlays_flux_spectroscopy(
        self,
        fig: go.Figure,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlays: List,
        row: int,
        col: int,
        ds_raw: xr.Dataset = None
    ) -> None:
        """Add overlays for flux spectroscopy using original logic."""
        
        # Use base class validation method
        if not self._validate_overlay_fit(ds_fit, qubit_id):
            return
            
        # Extract parameters using base class method
        parameter_map = {
            'idle_offset': 'idle_offset',
            'flux_min': 'flux_min',
            'sweet_spot_freq_ghz': 'sweet_spot_frequency'
        }
        unit_conversions = {
            'sweet_spot_freq_ghz': PlotConstants.GHZ_PER_HZ  # Convert Hz to GHz
        }
        
        params = self._extract_overlay_parameters(ds_fit, parameter_map, unit_conversions)
        
        if not all(key in params for key in ['idle_offset', 'flux_min', 'sweet_spot_freq_ghz']):
            logger.warning(f"Missing required parameters for flux spectroscopy overlays in qubit {qubit_id}")
            return
            
        # Get frequency range using base class method
        freq_range = self._get_frequency_range(ds_raw, qubit_id)
        if freq_range is None:
            logger.warning(f"Could not get frequency range for flux spectroscopy overlays")
            return
            
        freq_min, freq_max = freq_range
        
        try:
            # Magenta "×" at (idle_offset, sweet_spot_frequency) - exact copy of original 
            fig.add_trace(
                go.Scatter(
                    x=[params['idle_offset']],
                    y=[params['sweet_spot_freq_ghz']],
                    mode="markers",
                    marker=dict(
                        symbol="x",         # PLOTLY_SWEET_SPOT_MARKER_SYMBOL from original
                        color="#FF00FF",    # SWEET_SPOT_COLOR from original
                        size=15,           # PLOTLY_SWEET_SPOT_MARKER_SIZE from original
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=row,
                col=col,
            )
            
            # Red dashed vertical line at idle_offset - exact copy of original code
            fig.add_trace(
                go.Scatter(
                    x=[params['idle_offset'], params['idle_offset']],
                    y=[freq_min, freq_max],
                    mode="lines",
                    line=dict(
                        color="#FF0000",  # Red (IDLE_OFFSET_COLOR from original)
                        width=2.5,      # PLOTLY_FIT_LINE_WIDTH from original
                        dash="dash"     # PLOTLY_FIT_LINE_STYLE from original
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            
            # Purple dashed vertical line at flux_min - exact copy of original code
            fig.add_trace(
                go.Scatter(
                    x=[params['flux_min'], params['flux_min']],
                    y=[freq_min, freq_max],
                    mode="lines",
                    line=dict(
                        color="#800080",  # Purple (MIN_OFFSET_COLOR from original)
                        width=2.5,       # PLOTLY_FIT_LINE_WIDTH from original
                        dash="dash"      # PLOTLY_FIT_LINE_STYLE from original
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
                
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Could not add flux spectroscopy overlays for {qubit_id}: {e}")
            return

    def _add_overlays_power_rabi(
        self,
        fig: go.Figure,
        ds_fit: xr.Dataset,
        qubit_id: str,
        overlays: List,
        row: int,
        col: int,
        ds_raw: xr.Dataset = None,
    ) -> None:
        """Add overlays for power rabi experiment using logic from the original plotting script."""

        # Use base class validation method
        if not self._validate_overlay_fit(ds_fit, qubit_id):
            return

        # Extract parameters using base class method
        parameter_map = {
            'opt_amp_prefactor': 'opt_amp_prefactor'
        }
        
        params = self._extract_overlay_parameters(ds_fit, parameter_map)
        
        if 'opt_amp_prefactor' not in params:
            logger.warning(f"Missing opt_amp_prefactor for power rabi overlays in qubit {qubit_id}")
            return

        if ds_raw is None:
            return

        try:
            ds_qubit_raw = DataExtractor.extract_qubit_data(ds_raw, qubit_id)
            amp_prefactor_raw = ds_qubit_raw[CoordinateNames.AMP_PREFACTOR].values
            amp_mv_raw = ds_qubit_raw[CoordinateNames.FULL_AMP].values * PlotConstants.MV_PER_V

            # Find the amplitude (in mV) corresponding to the optimal prefactor
            try:
                opt_amp_mv = (
                    float(
                        ds_qubit_raw[CoordinateNames.FULL_AMP]
                        .sel({CoordinateNames.AMP_PREFACTOR: params['opt_amp_prefactor']}, method="nearest")
                        .values
                    )
                    * PlotConstants.MV_PER_V
                )
            except (KeyError, ValueError):
                opt_amp_mv = float(
                    amp_mv_raw[np.argmin(np.abs(amp_prefactor_raw - params['opt_amp_prefactor']))]
                )

            # Get pulse number range
            nb_of_pulses = ds_raw[CoordinateNames.NB_OF_PULSES].values
            pulse_min, pulse_max = nb_of_pulses.min(), nb_of_pulses.max()

            # Add red dashed vertical line at optimal amplitude
            fig.add_trace(
                go.Scatter(
                    x=[opt_amp_mv, opt_amp_mv],
                    y=[pulse_min, pulse_max],
                    mode="lines",
                    line=dict(color="#FF0000", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Could not add power rabi overlays for {qubit_id}: {e}")
            
    def _add_generic_trace(
        self,
        fig: go.Figure,
        ds: xr.Dataset,
        trace_config: TraceConfig,
        row: int,
        col: int,
        subplot_index: int
    ) -> None:
        """Add generic trace to figure."""
        
        custom_data = self._build_custom_data(ds, trace_config.custom_data_sources)
        
        trace_props = {
            "name": trace_config.name,
            "hovertemplate": trace_config.hover_template,
            "customdata": custom_data,
            "legendgroup": trace_config.name,
            "showlegend": subplot_index == 0
        }
        
        if trace_config.plot_type == "heatmap":
            # Handle as heatmap if it's a HeatmapTraceConfig
            if isinstance(trace_config, HeatmapTraceConfig):
                self._add_heatmap_trace(fig, ds, trace_config, row, col)
        else:
            # Handle as scatter/line
            trace = go.Scatter(
                x=ds[trace_config.x_source].values,
                y=ds[trace_config.y_source].values,
                mode=trace_config.mode,
                line=trace_config.style,
                **trace_props,
            )
            fig.add_trace(trace, row=row, col=col)
    
    def _add_overlays(
        self,
        fig: go.Figure,
        ds_qubit_fit: xr.Dataset,
        qubit_id: str,
        overlays: List[Union[LineOverlayConfig, MarkerOverlayConfig]],
        row: int,
        col: int
    ) -> None:
        """Add overlay traces to figure."""
        
        for overlay in overlays:
            if not self.overlay_renderer.should_render_overlay(ds_qubit_fit, overlay, qubit_id):
                continue
            
            if overlay.type == "line":
                self._add_line_overlay(fig, ds_qubit_fit, qubit_id, overlay, row, col)
            elif overlay.type == "marker":
                self._add_marker_overlay(fig, ds_qubit_fit, qubit_id, overlay, row, col)
    
    def _add_line_overlay(
        self,
        fig: go.Figure,
        ds_qubit_fit: xr.Dataset,
        qubit_id: str,
        overlay: LineOverlayConfig,
        row: int,
        col: int
    ) -> None:
        """Add line overlay to figure."""
        
        position = self.overlay_renderer.get_overlay_position(ds_qubit_fit, overlay.position_source, qubit_id)
        
        if position is None:
            return
        
        if overlay.orientation == "vertical":
            trace = go.Scatter(
                x=[position, position],
                y=[0, 1],  # Will be auto-scaled by Plotly
                mode="lines",
                line=overlay.line_style,
                showlegend=False,
                hoverinfo="skip"
            )
        elif overlay.orientation == "horizontal":
            trace = go.Scatter(
                x=[0, 1],  # Will be auto-scaled by Plotly
                y=[position, position],
                mode="lines",
                line=overlay.line_style,
                showlegend=False,
                hoverinfo="skip"
            )
        else:
            return
        
        fig.add_trace(trace, row=row, col=col)
    
    def _add_marker_overlay(
        self,
        fig: go.Figure,
        ds_qubit_fit: xr.Dataset,
        qubit_id: str,
        overlay: MarkerOverlayConfig,
        row: int,
        col: int
    ) -> None:
        """Add marker overlay to figure."""
        
        x_pos = self.overlay_renderer.get_overlay_position(ds_qubit_fit, overlay.x_source, qubit_id)
        y_pos = self.overlay_renderer.get_overlay_position(ds_qubit_fit, overlay.y_source, qubit_id)
        
        if x_pos is None or y_pos is None:
            return
        
        fig.add_trace(
            go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode="markers",
                marker=overlay.marker_style,
                showlegend=False,
                hoverinfo="skip"
            ),
            row=row,
            col=col
        )
    
    def _add_dual_axis(
        self,
        fig: go.Figure,
        ds: xr.Dataset,
        dual_config: DualAxisConfig,
        row: int,
        col: int,
        n_cols: int
    ) -> None:
        """Add dual axis (top x-axis) to subplot."""
        
        if dual_config.top_axis_source not in ds:
            return
        
        # Special handling for power rabi dual axis (mV vs prefactor)
        if self.experiment_detector.detect_experiment_type(ds) == ExperimentTypes.POWER_RABI and dual_config.top_axis_source == CoordinateNames.AMP_PREFACTOR:
            self._add_power_rabi_dual_axis(fig, ds, dual_config, row, col, n_cols)
            return
        
        top_axis_data = ds[dual_config.top_axis_source].values
        subplot_index = (row - 1) * n_cols + col
        
        # Create overlay axis
        top_xaxis_name = f"xaxis{subplot_index + dual_config.overlay_offset}"
        main_xaxis_name = f"x{subplot_index}"
        
        if len(top_axis_data) > 1:
            tick_text = [dual_config.top_axis_format.format(v) for v in top_axis_data]
            
            axis_config = dict(
                overlaying=main_xaxis_name,
                side="top",
                title=dual_config.top_axis_title,
                showgrid=False,
                tickmode="array",
                tickvals=list(top_axis_data),
                ticktext=tick_text,
                range=[float(np.min(top_axis_data)), float(np.max(top_axis_data))]
            )
        else:
            axis_config = dict(
                overlaying=main_xaxis_name,
                side="top",
                title=dual_config.top_axis_title,
                showgrid=False,
                tickmode="auto"
            )
        
        # Update layout with new axis
        fig.layout[top_xaxis_name] = axis_config
    
    def _position_colorbars(
        self,
        fig: go.Figure,
        heatmap_info: List[tuple],
        n_cols: int,
        trace_configs: List[HeatmapTraceConfig]
    ) -> None:
        """Position colorbars for heatmap subplots."""
        
        if not trace_configs or not trace_configs[0].colorbar:
            return
        
        # Get the number of rows from heatmap info
        n_rows = max(row for _, row, _ in heatmap_info) if heatmap_info else 1
        
        # Use base class method to calculate positions
        subplot_indices = [(row - 1, col - 1) for _, row, col in heatmap_info]
        colorbar_positions = self._calculate_colorbar_positions(n_rows, n_cols, subplot_indices)
        
        # Apply positions to each heatmap
        for (hm_idx, _, _), position in zip(heatmap_info, colorbar_positions):
            if hm_idx >= len(fig.data):
                continue
            
            fig.data[hm_idx].colorbar.update(position)
    
    def _add_power_rabi_dual_axis(
        self,
        fig: go.Figure,
        ds: xr.Dataset,
        dual_config: DualAxisConfig,
        row: int,
        col: int,
        n_cols: int
    ) -> None:
        """Add power rabi specific dual axis that synchronizes mV and prefactor scales."""
        
        # Get amplitude data (exactly like original)
        amp_mV = ds[CoordinateNames.AMP_MV].values if CoordinateNames.AMP_MV in ds else ds[CoordinateNames.FULL_AMP].values * PlotConstants.MV_PER_V
        amp_prefactor = ds[CoordinateNames.AMP_PREFACTOR].values
        
        subplot_index = (row - 1) * n_cols + col
        main_xaxis_name = f"x{subplot_index}"
        top_xaxis_name = f"xaxis{subplot_index + 100}"  # Use fixed offset like original
        
        # Create synchronized dual axis (EXACT copy of original logic)
        axis_config = dict(
            overlaying=main_xaxis_name,
            side="top",
            title="Amplitude prefactor",
            showgrid=False,
            tickmode="array",
            tickvals=list(amp_mV),  # Use mV positions for tick placement
            ticktext=[f"{v:.2f}" for v in amp_prefactor],  # Show prefactor values
            range=[float(np.min(amp_mV)), float(np.max(amp_mV))]  # Match main axis range
        )
        
        fig.layout[top_xaxis_name] = axis_config
