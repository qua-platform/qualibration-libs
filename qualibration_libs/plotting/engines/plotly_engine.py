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

from ..configs import (DualAxisConfig, FigureDimensions, HeatmapConfig,
                       HeatmapTraceConfig, LineOverlayConfig,
                       MarkerOverlayConfig, PlotConfig, SpectroscopyConfig,
                       SubplotSpacing, TraceConfig, get_standard_plotly_style)
from ..grids import QubitGrid
from .common import GridManager, OverlayRenderer, PlotlyEngineUtils
from .data_validators import DataValidator
from .base_engine import BaseRenderingEngine
from .experiment_detector import ExperimentDetector
from ..exceptions import (
    DataSourceError, QubitError, OverlayError, ConfigurationError
)

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
        
        grid = self.grid_manager.create_grid(ds_raw, qubits, create_figure=False)
        
        fig = make_subplots(
            rows=grid.n_rows,
            cols=grid.n_cols,
            subplot_titles=grid.get_subplot_titles(),
            horizontal_spacing=SubplotSpacing.STANDARD_HORIZONTAL,
            vertical_spacing=SubplotSpacing.STANDARD_VERTICAL,
        )
        
        # Add traces for each qubit
        for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
            row = grid_row + 1
            col = grid_col + 1
            qubit_id = list(name_dict.values())[0]
            
            ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
            ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None
            
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
        
        # Apply layout settings
        layout_settings = get_standard_plotly_style()
        layout_settings.update({
            "title_text": config.layout.title,
            "height": FigureDimensions.SUBPLOT_HEIGHT * grid.n_rows,
            "width": max(FigureDimensions.PLOTLY_MIN_WIDTH, FigureDimensions.SUBPLOT_WIDTH * grid.n_cols)
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
        
        grid = self.grid_manager.create_grid(ds_raw, qubits, create_figure=False)
        
        # Use config-specific spacing
        spacing = config.subplot_spacing
        fig = make_subplots(
            rows=grid.n_rows,
            cols=grid.n_cols,
            subplot_titles=grid.get_subplot_titles(),
            horizontal_spacing=spacing.get("horizontal", SubplotSpacing.HEATMAP_HORIZONTAL),
            vertical_spacing=spacing.get("vertical", SubplotSpacing.HEATMAP_VERTICAL),
        )
        
        heatmap_info = []
        
        # Add traces for each qubit
        for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
            row = grid_row + 1
            col = grid_col + 1
            qubit_id = list(name_dict.values())[0]
            
            # For multi-qubit datasets with 2D frequency arrays, handle differently
            ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
            ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None
            
            # Add heatmap traces  
            for trace_config in config.traces:
                if self.utils.check_trace_visibility(ds_qubit_raw, trace_config, qubit_id):
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
        
        # Apply layout settings
        layout_settings = get_standard_plotly_style()
        layout_settings.update({
            "title_text": config.layout.title,
            "showlegend": False,  # Heatmaps typically don't show legends
            "height": FigureDimensions.SUBPLOT_HEIGHT * grid.n_rows,
            "width": max(FigureDimensions.PLOTLY_MIN_WIDTH, FigureDimensions.SUBPLOT_WIDTH * grid.n_cols)
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
        
        grid = self.grid_manager.create_grid(ds_raw, qubits, create_figure=False)
        
        fig = make_subplots(
            rows=grid.n_rows,
            cols=grid.n_cols,
            subplot_titles=grid.get_subplot_titles(),
            horizontal_spacing=SubplotSpacing.STANDARD_HORIZONTAL,
            vertical_spacing=SubplotSpacing.STANDARD_VERTICAL,
        )
        
        # Add traces for each qubit
        for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
            row = grid_row + 1
            col = grid_col + 1
            qubit_id = list(name_dict.values())[0]
            
            ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
            ds_qubit_fit = ds_fit.sel(qubit=qubit_id) if ds_fit is not None else None
            
            # Add traces based on type
            for trace_config in config.traces + config.fit_traces:
                if not self.utils.check_trace_visibility(ds_qubit_raw, trace_config, qubit_id):
                    continue
                
                is_fit_trace = trace_config in config.fit_traces
                ds_source = ds_qubit_fit if is_fit_trace else ds_qubit_raw
                
                if ds_source is None or not self.utils.validate_trace_sources(ds_source, trace_config):
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
            if is_fit and ("outcome" not in ds.coords or ds.outcome != "successful"):
                continue

            if not self.utils.validate_trace_sources(ds, trace_config):
                continue
            
            # Build custom data
            custom_data = self.utils.build_custom_data(ds, trace_config.custom_data_sources)
            
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
        
        if not self.utils.validate_trace_sources(ds, trace_config):
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
            if np.all(np.isnan(z_data)):
                z_data = np.zeros_like(z_data)
        elif z_data.ndim == 1:
            z_data = z_data[np.newaxis, :]
        
        # Calculate robust z-limits
        zmin, zmax = self.utils.calculate_robust_zlimits(
            z_data, trace_config.zmin_percentile, trace_config.zmax_percentile
        )
        
        # Build custom data - for 2D heatmaps need to reshape to match z_data
        custom_data = self.utils.build_custom_data(ds, trace_config.custom_data_sources)
        if custom_data is not None and z_data.ndim == 2:
            # For 2D heatmaps, custom data needs to be tiled to match heatmap shape
            if len(custom_data.shape) == 2 and custom_data.shape[1] == 1:
                # Remove extra dimension first: (150, 1) -> (150,)
                custom_data = custom_data.squeeze()
            if custom_data.shape[0] == x_data.shape[0]:  # matches frequency dimension
                # Tile to (n_powers, n_freqs) like the original det2d
                custom_data = np.tile(custom_data[np.newaxis, :], (z_data.shape[0], 1))
        
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
        """Add heatmap trace using EXACT original logic."""
        
        # EXACTLY REPLICATE the original plotting.py logic
        # 1) Transpose ds_raw so that its dims become (qubit, detuning, power)
        ds2 = ds_full.transpose("qubit", "detuning", "power")
        
        # 2) Pull out the raw arrays exactly like original
        if "full_freq" in ds2:
            freq_array = ds2["full_freq"].values  # (n_qubits, n_freqs)
        elif "freq_full" in ds2:
            freq_array = ds2["freq_full"].values
        else:
            return
            
        if "IQ_abs_norm" not in ds2:
            return
        IQ_array = ds2["IQ_abs_norm"].values  # (n_qubits, n_freqs, n_powers)
        
        # Pick the power axis:
        if "power" in ds2.coords:
            power_array = ds2["power"].values  # (n_powers,) in dBm
        elif "power_dbm" in ds2.coords:
            power_array = ds2["power_dbm"].values
        else:
            return
            
        # Detuning axis (Hz):
        if "detuning" not in ds2.coords:
            return
        detuning_array = ds2["detuning"].values  # (n_freqs,) in Hz
        
        n_qubits, n_freqs, n_powers = IQ_array.shape
        
        # 3) Find qubit index in dataset (NOT grid iteration index)
        q_labels = list(ds2.qubit.values)  # e.g. ["q1", "q2", "q3", "q4"] 
        try:
            q_idx = q_labels.index(qubit_id)  # CRITICAL: Use dataset array index
        except ValueError:
            return
            
        # 4) Extract data for this specific qubit using DATASET INDEX
        GHZ_PER_HZ = 1e-9
        MHZ_PER_HZ = 1e-6
        
        freq_vals = freq_array[q_idx] * GHZ_PER_HZ  # (n_freqs,) in GHz
        power_vals = power_array  # (n_powers,) in dBm
        
        
        # 5) Build 2D z‐matrix for heatmap (n_powers, n_freqs):
        z_mat = IQ_array[q_idx].T  # Use DATASET index, not grid index
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape[0] != n_powers:
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)
            
        # 6) Build custom data for hover (exactly like original)
        detuning_MHz = (detuning_array * MHZ_PER_HZ).astype(float)  # (n_freqs,) in MHz
        det2d = np.tile(detuning_MHz[np.newaxis, :], (n_powers, 1))  # (n_powers, n_freqs)
        
        # 7) Calculate z-limits (use robust percentiles like original)
        zmin = float(np.nanpercentile(z_mat.flatten(), 2))
        zmax = float(np.nanpercentile(z_mat.flatten(), 98))
        
        
        # 8) Create heatmap trace EXACTLY like original
        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=freq_vals,
                y=power_vals,
                customdata=det2d,
                colorscale="Viridis",
                zmin=zmin,
                zmax=zmax,
                showscale=False,  # Original uses False, positions later
                colorbar=dict(
                    x=1.0,  # placeholder (moved later like original)
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
        
        # CRITICAL: Set axis ranges to match actual data (fix auto-scaling issue)
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
        """Add heatmap trace for power rabi using EXACT original logic."""
        
        # EXACTLY REPLICATE the original plotting.py logic for power rabi 2D
        if hasattr(ds, "I"):
            data = "I"
        elif hasattr(ds, "state"):
            data = "state"
        else:
            return
        
        # Get the data arrays
        amp_mV = ds['full_amp'].values * 1e3  # MV_PER_V = 1e3
        amp_prefactor = ds['amp_prefactor'].values
        nb_of_pulses = ds['nb_of_pulses'].values
        z_data = ds[data].values
        
        # Ensure z_data shape is (nb_of_pulses, amp_mV)
        if z_data.shape[0] == len(nb_of_pulses) and z_data.shape[1] == len(amp_mV):
            z_plot = z_data
        elif z_data.shape[1] == len(nb_of_pulses) and z_data.shape[0] == len(amp_mV):
            z_plot = z_data.T
        else:
            z_plot = z_data
        
        # Customdata for hover: each column is the prefactor value
        customdata = np.tile(amp_prefactor, (len(nb_of_pulses), 1))
        
        hm_trace = go.Heatmap(
            z=z_plot,
            x=amp_mV,
            y=nb_of_pulses,
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
        """Add heatmap trace for flux spectroscopy using EXACT original logic."""
        
        # EXACTLY REPLICATE the original 02c plotting.py logic
        # 1) Transpose ds_raw so that its dims become (qubit, detuning, flux_bias)
        ds2 = ds_full.transpose("qubit", "detuning", "flux_bias")
        
        # 2) Pull out the raw arrays exactly like original
        if "full_freq" in ds2:
            freq_array = ds2["full_freq"].values  # (n_qubits, n_freqs)
        elif "freq_full" in ds2:
            freq_array = ds2["freq_full"].values
        else:
            return
            
        if "IQ_abs" not in ds2:
            return
        IQ_array = ds2["IQ_abs"].values  # (n_qubits, n_freqs, n_flux)
        
        # Pick the flux_bias axis:
        if "flux_bias" not in ds2.coords:
            return
        flux_array = ds2["flux_bias"].values  # (n_flux,) in V
        
        # Attenuated current axis:
        if "attenuated_current" not in ds2.coords:
            return
        current_array = ds2["attenuated_current"].values  # (n_flux,) in A
            
        # Detuning axis (Hz):
        if "detuning" not in ds2.coords:
            return
        detuning_array = ds2["detuning"].values  # (n_freqs,) in Hz
        
        n_qubits, n_freqs, n_flux = IQ_array.shape
        
        # 3) Find qubit index in dataset (NOT grid iteration index)
        q_labels = list(ds2.qubit.values)  # e.g. ["qC1", "qC2", "qC3"] 
        try:
            q_idx = q_labels.index(qubit_id)  # CRITICAL: Use dataset array index
        except ValueError:
            return
            
        # 4) Extract data for this specific qubit using DATASET INDEX
        GHZ_PER_HZ = 1e-9
        MHZ_PER_HZ = 1e-6
        
        freq_vals = freq_array[q_idx] * GHZ_PER_HZ  # (n_freqs,) in GHz
        flux_vals = flux_array  # (n_flux,) in V
        current_vals = current_array  # (n_flux,) in A
        
        # 5) Build 2D z‐matrix for heatmap (n_freqs, n_flux)
        # Note: 02c uses different orientation than 02b
        z_mat = IQ_array[q_idx]  # Use DATASET index, shape (n_freqs, n_flux)
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape != (n_freqs, n_flux):
            z_mat = z_mat.T  # fallback transpose if needed
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)
            
        # 6) Build custom data for hover (exactly like original)
        detuning_MHz = (detuning_array * MHZ_PER_HZ).astype(float)  # (n_freqs,) in MHz
        det2d = np.tile(detuning_MHz[:, None], (1, n_flux))  # shape (n_freqs, n_flux)
        
        # Build 2D current array for hover
        current2d = np.tile(current_array[np.newaxis, :], (n_freqs, 1))  # shape (n_freqs, n_flux)
        
        # Stack them for custom data
        customdata = np.stack([det2d, current2d], axis=-1)  # shape (n_freqs, n_flux, 2)
        
        # 7) Calculate z-limits (use robust percentiles like original)
        zmin = float(np.nanmin(z_mat))
        zmax = float(np.nanmax(z_mat))
        
        # 8) Create heatmap trace EXACTLY like original
        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=flux_vals,  # flux on x-axis 
                y=freq_vals,  # frequency on y-axis
                customdata=customdata,
                colorscale="Viridis",
                zmin=zmin,
                zmax=zmax,
                showscale=False,  # Original uses False, positions later
                colorbar=dict(
                    x=1.0,  # placeholder (moved later like original)
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
        
        # CRITICAL: Set axis ranges to match actual data (fix auto-scaling issue)
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
        """Add overlays for multi-qubit heatmap using original logic."""
        
        # Get fit data for this qubit like original code
        try:
            fit_ds = ds_fit.sel(qubit=qubit_id)
        except (KeyError, ValueError):
            return
            
        # Check if fit was successful (like original: outcome == "successful")
        if "outcome" not in fit_ds.coords or fit_ds.outcome != "successful":
            return
            
        # Extract values like original code
        try:
            # Red dashed vertical line at resonator frequency
            res_freq_hz = float(fit_ds.res_freq.values)
            res_freq_ghz = res_freq_hz * 1e-9  # Convert Hz to GHz
            
            # Get power range for vertical line (from -50 to -25 dBm)
            power_min, power_max = -50.0, -25.0
            
            # Add red dashed vertical line  
            fig.add_trace(
                go.Scatter(
                    x=[res_freq_ghz, res_freq_ghz],
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
            
            # Magenta horizontal line at optimal power
            optimal_power = float(fit_ds.optimal_power.values)
            
            # Get current axis data for frequency range
            ds2 = ds_fit.transpose("qubit", "detuning", "power")
            freq_coord_name = "full_freq" if "full_freq" in ds2 else "freq_full"
            if freq_coord_name in ds2:
                freq_array = ds2[freq_coord_name].values
                q_labels = list(ds2.qubit.values)
                q_idx = q_labels.index(qubit_id)
                freq_vals = freq_array[q_idx] * 1e-9  # Convert to GHz
                freq_min, freq_max = freq_vals.min(), freq_vals.max()
                
                # Add magenta horizontal line
                fig.add_trace(
                    go.Scatter(
                        x=[freq_min, freq_max],
                        y=[optimal_power, optimal_power],
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
        
        # Get fit data for this qubit like original code
        try:
            fit_ds = ds_fit.sel(qubit=qubit_id)
        except (KeyError, ValueError):
            return
            
        # Check if fit was successful (like original: outcome == "successful")
        if "outcome" not in fit_ds.coords or fit_ds.outcome != "successful":
            return
            
        # Extract values like original code - following exact same pattern as original plotting.py
        try:
            # The fit dataset structure uses fit_results just like the original
            if hasattr(fit_ds, 'fit_results'):
                # Extract from fit_results like original code
                idle_offset = float(fit_ds.fit_results.idle_offset.values)
                flux_min = float(fit_ds.fit_results.flux_min.values)
                sweet_spot_freq = float(fit_ds.fit_results.sweet_spot_frequency.values) * 1e-9  # Convert Hz to GHz
            else:
                # Fallback: direct access (for different fit dataset structure)
                idle_offset = float(fit_ds.idle_offset.values) * 1e-3  # Convert mV to V
                sweet_spot_freq = float(fit_ds.sweet_spot_frequency.values) * 1e-9  # Convert Hz to GHz
                flux_min = float(fit_ds.flux_min.values) * 1e-3  # Convert mV to V
            
            # Get actual frequency and flux ranges from the raw dataset, following original pattern
            # Extract frequency array and compute freq_vals like original code does from RAW dataset
            # We need to use ds_raw because ds_fit doesn't have the detuning dimension
            if ds_raw is None:
                # If no raw dataset provided, we can't get the frequency range
                return
                
            ds_raw_transposed = ds_raw.transpose("qubit", "detuning", "flux_bias")
            freq_coord_name = "full_freq" if "full_freq" in ds_raw_transposed else "freq_full"
            if freq_coord_name not in ds_raw_transposed:
                return
            
            freq_array = ds_raw_transposed[freq_coord_name].values  # (n_qubits, n_freqs)
            q_labels = list(ds_raw_transposed.qubit.values)
            q_idx = q_labels.index(qubit_id)
            freq_vals = freq_array[q_idx] * 1e-9  # Convert Hz to GHz, like original
            
            flux_vals = ds_raw_transposed["flux_bias"].values  # Get actual flux range
            
            # Magenta "×" at (idle_offset, sweet_spot_frequency) - exact copy of original 
            fig.add_trace(
                go.Scatter(
                    x=[idle_offset],
                    y=[sweet_spot_freq],
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
                    x=[idle_offset, idle_offset],
                    y=[freq_vals.min(), freq_vals.max()],
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
                    x=[flux_min, flux_min],
                    y=[freq_vals.min(), freq_vals.max()],
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

        try:
            # Check if qubit is a dimension we can select by
            if 'qubit' in ds_fit.dims:
                ds_qubit_fit = ds_fit.sel(qubit=qubit_id)
            else:
                # Dataset is already per-qubit or qubit is just a coordinate
                ds_qubit_fit = ds_fit
        except (KeyError, ValueError):
            return

        if "outcome" not in ds_qubit_fit.coords:
            return
            
        if ds_qubit_fit.outcome != "successful":
            return

        try:
            opt_amp_prefactor = float(ds_qubit_fit.opt_amp_prefactor.values)

            if ds_raw is None:
                return

            ds_qubit_raw = ds_raw.sel(qubit=qubit_id)
            amp_prefactor_raw = ds_qubit_raw["amp_prefactor"].values
            amp_mv_raw = ds_qubit_raw["full_amp"].values * 1e3

            try:
                opt_amp_mv = (
                    float(
                        ds_qubit_raw["full_amp"]
                        .sel(amp_prefactor=opt_amp_prefactor, method="nearest")
                        .values
                    )
                    * 1e3
                )
            except (KeyError, ValueError):
                opt_amp_mv = float(
                    amp_mv_raw[np.argmin(np.abs(amp_prefactor_raw - opt_amp_prefactor))]
                )

            nb_of_pulses = ds_raw["nb_of_pulses"].values
            pulse_min, pulse_max = nb_of_pulses.min(), nb_of_pulses.max()

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
        
        custom_data = self.utils.build_custom_data(ds, trace_config.custom_data_sources)
        
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
        if self.experiment_detector.detect_experiment_type(ds) == "power_rabi" and dual_config.top_axis_source == "amp_prefactor":
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
        
        colorbar_config = trace_configs[0].colorbar
        
        for hm_idx, row, col in heatmap_info:
            if hm_idx >= len(fig.data):
                continue
            
            # Calculate colorbar position based on subplot location
            subplot_index = (row - 1) * n_cols + col
            
            # Basic positioning - can be enhanced based on layout
            x_offset = colorbar_config.x_offset
            
            fig.data[hm_idx].colorbar.update({
                "x": 1.0 + x_offset,
                "thickness": colorbar_config.thickness,
                "len": colorbar_config.height_ratio,
                "xanchor": "left",
                "ticks": colorbar_config.ticks,
                "ticklabelposition": colorbar_config.ticklabelposition
            })
    
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
        amp_mV = ds['amp_mV'].values if 'amp_mV' in ds else ds['full_amp'].values * 1e3
        amp_prefactor = ds['amp_prefactor'].values
        
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
