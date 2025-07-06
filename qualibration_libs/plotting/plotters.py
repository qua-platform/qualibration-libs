from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.plotting.configs import PlotStyling
from qualibration_libs.plotting.helpers import add_plotly_top_axis
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# Constants and styling
MV_PER_V = 1e3
u = unit(coerce_to_integer=True)
styling = PlotStyling()

class BasePlotter(ABC):
    """
    An abstract base class for creating standardized Matplotlib and Plotly figures.

    This class provides a framework for generating grid-based plots for multiple
    qubits. It handles the boilerplate of creating subplots and iterating over
    qubits, while delegating the specific plotting logic for each subplot to
    the concrete child classes.
    """

    def __init__(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        ds_fit: Optional[xr.Dataset] = None,
    ):
        """
        Initializes the BasePlotter.

        Parameters
        ----------
        ds_raw : xr.Dataset
            The raw dataset to plot.
        qubits : List[AnyTransmon]
            A list of qubits to plot.
        ds_fit : Optional[xr.Dataset], optional
            The dataset containing fit results, by default None.
        """
        self.ds_raw = ds_raw
        self.qubits = qubits
        self.ds_fit = ds_fit
        self.styling = PlotStyling()
        self.grid = QubitGrid(
            self.ds_raw, [q.grid_location for q in self.qubits], create_figure=False
        )

    def get_plot_title(self) -> str:
        """
        Returns the main title for the plot. Can be overridden by subclasses.
        """
        return "Qualibration Plot"

    def get_xaxis_title(self) -> Optional[str]:
        """Returns the primary x-axis title. Can be overridden by subclasses."""
        return None

    def get_yaxis_title(self) -> Optional[str]:
        """Returns the primary y-axis title. Can be overridden by subclasses."""
        return None

    def get_secondary_xaxis_title(self) -> Optional[str]:
        """Returns the secondary x-axis title for a twin axis. Can be overridden by subclasses."""
        return None

    def get_secondary_yaxis_title(self) -> Optional[str]:
        """Returns the secondary y-axis title for a twin axis. Can be overridden by subclasses."""
        return None

    def _get_make_subplots_kwargs(self) -> dict:
        """
        Returns keyword arguments for `plotly.subplots.make_subplots`.
        Can be overridden by subclasses to customize subplot creation.
        """
        return {"shared_xaxes": False, "shared_yaxes": False}

    def _get_final_layout_updates(self) -> dict:
        """
        Returns keyword arguments for the final `fig.update_layout` call.
        Can be overridden by subclasses to customize the final layout.
        """
        return {
            "height": self.styling.plotly_fig_height,
            "width": self.styling.plotly_fig_width,
            "showlegend": False,
        }

    @abstractmethod
    def _plot_matplotlib_subplot(self, ax, qubit_dict: dict, fit_data: Optional[xr.Dataset]):
        """
        Abstract method to plot data for a single qubit on a Matplotlib subplot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Matplotlib axis to plot on.
        qubit_dict : dict
            A dictionary identifying the qubit for the current subplot.
        fit_data : Optional[xr.Dataset]
            The fit data corresponding to the qubit.
        """
        pass

    def _after_plotly_plotting_loop(self, fig: go.Figure):
        """
        A hook for post-processing the Plotly figure after all subplots are drawn.
        Can be overridden by subclasses for tasks like repositioning colorbars.
        """
        pass

    @abstractmethod
    def _plot_plotly_subplot(
        self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]
    ):
        """
        Abstract method to plot data for a single qubit on a Plotly subplot.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure object.
        qubit_id : str
            The ID of the qubit for the current subplot.
        row : int
            The 1-based row index for the subplot.
        col : int
            The 1-based column index for the subplot.
        fit_data : Optional[xr.Dataset]
            The fit data corresponding to the qubit.
        """
        pass

    def create_matplotlib_plot(self) -> Figure:
        """
        Creates and returns a Matplotlib figure with all subplots.
        """
        # Create a new grid object specifically for Matplotlib.
        # This ensures a fresh figure is created every time, which is important
        # for interactive environments, and avoids modifying the QubitGrid class.
        grid = QubitGrid(
            self.ds_raw, [q.grid_location for q in self.qubits], create_figure=True
        )

        for ax, qubit_dict in grid_iter(grid):
            fit_sel = (
                self.ds_fit.sel(qubit=qubit_dict["qubit"])
                if self.ds_fit is not None
                else None
            )
            self._plot_matplotlib_subplot(ax, qubit_dict, fit_sel)

        grid.fig.suptitle(self.get_plot_title())
        grid.fig.set_size_inches(
            self.styling.matplotlib_fig_width, self.styling.matplotlib_fig_height
        )
        try:
            grid.fig.tight_layout()
        except ValueError:
            # Note: tight_layout() can fail with an error in some cases,
            # we can safely ignore it.
            pass
        return grid.fig

    def create_plotly_plot(self) -> go.Figure:
        """
        Creates and returns a Plotly figure with all subplots.
        """
        fig = make_subplots(
            rows=self.grid.n_rows,
            cols=self.grid.n_cols,
            subplot_titles=self.grid.get_subplot_titles(),
            **self._get_make_subplots_kwargs(),
        )

        for (grid_row, grid_col), name_dict in self.grid.plotly_grid_iter():
            row = grid_row + 1  # 1-based for Plotly
            col = grid_col + 1  # 1-based for Plotly
            qubit_id = list(name_dict.values())[0]
            fit_sel = self.ds_fit.sel(qubit=qubit_id) if self.ds_fit is not None else None
            self._plot_plotly_subplot(fig, qubit_id, row, col, fit_sel)

        self._after_plotly_plotting_loop(fig)

        fig.update_layout(
            title_text=self.get_plot_title(), **self._get_final_layout_updates()
        )
        return fig

    def plot(self) -> Tuple[go.Figure, Figure]:
        """
        Generates and returns both Plotly and Matplotlib figures.

        Returns
        -------
        Tuple[go.Figure, Figure]
            A tuple containing the Plotly figure and the Matplotlib figure.
        """
        plotly_fig = self.create_plotly_plot()
        matplotlib_fig = self.create_matplotlib_plot()
        return plotly_fig, matplotlib_fig


class HeatmapPlotter(BasePlotter):
    """
    A specialized plotter for 2D heatmap figures.

    It automates z-limit calculation for consistent color scaling and handles
    the repositioning of colorbars for each subplot in Plotly.
    """

    def __init__(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        ds_fit: Optional[xr.Dataset] = None,
    ):
        super().__init__(ds_raw, qubits, ds_fit)
        self.per_zmin: List[float] = []
        self.per_zmax: List[float] = []
        self._calculate_z_limits()

    def _get_z_matrix(self, qubit_id: str) -> np.ndarray:
        """
        Returns the 2D matrix for the z-axis of the heatmap.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _z_limit_calculator(self, z_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the z-limits (min, max) for a given z_matrix.
        Default implementation uses nanmin and nanmax. Can be overridden.
        """
        return float(np.nanmin(z_matrix)), float(np.nanmax(z_matrix))

    def _calculate_z_limits(self):
        """
        Precomputes z-limits for each qubit for consistent color scaling.
        """
        for qubit in self.qubits:
            z_mat = self._get_z_matrix(qubit.name)
            if z_mat.ndim == 1:
                z_mat = z_mat[np.newaxis, :]
            if np.all(np.isnan(z_mat)):
                z_mat = np.zeros_like(z_mat)
            zmin, zmax = self._z_limit_calculator(z_mat)
            self.per_zmin.append(zmin)
            self.per_zmax.append(zmax)

    def _get_make_subplots_kwargs(self) -> dict:
        return {
            "shared_xaxes": False,
            "shared_yaxes": False,
            "horizontal_spacing": self.styling.plotly_horizontal_spacing,
            "vertical_spacing": self.styling.plotly_vertical_spacing,
        }

    def _get_final_layout_updates(self) -> dict:
        return {
            "width": max(
                self.styling.plotly_min_width,
                self.styling.plotly_subplot_width * self.grid.n_cols,
            ),
            "height": self.styling.plotly_subplot_height * self.grid.n_rows,
            "margin": self.styling.plotly_margin,
            "showlegend": False,
        }

    def _after_plotly_plotting_loop(self, fig: go.Figure):
        """
        Repositions colorbars for each heatmap subplot.
        This logic assumes the first trace added to a subplot is the heatmap.
        """
        trace_counter = 0
        for (grid_row, grid_col), name_dict in self.grid.plotly_grid_iter():
            # This logic assumes the first trace for a subplot is always the heatmap
            if trace_counter >= len(fig.data):
                break

            hm_trace = fig.data[trace_counter]
            qubit_id = list(name_dict.values())[0]

            if not isinstance(hm_trace, go.Heatmap):
                # If the first trace isn't a heatmap, advance the counter by the
                # number of traces for this subplot. This is simplified logic
                # and might need adjustment if plot structures become more complex.
                fit_ds = self.ds_fit.sel(qubit=qubit_id) if self.ds_fit is not None else None
                trace_counter += self._get_num_traces_per_subplot(fit_ds)
                continue

            row, col = grid_row + 1, grid_col + 1
            axis_num = (row - 1) * self.grid.n_cols + col
            xaxis_key = f"xaxis{axis_num}" if axis_num > 1 else "xaxis"
            yaxis_key = f"yaxis{axis_num}" if axis_num > 1 else "yaxis"

            if xaxis_key in fig.layout and yaxis_key in fig.layout:
                x_dom = fig.layout[xaxis_key].domain
                y_dom = fig.layout[yaxis_key].domain

                x0_cb = x_dom[1] + self.styling.plotly_colorbar_x_offset
                bar_len = (y_dom[1] - y_dom[0]) * self.styling.plotly_colorbar_height_ratio
                bar_center_y = (y_dom[0] + y_dom[1]) / 2

                hm_trace.update(
                    colorbar=dict(
                        x=x0_cb,
                        y=bar_center_y,
                        len=bar_len,
                        thickness=self.styling.plotly_colorbar_thickness,
                        xanchor="left",
                        yanchor="middle",
                        ticks="outside",
                        ticklabelposition="outside",
                        title=self.styling.colorbar_config.title,
                    )
                )

            # Advance trace counter by the number of traces in this subplot
            fit_ds = self.ds_fit.sel(qubit=qubit_id) if self.ds_fit is not None else None
            trace_counter += self._get_num_traces_per_subplot(fit_ds)

    def _get_num_traces_per_subplot(self, fit_ds: Optional[xr.Dataset]) -> int:
        """
        Returns the number of traces expected per subplot. Default is 1.
        Subclasses should override this if they add more traces (e.g., for fits).
        """
        return 1


class LinePlotter(BasePlotter):
    """
    A specialized plotter for 1D line plots that require a secondary top x-axis.

    It automates the creation of a twin axis in Matplotlib and a secondary
    labeled axis in Plotly.
    """

    def get_raw_x_values(self, qubit_id: str) -> np.ndarray:
        """Returns the numpy array for the main x-axis from raw data."""
        raise NotImplementedError

    def get_secondary_x_values(self, qubit_id: str) -> np.ndarray:
        """Returns the numpy array for the secondary (top) x-axis labels."""
        raise NotImplementedError

    def _plot_matplotlib_subplot(
        self, ax: Axes, qubit_dict: dict, fit_data: Optional[xr.Dataset]
    ):
        """Creates a twin-y axis and delegates plotting to a subclass method."""
        ax2 = ax.twiny()
        self._plot_matplotlib_data(ax, ax2, qubit_dict, fit_data)

        if title := self.get_xaxis_title():
            ax.set_xlabel(title)
        if title := self.get_yaxis_title():
            ax.set_ylabel(title)
        if title := self.get_secondary_xaxis_title():
            ax2.set_xlabel(title)

    def _plot_plotly_subplot(
        self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]
    ):
        """Adds a secondary top axis and delegates plotting to a subclass method."""
        self._plot_plotly_data(fig, qubit_id, row, col, fit_data)

        add_plotly_top_axis(
            fig,
            row,
            col,
            self.grid.n_cols,
            self.get_raw_x_values(qubit_id),
            self.get_secondary_x_values(qubit_id),
            self.get_secondary_xaxis_title() or "",
        )
        if title := self.get_xaxis_title():
            fig.update_xaxes(title_text=title, row=row, col=col)
        if title := self.get_yaxis_title():
            fig.update_yaxes(title_text=title, row=row, col=col)

    def _plot_matplotlib_data(
        self,
        ax: Axes,
        ax2: Axes,
        qubit_dict: dict,
        fit_data: Optional[xr.Dataset],
    ):
        """
        Abstract method for subclasses to implement the actual Matplotlib plotting logic
        on the primary (ax) and secondary (ax2) axes.
        """
        raise NotImplementedError

    def _plot_plotly_data(
        self,
        fig: go.Figure,
        qubit_id: str,
        row: int,
        col: int,
        fit_data: Optional[xr.Dataset],
    ):
        """
        Abstract method for subclasses to implement the actual Plotly plotting logic.
        """
        raise NotImplementedError 

# Node specific plotters

class BaseResonatorSpectroscopyPlotter(LinePlotter):
    """A common base for amplitude and phase spectroscopy plotters."""

    def get_xaxis_title(self) -> str:
        return "RF frequency [GHz]"

    def get_secondary_xaxis_title(self) -> str:
        return "Detuning [MHz]"

    def get_raw_x_values(self, qubit_id: str) -> np.ndarray:
        return (
            self.ds_raw.assign_coords(full_freq_GHz=self.ds_raw.full_freq / u.GHz)
            .loc[{"qubit": qubit_id}]
            .full_freq_GHz.values
        )

    def get_secondary_x_values(self, qubit_id: str) -> np.ndarray:
        return (
            self.ds_raw.assign_coords(detuning_MHz=self.ds_raw.detuning / u.MHz)
            .loc[{"qubit": qubit_id}]
            .detuning_MHz.values
        )


class PowerRabi1DPlotter(LinePlotter):
    """Internal plotter for 1D Power Rabi experiments."""

    def __init__(self, ds_raw, qubits, ds_fit, data_key):
        self.data_key = data_key
        super().__init__(ds_raw, qubits, ds_fit)

    def get_plot_title(self) -> str:
        return "Power Rabi"

    def get_xaxis_title(self) -> str:
        return "Pulse amplitude [mV]"

    def get_yaxis_title(self) -> str:
        return "Rotated I quadrature [mV]" if self.data_key == "I" else "Qubit state"

    def get_secondary_xaxis_title(self) -> str:
        return "Amplitude prefactor"

    def get_raw_x_values(self, qubit_id: str) -> np.ndarray:
        return self.ds_raw.sel(qubit=qubit_id).full_amp.values * MV_PER_V

    def get_secondary_x_values(self, qubit_id: str) -> np.ndarray:
        return self.ds_raw.sel(qubit=qubit_id).amp_prefactor.values

    def _plot_matplotlib_data(
        self, ax: Axes, ax2: Axes, qubit_dict: dict, fit_data: Optional[xr.Dataset]
    ):
        ds_qubit = self.ds_raw.loc[qubit_dict].isel(nb_of_pulses=0)
        y_data = ds_qubit[self.data_key].values * MV_PER_V

        # Plot raw data on main and secondary axes
        ax.plot(
            ds_qubit.full_amp.values * MV_PER_V,
            y_data,
            alpha=styling.matplotlib_raw_data_alpha,
        )
        ax2.plot(
            ds_qubit.amp_prefactor.values,
            y_data,
            alpha=styling.matplotlib_raw_data_alpha,
        )

        # Plot fit
        if fit_data and getattr(fit_data.outcome, "values", None) == "successful":
            fitted_data = oscillation(
                fit_data.amp_prefactor.data,
                fit_data.fit.sel(fit_vals="a").data,
                fit_data.fit.sel(fit_vals="f").data,
                fit_data.fit.sel(fit_vals="phi").data,
                fit_data.fit.sel(fit_vals="offset").data,
            )
            ax.plot(
                fit_data.full_amp * MV_PER_V,
                MV_PER_V * fitted_data,
                linewidth=styling.matplotlib_fit_linewidth,
                color=styling.fit_color,
            )

    def _plot_plotly_data(
        self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]
    ):
        ds_qubit = self.ds_raw.sel(qubit=qubit_id).isel(nb_of_pulses=0)
        amp_mV = ds_qubit["full_amp"].values * MV_PER_V
        amp_prefactor = ds_qubit["amp_prefactor"].values
        y_data = ds_qubit[self.data_key].values * MV_PER_V
        y_err_da = ds_qubit.get(f"{self.data_key}_std")
        y_err = y_err_da.values * MV_PER_V if y_err_da is not None else None

        fig.add_trace(
            go.Scatter(
                x=amp_mV,
                y=y_data,
                error_y=dict(type="data", array=y_err, visible=True)
                if y_err is not None
                else None,
                name=f"Qubit {qubit_id} Raw",
                mode="lines+markers",
                line=dict(color=styling.raw_data_color),
                opacity=styling.matplotlib_raw_data_alpha,
                customdata=np.stack([amp_prefactor], axis=-1),
                hovertemplate="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata[0]:.3f}<br>%{y:.3f} mV<extra></extra>",
            ),
            row=row,
            col=col,
        )

        if fit_data and getattr(fit_data.outcome, "values", None) == "successful":
            fitted_data = oscillation(
                fit_data.amp_prefactor.data,
                fit_data.fit.sel(fit_vals="a").data,
                fit_data.fit.sel(fit_vals="f").data,
                fit_data.fit.sel(fit_vals="phi").data,
                fit_data.fit.sel(fit_vals="offset").data,
            )
            fig.add_trace(
                go.Scatter(
                    x=fit_data.full_amp.values * MV_PER_V,
                    y=MV_PER_V * fitted_data,
                    name=f"Qubit {qubit_id} - Fit",
                    line=dict(
                        color=styling.fit_color, width=styling.matplotlib_fit_linewidth
                    ),
                ),
                row=row,
                col=col,
            )


class PowerRabi2DPlotter(HeatmapPlotter):
    """Internal plotter for 2D Power Rabi experiments."""

    def __init__(self, ds_raw, qubits, ds_fit, data_key):
        self.data_key = data_key
        super().__init__(ds_raw, qubits, ds_fit)

    def get_plot_title(self) -> str:
        return "Power Rabi"

    def _get_z_matrix(self, qubit_id: str) -> np.ndarray:
        z_data = self.ds_raw[self.data_key].sel(qubit=qubit_id).values
        # Ensure shape is (y-axis, x-axis) -> (nb_of_pulses, amps)
        if z_data.shape[0] != len(self.ds_raw.nb_of_pulses.values):
            return z_data.T
        return z_data

    def _plot_matplotlib_subplot(self, ax: Axes, qubit_dict: dict, fit_data: Optional[xr.Dataset]):
        ds_qubit = self.ds_raw.loc[qubit_dict]
        (ds_qubit.assign_coords(amp_mV=ds_qubit.full_amp * MV_PER_V))[
            self.data_key
        ].plot(ax=ax, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True)

        if fit_data and getattr(fit_data.outcome, "values", None) == "successful":
            opt_amp_mV = self._get_optimal_amplitude_mv(ds_qubit, fit_data)
            ax.axvline(
                x=opt_amp_mV,
                color=styling.fit_color,
                linestyle="-",
                linewidth=styling.matplotlib_fit_linewidth,
            )
        ax.set_ylabel("Number of pulses")
        ax.set_xlabel("Pulse amplitude [mV]")

    def _plot_plotly_subplot(self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]):
        ds_qubit = self.ds_raw.sel(qubit=qubit_id)
        amp_mV = ds_qubit["full_amp"].values * MV_PER_V
        amp_prefactor = ds_qubit["amp_prefactor"].values
        nb_of_pulses = ds_qubit["nb_of_pulses"].values
        z_plot = self._get_z_matrix(qubit_id)
        customdata = np.tile(amp_prefactor, (len(nb_of_pulses), 1))

        fig.add_trace(
            go.Heatmap(
                z=z_plot,
                x=amp_mV,
                y=nb_of_pulses,
                colorscale=styling.heatmap_colorscale,
                showscale=False,
                customdata=customdata,
                hovertemplate="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata:.3f}<br>Pulses: %{y}<br>Value: %{z:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        if fit_data and getattr(fit_data.outcome, "values", None) == "successful":
            opt_amp_mV = self._get_optimal_amplitude_mv(ds_qubit, fit_data)
            fig.add_trace(
                go.Scatter(
                    x=[opt_amp_mV, opt_amp_mV],
                    y=[nb_of_pulses.min(), nb_of_pulses.max()],
                    mode="lines",
                    line=dict(
                        color=styling.fit_color,
                        width=styling.matplotlib_fit_linewidth,
                    ),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Pulse amplitude [mV]", row=row, col=col)
        fig.update_yaxes(title_text="Number of pulses", row=row, col=col)

    def _get_optimal_amplitude_mv(self, ds_qubit: xr.Dataset, fit: xr.Dataset) -> float:
        try:
            opt_amp_mv = (
                float(
                    ds_qubit["full_amp"]
                    .sel(amp_prefactor=fit.opt_amp_prefactor, method="nearest")
                    .values
                )
                * MV_PER_V
            )
        except (KeyError, ValueError) as e:
            logging.warning(
                f"Could not select optimal amplitude for qubit {ds_qubit.qubit.item()} using xarray, falling back to numpy. Error: {e}"
            )
            amp_prefactor = ds_qubit["amp_prefactor"].values
            amp_mV = ds_qubit["full_amp"].values * MV_PER_V
            opt_amp_mv = float(
                amp_mV[np.argmin(np.abs(amp_prefactor - fit.opt_amp_prefactor))]
            )
        return opt_amp_mv
