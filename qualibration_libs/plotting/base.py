from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import plotly.graph_objects as go
import xarray as xr
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.plotting.configs import PlotStyling
from quam_builder.architecture.superconducting.qubit import AnyTransmon


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
    def _plot_plotly_subplot(self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]):
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
        grid.fig.tight_layout()
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