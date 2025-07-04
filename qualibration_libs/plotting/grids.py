import re
from typing import Iterator, List, Tuple, Union

import matplotlib
import xarray as xr
from matplotlib import pyplot as plt


def grid_pair_names(qubit_pairs) -> Tuple[List[str], List[str]]:
    """ "
    Runs over defined qubit pairs and returns a list of the grid_name attribute of each qubit, returns a list of the grid location and a list of the qubit pair names
    """
    return [
        f"{qp.qubit_control.grid_location}-{qp.qubit_target.grid_location}"
        for qp in qubit_pairs
    ], [qp.name for qp in qubit_pairs]


class QubitGrid:
    """Creates a grid object where the qubits are placed.
    Accepts a dataset whose dimension 'qubit' is used as the dimension on which the grid is built.
    It also accepts a parameter "grid_names" that specifies syntax for the position of each qubit on the grid. If none
    it assumes that qubit names are of the form: 'q-i,j' where i,j are integers describing the x and y coordinates of the grid.

    Iteration of the resulting grid can be done using 'grid_iter' defined in lib.plot_utils.

    :param ds: The dataset containing the names of the qubit under ds.qubit
    :params grid_names: a list of names in the required qubit names, in case the qubits names
                        given in a different format. Default is None

    :var fig: the created figure object
    :var all_axes: all the created axes, used and unused
    :var axes: a list of the axes relevant for the grid
    :name_dicts: a list containing the names of the qubit, taken from ds.qubit, in the
                convention of FacetGrid dict_names

    usage example:
    Assume we have a dataset with a data variable I, and a data coordinate q formated
    according to the naming convention. A way to plot the data on the grid would be

    '''
    from qualibration_libs.qua_datasets import grid_iter, QubitGrid
    grid = QubitGrid(ds)

    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].I.plot(ax = ax)
    '''

    If the names of the qubits are not of the acceptable form it is possible to use:

    '''
    from qualibration_libs.plot_utils import grid_iter, QubitGrid
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].I.plot(ax = ax)

    '''

    """

    def _list_clean(self, list_input_string):
        return [self._clean_up(input_string) for input_string in list_input_string]

    @staticmethod
    def _clean_up(input_string):
        return re.sub("[^0-9]", "", input_string)

    def __init__(
        self, ds: xr.Dataset, grid_names: Union[list[str], str] = None, size: int = 3, create_figure: bool = True
    ):
        if grid_names:
            if type(grid_names) == str:
                grid_names = [grid_names]
            grid_indices = [
                tuple(map(int, self._list_clean(grid_name.split(","))))
                for grid_name in grid_names
            ]
        else:
            grid_indices = [
                tuple(map(int, self._list_clean(ds.qubit.values[q_index].split(","))))
                for q_index in range(ds.qubit.size)
            ]

        if len(grid_indices) > 1:
            grid_name_mapping = dict(zip(grid_indices, ds.qubit.values))
        else:
            try:
                grid_name_mapping = dict(zip(grid_indices, [str(ds.qubit.values[0])]))
            except (Exception,):
                grid_name_mapping = dict(zip(grid_indices, [str(ds.qubit.values)]))

        grid_row_idxs = [idx[1] for idx in grid_indices]
        grid_col_idxs = [idx[0] for idx in grid_indices]
        min_grid_row = min(grid_row_idxs)
        min_grid_col = min(grid_col_idxs)
        shape = (
            max(grid_row_idxs) - min_grid_row + 1,
            max(grid_col_idxs) - min_grid_col + 1,
        )

        # Only create matplotlib figure if requested
        if create_figure:
            figure, all_axes = plt.subplots(
                *shape, figsize=(shape[1] * size, shape[0] * size), squeeze=False
            )

            grid_axes = []
            qubit_names = []

            for row, axis_row in enumerate(all_axes):
                for col, ax in enumerate(axis_row):
                    grid_row = max(grid_row_idxs) - row
                    grid_col = col + min_grid_col
                    if (grid_col, grid_row) in grid_indices:
                        grid_axes.append(ax)
                        name = grid_name_mapping.get((grid_col, grid_row))
                        if name is not None:
                            qubit_names.append(grid_name_mapping[(grid_col, grid_row)])
                    else:
                        ax.axis("off")

            self.fig = figure
            self.all_axes = all_axes
            self.axes = [grid_axes]
            self.name_dicts = [[{ds.qubit.name: value} for value in qubit_names]]
        else:
            # For Plotly-only usage, set matplotlib attributes to None
            self.fig = None
            self.all_axes = None
            self.axes = None
            self.name_dicts = None
        
        # Add Plotly-compatible attributes
        self.n_rows = shape[0]
        self.n_cols = shape[1]
        self.grid_positions = []
        self.plotly_name_dicts = []
        self.grid_order = []
        
        # Build grid order and positions for Plotly compatibility
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                grid_row = max(grid_row_idxs) - row
                grid_col = col + min_grid_col
                qubit_name = grid_name_mapping.get((grid_col, grid_row))
                self.grid_order.append(qubit_name)
                
                if qubit_name is not None:
                    self.plotly_name_dicts.append({ds.qubit.name: qubit_name})
                    self.grid_positions.append((row, col))
    
    def plotly_grid_iter(self) -> Iterator:
        """
        Generator to iterate over the QubitGrid for Plotly compatibility, yielding (grid_position, name_dict) for each qubit.
        Returns the actual grid position (row, col) rather than sequential index to preserve layout.
        """
        for i, name_dict in enumerate(self.plotly_name_dicts):
            row, col = self.grid_positions[i]
            yield (row, col), name_dict
    
    def get_subplot_titles(self, title_template: str = "Qubit {qubit}") -> List[str]:
        """
        Generate subplot titles for the full grid layout, including empty positions.
        
        Parameters
        ----------
        title_template : str
            Template string for titles. Use {qubit} as placeholder for qubit name.
            
        Returns
        -------
        List[str]
            List of subplot titles matching the grid layout (row-major order).
            Empty positions get empty string titles.
        """
        titles = []
        for qubit_name in self.grid_order:
            if qubit_name is not None:
                titles.append(title_template.format(qubit=qubit_name))
            else:
                titles.append("")
        return titles


def grid_iter(grid: xr.plot.FacetGrid) -> Tuple[matplotlib.axes.Axes, dict]:
    """Create a generator to iterate over a facet grid.
    For each iteration, return a tuple of (axis object, name dict of this axis).

    This is useful for adding annotations and additional data to facet grid figures.

    :param grid: The grid to iterate over
    :type grid: xr.plot.FacetGrid
    :yield: a tuple with the axis and name of the facet
    :rtype: _type_
    """
    for axr, ndr in zip(grid.axes, grid.name_dicts):
        for ax, nd in zip(axr, ndr):
            yield ax, nd

