import re
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
import matplotlib
import xarray as xr
import numpy as np


def grid_pair_names(qubit_pairs) -> Tuple[List[str], List[str]]:
    """ "
    Runs over defined qubit pairs and returns a list of the grid_name attribute of each qubit, returns a list of the grid location and a list of the qubit pair names
    """
    return [f"{qp.qubit_control.grid_location}-{qp.qubit_target.grid_location}" for qp in qubit_pairs], [
        qp.name for qp in qubit_pairs
    ]


class QubitPairGrid:
    # TODO: not sure this is working, to check...
    """Creates a grid object where qubit pairs are placed on a grid.
    The grid is built with references to the qubit pair names,
    which should of the form: 'q-i,j-q-n,m' where i,j and n,m are
    integers describing the x and y coordinates of the qubits of
    the pair on a qubit grid.

    Iteration of the resulting grid can be done using 'grid_iter'
    defined in lib.qua_datasets

    :param ds: The ds containing the names of the qubit in ds.qubit

    :var fig: the created figure object
    :var all_axes: all the created axes, used and unused
    :var axes: a list of the axes relevant for the grid
    :name_dicts: a list containing the names of the qubit, taken from ds.qubit, in the
                convention of FacetGrid dict_names

    usage example:
    Assume we have a dataset with a data variable I, and a data coordinate q formated
    according to the naming convention. A way to plot the data on the grid would be

    '''
    from qualibration_libs.qua_datasets import grid_iter
    grid_names = [ dict(zip(q.extras_[0:len(q.extras_):2],q.extras_[1:len(q.extras_):2]))['grid_name'] for q in qubits]
    grid = coupler_grid(ds, grid_names)

    for ax, coupler in grid_iter(grid):
        ds.loc[coupler].I.plot(ax = ax)
    '''

    """

    @staticmethod
    def _convert_to_int(incoming_string):
        return tuple(map(int, incoming_string.split("_")))

    def _list_clean(self, list_input_string):
        return [self._clean_up(input_string) for input_string in list_input_string]

    @staticmethod
    def _clean_up(input_string):
        return re.sub("[^0-9]", "", input_string)

    def __init__(self, grid_names: list[str], qubit_pair_names: list[str], size: int = 4):
        if len(grid_names) > 1:
            qubit_indices = [
                tuple(
                    [
                        tuple(map(int, self._list_clean(gp.split("-")[0].split(",")))),
                        tuple(map(int, self._list_clean(gp.split("-")[1].split(",")))),
                    ]
                )
                for gp in grid_names
            ]
        else:
            qubit_indices = [
                tuple(
                    [
                        tuple(map(int, self._list_clean(gp.split("-")[0].split(",")))),
                        tuple(map(int, self._list_clean(gp.split("-")[1].split(",")))),
                    ]
                )
                for gp in grid_names
            ]
        row_diffs = [pair[1][0] - pair[0][0] for pair in qubit_indices]
        col_diffs = [pair[1][1] - pair[0][1] for pair in qubit_indices]
        coupler_indices = [[2 * pair[0][1], 2 * pair[0][0]] for pair in qubit_indices]
        for k, (col_diff, row_diff) in enumerate(zip(col_diffs, row_diffs)):
            coupler_indices[k][0] += col_diff
            coupler_indices[k][1] += row_diff
        coupler_indices = [tuple(coupler) for coupler in coupler_indices]

        grid_row_idxs = [idx[0] for idx in coupler_indices]
        grid_col_idxs = [idx[1] for idx in coupler_indices]
        min_grid_row = min(grid_row_idxs)
        min_grid_col = min(grid_col_idxs)
        shape = (
            max(grid_row_idxs) - min_grid_row + 1,
            max(grid_col_idxs) - min_grid_col + 1,
        )

        figure, all_axes = plt.subplots(*shape, figsize=(shape[1] * size, shape[0] * size), squeeze=False)

        if shape == (1, 1):
            # If (1, 1), subplots returns a single axis, which we convert into
            # a nested array
            axes = np.array(((all_axes,),))
        else:
            # If (1, N) or (N, 1), subplots returns a 1D array of axes, which we
            # convert into a 2D array.
            axes = all_axes.reshape(shape)

        axes = []
        qubit_names = []

        for row, axis_row in enumerate(all_axes):
            for col, ax in enumerate(axis_row):
                grid_row = max(grid_row_idxs) - row
                grid_col = col + min_grid_col
                if (grid_row, grid_col) in coupler_indices:

                    axes.append(ax)
                    qubit_names.append(qubit_pair_names[coupler_indices.index((grid_row, grid_col))])
                else:
                    ax.axis("off")
        self.fig = figure
        self.all_axes = all_axes
        self.axes = [axes]
        self.name_dicts = [[{"qubit": qubit_pair_name} for qubit_pair_name in qubit_names]]


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

    def __init__(self, ds: xr.Dataset, grid_names: Union[list[str], str] = None, size: int = 3):
        if grid_names:
            if type(grid_names) == str:
                grid_names = [grid_names]
            grid_indices = [tuple(map(int, self._list_clean(grid_name.split(",")))) for grid_name in grid_names]
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

        figure, all_axes = plt.subplots(*shape, figsize=(shape[1] * size, shape[0] * size), squeeze=False)

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
