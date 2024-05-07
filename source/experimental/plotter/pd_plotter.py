"""Define a plotter that rely on the pandas plotting methods.

.. todo::
    Maybe should inherit from MatplotlibPlotter?

"""

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd
from matplotlib.axes import Axes

from experimental.plotter.matplotlib_plotter import MatplotlibPlotter
from util.dicts_output import markdown


class PandasPlotter(MatplotlibPlotter):
    """A plotter that takes in pandas dataframe.

    .. note::
        Under the hood, this is matplotlib which is called.

    """

    def _actual_plot(
        self,
        data: pd.DataFrame,
        ylabel: str,
        axes: Sequence[Axes],
        axes_index: int,
        xlabel: str = markdown["z_abs"],
        dump_no_numerical_data_to_plot: bool = False,
        **plot_kwargs: Any,
    ) -> Sequence[Axes]:
        """Create the plot itself."""
        try:
            data.plot(
                ax=axes[axes_index],
                sharex=self._sharex,
                grid=self._grid,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=self._legend,
                **plot_kwargs,
            )
        except TypeError as err:
            if dump_no_numerical_data_to_plot:
                logging.info(f"Dumped a Pandas.plot error: {err}.")
                return axes
            raise err
        return axes
