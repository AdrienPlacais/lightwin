"""Define a plotter that rely on the pandas plotting methods.

.. todo::
    Maybe should inherit from MatplotlibPlotter?

"""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from matplotlib.axes import Axes

from plotter.matplotlib_plotter import MatplotlibPlotter
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
        **plot_kwargs: Any,
    ) -> Sequence[Axes]:
        """Create the plot itself."""
        data.plot(
            ax=axes[axes_index],
            sharex=self._sharex,
            grid=self._grid,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=self._legend,
            **plot_kwargs,
        )
        return axes
