"""Define a plotter that rely on the pandas plotting methods.

.. todo::
    Maybe should inherit from MatplotlibPlotter?

"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.axes import Axes

from core.list_of_elements.list_of_elements import ListOfElements
from plotter.i_plotter import IPlotter
from plotter.matplotlib_helper import (
    create_fig_if_not_exists,
    plot_section,
    plot_structure,
)
from util.dicts_output import markdown


class PandasPlotter(IPlotter):
    """A plotter that takes in pandas dataframe.

    .. note::
        Under the hood, this is matplotlib which is called.

    """

    def __init__(self, elts: ListOfElements) -> None:
        """Instantiate some common attributes."""
        super().__init__(elts)

    def _setup_fig(self, fignum: int, title: str, **kwargs) -> list[Axes]:
        """Setup the figure and axes."""
        axnum = 2
        if not self._structure:
            axnum = 1
        return create_fig_if_not_exists(
            axnum,
            title=title,
            sharex=self._sharex,
            num=fignum,
            clean_fig=True,
            **kwargs,
        )

    def _actual_plotting(
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

    def save_figure(self, axes: Axes, save_path: Path) -> None:
        figure = axes.get_figure()
        assert figure is not None
        return figure.savefig(save_path)

    def _plot_structure(
        self,
        axes: Sequence[Axes],
        elts: ListOfElements | None = None,
        x_axis: str = "z_abs",
    ) -> None:
        """Add a plot to show the structure of the linac."""
        if elts is None:
            elts = self._elts
        plot_structure(axes[-1], elts, x_axis)

        if not self._sections:
            return
        return self._plot_sections(axes[-1], elts, x_axis)

    def _plot_sections(
        self, axes: Any, elts: ListOfElements, x_axis: str
    ) -> None:
        """Add the sections on the structure plot."""
        return plot_section(axes, elts, x_axis)
