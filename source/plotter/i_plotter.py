"""Define the base class for all plotters."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, final

from core.list_of_elements.list_of_elements import ListOfElements
from util.dicts_output import markdown


class IPlotter(ABC):
    """The base plotting class."""

    _grid = True
    _sharex = True
    _legend = True
    _structure = True
    _sections = True

    def __init__(self, elts: ListOfElements) -> None:
        """Instantiate some base attributes."""
        self._elts = elts

    @final
    def plot(
        self,
        data: Any,
        ref_data: Any | None = None,
        save_path: Path | None = None,
        elts: ListOfElements | None = None,
        fignum: int = 1,
        axes_index: int = 0,
        title: str = "",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot the provided data.

        Parameters
        ----------
        data : Any
            Data to be plotted. According to the subclass, it can be a numpy
            array, a pandas dataframe...
        ref_data : Sequence[float] | None, optional
            Reference data, to plot if provided.
        save_path : Path | None, optional
            Where the figure will be saved. The default is None, in which case
            figure is not plotted.
        elts : ListOfElements | None, optional
            Elements to plot if :attr:`_structure` is True. If not provided, we
            take default :attr:`_elts` instead. Note that the colour of the
            failed, compensating, rephased cavities is given by this object.
            The default is None.
        fignum : int, optional
            Figure number. The default is 1.
        axes_index : int, optional
            Axes identifier. The default is 0, corresponding to the topmost
            sub-axes.
        title : str, optional
            Title of the figure. The default is an empty string.
        plot_kwargs : Any, optional
            Other keyword arguments passed to the :meth:`_actual_plotting`.

        Returns
        -------
        Any
            The created axes object(s).

        """
        axes = self._setup_fig(fignum, title)

        if ref_data is not None:
            self._actual_plot(
                ref_data, axes=axes, axes_index=axes_index, **plot_kwargs
            )

        self._actual_plot(
            data, axes=axes, axes_index=axes_index, **plot_kwargs
        )

        if self._structure:
            if elts is None:
                elts = self._elts
            self._plot_structure(axes, elts)

        if save_path is not None:
            self.save_figure(axes, save_path)

    @abstractmethod
    def _setup_fig(self, fignum: int, title: str, **kwargs) -> Sequence[Any]:
        """Create the figure.

        This method should create the figure with figure number ``fignum``,
        with title ``title``, and eventual keyword arguments. It must return
        one or several axes where data can be plotted.

        """

    @abstractmethod
    def _actual_plot(
        self,
        data: Any,
        ylabel: str,
        axes: Any,
        axes_index: int,
        xlabel: str = markdown["z_abs"],
        **plot_kwargs: Any,
    ) -> Any:
        """Create the plot itself."""

    @abstractmethod
    def _plot_structure(
        self,
        axes: Any,
        elts: ListOfElements | None = None,
        x_axis: str = "z_abs",
    ) -> None:
        """Add a plot to show the structure of the linac."""
        if elts is None:
            elts = self._elts
        if self._sections:
            self._plot_sections(axes, elts, x_axis)

    @abstractmethod
    def _plot_sections(
        self, axes: Any, elts: ListOfElements, x_axis: str
    ) -> None:
        """Add the sections on the structure plot."""

    @abstractmethod
    def save_figure(self, axes: Any, save_path: Path) -> None:
        """Save the created figure."""

    @final
    def plot_constants(
        self,
        axes: Any,
        constants: Iterable[float] | float,
        color: str = "red",
        ls: str = "dashed",
        **kwargs,
    ) -> None:
        """Add one or several constants to a plot."""
        if isinstance(constants, float | int):
            constants = (constants,)

        for constant in constants:
            self._actual_constant_plot(axes, constant, color, ls, **kwargs)

    @abstractmethod
    def _actual_constant_plot(
        self, axes: Any, constant: float, color: str, ls: str, **kwargs
    ) -> None:
        """Add one constant to a plot."""
