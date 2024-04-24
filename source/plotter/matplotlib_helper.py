"""Create helper functions specific to matplotlib-based plotters."""

from collections.abc import Sequence

import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from core.elements.aperture import Aperture
from core.elements.bend import Bend
from core.elements.drift import Drift
from core.elements.edge import Edge
from core.elements.field_maps.field_map import FieldMap
from core.elements.field_maps.field_map_100 import FieldMap100
from core.elements.field_maps.field_map_1100 import FieldMap1100
from core.elements.field_maps.field_map_7700 import FieldMap7700
from core.elements.quad import Quad
from core.list_of_elements.list_of_elements import ListOfElements


def plot_structure(
    axes: Axes, elts: ListOfElements, x_axis: str = "z_abs"
) -> None:
    """Plot structure of the linac under study."""
    type_to_plot_func = {
        Aperture: _plot_aperture,
        Bend: _plot_bend,
        Drift: _plot_drift,
        Edge: _plot_edge,
        FieldMap: _plot_field_map,
        FieldMap100: _plot_field_map,
        FieldMap1100: _plot_field_map,
        FieldMap7700: _plot_field_map,
        Quad: _plot_quad,
    }

    patch_kw = {
        "z_abs": lambda elt, _: {
            "x_0": elt.get("abs_mesh")[0],
            "width": elt.length_m,
        },
        "elt_idx": lambda _, idx: {"x_0": idx, "width": 1},
    }
    x_limits = {
        "z_abs": [elts[0].get("abs_mesh")[0], elts[-1].get("abs_mesh")[-1]],
        "elt_idx": [0, len(elts)],
    }

    for i, elt in enumerate(elts):
        kwargs = patch_kw[x_axis](elt, i)
        plot_func = type_to_plot_func.get(type(elt), _plot_drift)
        axes.add_patch(plot_func(elt, **kwargs))

    axes.set_xlim(x_limits[x_axis])
    axes.set_yticklabels([])
    axes.set_yticks([])
    axes.set_ylim((-0.55, 0.55))


def _plot_aperture(
    aperture: Aperture, x_0: float, width: float
) -> pat.Rectangle:
    """Add a thin line to show an aperture."""
    height = 1.0
    y_0 = -height * 0.5
    patch = pat.Rectangle((x_0, y_0), width, height, fill=False, lw=0.5)
    return patch


def _plot_bend(bend: Bend, x_0: float, width: float) -> pat.Rectangle:
    """Add a greyed rectangle to show a bend."""
    height = 0.7
    y_0 = -height * 0.5
    patch = pat.Rectangle(
        (x_0, y_0), width, height, fill=True, fc="gray", lw=0.5
    )
    return patch


def _plot_drift(drift: Drift, x_0: float, width: float) -> pat.Rectangle:
    """Add a little rectangle to show a drift."""
    height = 0.4
    y_0 = -height * 0.5
    patch = pat.Rectangle((x_0, y_0), width, height, fill=False, lw=0.5)
    return patch


def _plot_field_map(
    field_map: FieldMap, x_0: float, width: float
) -> pat.Ellipse:
    """Add an ellipse to show a field_map."""
    height = 1.0
    y_0 = 0.0
    colors = {
        "nominal": "green",
        "rephased (in progress)": "olive",
        "rephased (ok)": "olive",
        "failed": "red",
        "compensate (in progress)": "green",
        "compensate (ok)": "orange",
        "compensate (not ok)": "orange",
    }
    color = colors[field_map.get("status", to_numpy=False)]
    patch = pat.Ellipse(
        (x_0 + 0.5 * width, y_0),
        width,
        height,
        fill=True,
        lw=0.5,
        fc=color,
        ec="k",
    )
    return patch


def _plot_edge(edge: Edge, x_0: float, width: float) -> pat.Rectangle:
    """Add a thin line to show an edge."""
    height = 1.0
    y_0 = -height * 0.5
    patch = pat.Rectangle((x_0, y_0), width, height, fill=False, lw=0.5)
    return patch


def _plot_quad(quad: Quad, x_0: float, width: float) -> pat.Polygon:
    """Add a crossed large rectangle to show a quad."""
    height = 1.0
    y_0 = -height * 0.5
    path = np.array(
        (
            [x_0, y_0],
            [x_0 + width, y_0],
            [x_0 + width, y_0 + height],
            [x_0, y_0 + height],
            [x_0, y_0],
            [x_0 + width, y_0 + height],
            [np.NaN, np.NaN],
            [x_0, y_0 + height],
            [x_0 + width, y_0],
        )
    )
    patch = pat.Polygon(path, closed=False, fill=False, lw=0.5)
    return patch


def plot_section(
    axes: Axes, elts: ListOfElements, x_axis: str = "z_abs"
) -> None:
    """Add light grey rectangles behind the plot to show the sections."""
    dict_x_axis = {
        "last_elt_of_sec": lambda sec: sec[-1][-1],
        "z_abs": lambda elt: elts.get("z_abs", elt=elt, pos="out"),
        "elt_idx": lambda elt: elt.get("elt_idx") + 1,
    }
    x_ax = [0]
    by_section_and_lattice = elts.by_section_and_lattice
    assert by_section_and_lattice is not None
    for i, section in enumerate(by_section_and_lattice):
        elt = dict_x_axis["last_elt_of_sec"](section)
        x_ax.append(dict_x_axis[x_axis](elt))

    for i in range(len(x_ax) - 1):
        if i % 2 == 1:
            axes.axvspan(
                x_ax[i],
                x_ax[i + 1],
                ymin=-1e8,
                ymax=1e8,
                fill=True,
                alpha=0.1,
                fc="k",
            )


def create_fig_if_not_exists(
    axnum: int | Sequence[int],
    sharex: bool = False,
    num: int = 1,
    clean_fig: bool = False,
    **kwargs: bool | str | int,
) -> list[Axes]:
    """
    Check if figures were already created, create it if not.

    Parameters
    ----------
    axnum : int | Sequence[int]
        Axes indexes as understood by fig.add_subplot or number of desired
        axes.
    sharex : boolean, optional
        If x axis should be shared. The default is False.
    num : int, optional
        Fig number. The default is 1.
    clean_fig: bool, optional
        If the previous plot should be erased from Figure. The default is
        False.

    """
    if isinstance(axnum, int):
        # We make a one-column, `axnum` rows figure
        axnum = range(100 * axnum + 11, 101 * axnum + 11)

    if plt.fignum_exists(num):
        fig = plt.figure(num)
        axlist = fig.get_axes()
        if clean_fig:
            clean_figure([num])
        return axlist
    fig = plt.figure(num)
    axlist = [fig.add_subplot(axnum[0])]
    shared_ax = None
    if sharex:
        shared_ax = axlist[0]
    axlist += [fig.add_subplot(num, sharex=shared_ax) for num in axnum[1:]]
    return axlist


def clean_figure(fignumlist: Sequence[int]) -> None:
    """Clean axis of Figs in fignumlist."""
    for fignum in fignumlist:
        fig = plt.figure(fignum)
        clean_axes(fig.get_axes())


def clean_axes(axlist: Sequence[Axes]) -> None:
    """Clean given axis."""
    for axx in axlist:
        axx.cla()
