#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:35:54 2023.

@author: placais
"""

import os
import logging
from typing import Any, Callable
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as pat

from palettable.colorbrewer.qualitative import Dark2_8
from cycler import cycler

from util import helper
from core.accelerator import Accelerator
from beam_calculation.output import SimulationOutput
import util.dicts_output as dic

figure_type = matplotlib.figure.Figure
ax_type = matplotlib.axes._axes.Axes

font = {'family': 'serif', 'size': 25}
plt.rc('font', **font)
plt.rcParams['axes.prop_cycle'] = cycler(color=Dark2_8.mpl_colors)
plt.rcParams["figure.figsize"] = (19.2, 11.24)
plt.rcParams["figure.dpi"] = 100

FALLBACK_PRESETS = {'x_axis': 'z_abs',
                    'plot_section': True, 'clean_fig': False, 'sharex': True}
PLOT_PRESETS = {
    "energy": {'x_axis': 'z_abs',
               'all_y_axis': ["w_kin", "w_kin_err", "struct"],
               'num': 21},
    "phase": {'x_axis': 'z_abs',
              'all_y_axis': ["phi_abs", "phi_abs_err", "struct"],
              'num': 22},
    "cav": {'x_axis': 'elt_idx',
            'all_y_axis': ["v_cav_mv", "phi_s", "struct"],
            'num': 23},
    "emittance": {'x_axis': 'z_abs',
                  'all_y_axis': ["eps_zdelta", "struct"],
                  'num': 24},
    "twiss": {'x_axis': 'z_abs',
              'all_y_axis': ["alpha_w", "beta_w", "gamma_w"],
              'num': 25},
    "envelopes": {
        'x_axis': 'z_abs',
        'all_y_axis':
            ["envelope_pos_zdelta", "envelope_energy_zdelta", "struct"],
        'num': 26},
    "mismatch_factor": {'x_axis': 'z_abs',
                        'all_y_axis': ["mismatch_factor", "struct"],
                        'num': 27},
}
ERROR_PRESETS = {'w_kin_err': {'scale': 1., 'diff': 'simple'},
                 'phi_abs_err': {'scale': 1., 'diff': 'simple'},
                 }

# The one you generally want
ERROR_REFERENCE = "ref accelerator (1st solv w/ 1st solv, 2nd w/ 2nd)"

# These two are mostly useful when you want to study the differences between
# two solvers
# ERROR_REFERENCE = "ref accelerator (1st solver)"
# ERROR_REFERENCE = "ref accelerator (2nd solver)"


# =============================================================================
# Front end
# =============================================================================
def factory(accelerators: list[Accelerator], plots: dict[str, bool],
            **kwargs: bool) -> list[figure_type]:
    """Create all the desired plots."""
    if (kwargs['clean_fig']
            and not kwargs['save_fig']
            and len(accelerators) > 2):
        logging.warning("You will only see the plots of the last accelerators,"
                        " previous will be erased without saving.")

    ref_acc = accelerators[0]
    figs = [_plot_preset(preset, *(ref_acc, fix_acc),
                         **_proper_kwargs(preset, kwargs))
            for fix_acc in accelerators[1:]
            for preset, plot_me in plots.items() if plot_me]
    return figs


def plot_evaluate(z_m: np.ndarray, reference_values: list[dict],
                  fixed_values: list[dict], acceptable_values: list[dict],
                  lin_fix: Accelerator, evaluation: str = 'test',
                  save_fig: bool = True, num: int = 1) -> None:
    """Plot data from util.evaluate."""
    x_axis = 'z_abs'

    for i, (ref, fix, limits) in enumerate(zip(reference_values,
                                               fixed_values,
                                               acceptable_values)):
        n_axes = len(ref) + 1
        num += 1
        fig, axx = _create_fig_if_not_exists(n_axes, sharex=True, num=num,
                                             clean_fig=True)
        axx[-1].set_xlabel(dic.markdown[x_axis])
        # TODO : structure plot (needs a linac)

        for ax, (key, ref), fix, lim in zip(axx[:-1], ref.items(),
                                            fix.values(), limits.values()):
            ax.set_ylabel(dic.markdown[key])
            ax.grid(True)
            ax.plot(z_m, ref, label="TW ref")
            ax.plot(z_m, fix, label=lin_fix.name)

            for key_lim in ['max', 'min']:
                if key_lim in lim.keys() and lim[key_lim] is not None:
                    dat = lim[key_lim]
                    if isinstance(dat, float):
                        ax.axhline(dat, c='r', ls='--')
                        continue

                    ax.plot(z_m, dat, c='r', ls='--')

        axx[0].legend()
        _plot_structure(lin_fix, axx[-1], x_axis=x_axis)

        if save_fig:
            file = os.path.join(lin_fix.get('beam_calc_path'), '..',
                                f"{evaluation}_{i}.png")
            _savefig(fig, file)


# =============================================================================
# Used in factory
# =============================================================================
# Main func
def _plot_preset(str_preset: str, *args: Accelerator,
                 x_axis: str = 'z_abs', all_y_axis: list[str] | None = None,
                 save_fig: bool = True, **kwargs: bool | str | int,
                 ) -> plt.figure:
    """
    Plot a preset.

    Parameters
    ----------
    str_preset : str
        Key of PLOT_PRESETS.
    *args : Accelerator
        Accelerators to plot. In typical usage, args = (Working, Fixed)
        (previously: (Working, Broken, Fixed). Useful to reimplement?)
    x_axis : str, optional
        Name of the x axis. The default is 'z_abs'.
    all_y_axis : list[str] | None, optional
        Name of all the y axis. The default is None.
    save_fig : bool, optional
        To save Figures or not. The default is True.
    **kwargs : bool | str | int
        Holds all complementary data on the plots.
    """
    fig, axx = _create_fig_if_not_exists(len(all_y_axis), **kwargs)

    colors = None
    for i, (axe, y_axis) in enumerate(zip(axx, all_y_axis)):
        _make_a_subplot(axe, x_axis, y_axis, colors, *args, **kwargs)
        if i == 0:
            colors = _keep_colors(axe)
    axx[0].legend()
    axx[-1].set_xlabel(dic.markdown[x_axis])

    if save_fig:
        file = os.path.join(os.path.dirname(args[-1].get('beam_calc_path')),
                            f"{str_preset}.png")
        _savefig(fig, file)

    return fig


# Plot style
def _proper_kwargs(preset: str, kwargs: dict[str, bool]
                   ) -> dict[str, bool | int | str]:
    """Merge dicts, priority kwargs > PLOT_PRESETS > FALLBACK_PRESETS."""
    return FALLBACK_PRESETS | PLOT_PRESETS[preset] | kwargs


def _keep_colors(axe: ax_type) -> dict[str, str]:
    """Keep track of the color associated with each SimulationOutput."""
    lines = axe.get_lines()
    colors = {line.get_label(): line.get_color() for line in lines}
    return colors


def _y_label(y_axis: str) -> str:
    """Set the proper y axis label."""
    if '_err' in y_axis:
        key = ERROR_PRESETS[y_axis]['diff']
        y_label = dic.markdown["err_" + key]
        return y_label
    y_label = dic.markdown[y_axis]
    return y_label


# Data getters
def _single_simulation_data(axis: str, simulation_output: SimulationOutput
                            ) -> list[float] | None:
    """Get single data array from single SimulationOutput."""
    data = simulation_output.get(axis, to_numpy=False, to_deg=True)
    return data


def _single_simulation_all_data(x_axis: str, y_axis: str,
                                simulation_output: SimulationOutput
                                ) -> tuple[np.ndarray, np.ndarray,
                                           dict | None]:
    """Get x data, y data, kwargs from a SimulationOutput."""
    x_data = _single_simulation_data(x_axis, simulation_output)
    y_data = _single_simulation_data(y_axis, simulation_output)

    if None in (x_data, y_data):
        x_data = np.full((10, 1), np.NaN)
        y_data = np.full((10, 1), np.NaN)
        logging.warning(
            f"{x_axis} or {y_axis} not found in {simulation_output}")
        return x_data, y_data, None

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    plt_kwargs = dic.plot_kwargs[y_axis].copy()
    return x_data, y_data, plt_kwargs


def _single_accelerator_all_simulations_data(
        x_axis: str, y_axis: str, accelerator: Accelerator
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Get x_data, y_data, kwargs from all SimulationOutputs of Accelerator."""
    x_data, y_data, plt_kwargs = [], [], []
    ls = '-'
    for solver, simulation_output in accelerator.simulation_outputs.items():
        x_dat, y_dat, plt_kw = _single_simulation_all_data(x_axis, y_axis,
                                                           simulation_output)
        short_solver = solver.split('(')[0]
        plt_kw['label'] = ' '.join([accelerator.name, short_solver])
        plt_kw['ls'] = ls
        ls = '--'

        x_data.append(x_dat)
        y_data.append(y_dat)
        plt_kwargs.append(plt_kw)

    return x_data, y_data, plt_kwargs


def _all_accelerators_data(
        x_axis: str, y_axis: str, *accelerators: Accelerator
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Get x_data, y_data, kwargs from all Accelerators (<=> for 1 subplot)."""
    x_data, y_data, plt_kwargs = [], [], []

    key = y_axis
    error_plot = y_axis[-4:] == '_err'
    if error_plot:
        key = y_axis[:-4]

    for accelerator in accelerators:
        x_dat, y_dat, plt_kw = _single_accelerator_all_simulations_data(
            x_axis, key, accelerator)
        x_data += x_dat
        y_data += y_dat
        plt_kwargs += plt_kw

    if error_plot:
        fun_error = _error_calculation_function(y_axis)
        x_data, y_data, plt_kwargs = _compute_error(x_data, y_data, plt_kwargs,
                                                    fun_error)

    return x_data, y_data, plt_kwargs


# Error related
def _error_calculation_function(y_axis: str
                                ) -> tuple[Callable[[np.ndarray, np.ndarray],
                                                    np.ndarray],
                                           str]:
    """Set the function called to compute error."""
    scale = ERROR_PRESETS[y_axis]['scale']
    error_computers = {
        'simple': lambda y_ref, y_lin: scale * (y_ref - y_lin),
        'abs': lambda y_ref, y_lin: scale * np.abs(y_ref - y_lin),
        'rel': lambda y_ref, y_lin: scale * (y_ref - y_lin) / y_ref,
        'log': lambda y_ref, y_lin: scale * np.log10(np.abs(y_lin / y_ref)),
    }
    key = ERROR_PRESETS[y_axis]['diff']
    fun_error = error_computers[key]
    return fun_error


def _compute_error(x_data: list[np.ndarray], y_data: list[np.ndarray],
                   plt_kwargs: dict[str, int | bool | str],
                   fun_error: Callable[[np.ndarray, np.ndarray], np.ndarray]
                   ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute error with proper reference and proper function."""
    simulation_indexes = range(len(x_data))
    if ERROR_REFERENCE == "ref accelerator (1st solv w/ 1st solv, 2nd w/ 2nd)":
        i_ref = [i for i in range(len(x_data) // 2)]
    elif ERROR_REFERENCE == "ref accelerator (1st solver)":
        i_ref = [0]
    elif ERROR_REFERENCE == "ref accelerator (2nd solver)":
        i_ref = [1]
        if len(x_data) < 4:
            logging.error(f"{ERROR_REFERENCE = } not supported when only one "
                          "simulation is performed.")

            return np.full((10, 1), np.NaN), np.full((10, 1), np.NaN)
    else:
        logging.error(f"{ERROR_REFERENCE = }, which is not allowed. Check "
                      "allowed values in _compute_error.")
        return np.full((10, 1), np.NaN), np.full((10, 1), np.NaN)

    i_err = [i for i in simulation_indexes if i not in i_ref]
    indexes_ref_with_err = itertools.zip_longest(i_ref, i_err,
                                                 fillvalue=i_ref[0])

    x_data_error, y_data_error = [], []
    for (ref, err) in indexes_ref_with_err:
        x_interp, y_ref, _, y_err = helper.resample(x_data[ref], y_data[ref],
                                                    x_data[err], y_data[err])
        error = fun_error(y_ref, y_err)

        x_data_error.append(x_interp)
        y_data_error.append(error)

    plt_kwargs = [plt_kwargs[i] for i in i_err]
    return x_data_error, y_data_error, plt_kwargs


# Actual interface with matplotlib
def _make_a_subplot(axe: ax_type, x_axis: str, y_axis: str,
                    colors: dict[str, str] | None,
                    *accelerators: Accelerator, plot_section: bool = True,
                    **kwargs: bool | int | str) -> None:
    """Get proper data and plot it on an Axe."""
    if plot_section:
        _plot_section(accelerators[0], axe, x_axis=x_axis)

    if y_axis == 'struct':
        _plot_structure(accelerators[-1], axe, x_axis=x_axis)
        return

    all_my_data = _all_accelerators_data(x_axis, y_axis, *accelerators)
    for x_data, y_data, plt_kwargs in zip(
            all_my_data[0], all_my_data[1], all_my_data[2]):
        if colors is not None and plt_kwargs['label'] in colors:
            plt_kwargs['color'] = colors[plt_kwargs['label']]
        axe.plot(x_data, y_data, **plt_kwargs)

    axe.grid(True)
    axe.set_ylabel(_y_label(y_axis))
    _autoscale_based_on(axe, str_ignore='Broken')


# =============================================================================
# Basic helpers
# =============================================================================
def _create_fig_if_not_exists(axnum: int | list[int], sharex: bool = False,
                              num: int = 1, clean_fig: bool = False,
                              **kwargs: bool | str | int
                              ) -> tuple[figure_type, list[ax_type]]:
    """
    Check if figures were already created, create it if not.

    Parameters
    ----------
    axnum : int | list[int]
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
            _clean_fig([num])
        return fig, axlist
    fig = plt.figure(num)
    axlist = [fig.add_subplot(axnum[0])]
    shared_ax = None
    if sharex:
        shared_ax = axlist[0]
    axlist += [fig.add_subplot(num, sharex=shared_ax) for num in axnum[1:]]
    return fig, axlist


def _clean_fig(fignumlist):
    """Clean axis of Figs in fignumlist."""
    for fignum in fignumlist:
        fig = plt.figure(fignum)
        for axx in fig.get_axes():
            axx.cla()


def _autoscale_based_on(axx, str_ignore):
    """Rescale axis, ignoring Lines with str_ignore in their label."""
    lines = [line for line in axx.get_lines()
             if str_ignore not in line.get_label()]
    axx.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        datxy = np.vstack(line.get_data()).T
        axx.dataLim.update_from_data_xy(datxy, ignore=False)
    axx.autoscale_view()


def _savefig(fig, filepath):
    """Save the figure."""
    fig.savefig(filepath)
    logging.debug(f"Fig. saved in {filepath}")


# =============================================================================
# General plots
# =============================================================================
def plot_pty_with_data_tags(ax, x, y, idx_list, tags=True):
    """
    Plot y vs x.

    Data at idx_list are magnified with bigger points and data tags.
    """
    line, = ax.plot(x, y)
    ax.scatter(x[idx_list], y[idx_list], color=line.get_color())

    if tags:
        n = len(idx_list)
        for i in range(n):
            txt = str(np.round(x[idx_list][i], 4)) + ',' \
                + str(np.round(y[idx_list][i], 4))
            ax.annotate(txt, (x[idx_list][i], y[idx_list[i]]), size=8)


# =============================================================================
# Specific plots: structure
# =============================================================================
def _plot_structure(linac, ax, x_axis='z_abs'):
    """Plot a structure of the linac under study."""
    d_elem_plot = {
        'DRIFT': _plot_drift,
        'QUAD': _plot_quad,
        'FIELD_MAP': _plot_field_map,
    }
    d_x_axis = {  # first element is patch dimension. second is x limits
        'z_abs': lambda elt, i: [
            {'x0': elt.get('abs_mesh')[0], 'width': elt.length_m},
            [linac.elts[0].get('abs_mesh')[0],
             linac.elts[-1].get('abs_mesh')[-1]]
        ],
        'elt_idx': lambda elt, i: [
            {'x0': i, 'width': 1},
            [0, i]
        ]
    }

    for i, elt in enumerate(linac.elts):
        kwargs = d_x_axis[x_axis](elt, i)[0]
        ax.add_patch(d_elem_plot[elt.get('nature', to_numpy=False)](
            elt, **kwargs))

    ax.set_xlim(d_x_axis[x_axis](elt, i)[1])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylim([-.05, 1.05])


def _plot_drift(drift, x0, width):
    """Add a little rectangle to show a drift."""
    height = .4
    y0 = .3
    patch = pat.Rectangle((x0, y0), width, height, fill=False, lw=0.5)
    return patch


def _plot_quad(quad, x0, width):
    """Add a crossed large rectangle to show a quad."""
    height = 1.
    y0 = 0.
    path = np.array(([x0, y0], [x0 + width, y0], [x0 + width, y0 + height],
                     [x0, y0 + height], [x0, y0], [x0 + width, y0 + height],
                     [np.NaN, np.NaN], [x0, y0 + height], [x0 + width, y0]))
    patch = pat.Polygon(path, closed=False, fill=False, lw=0.5)
    return patch


def _plot_field_map(field_map, x0, width):
    """Add an ellipse to show a field_map."""
    height = 1.
    y0 = height * .5
    colors = {
        'nominal': 'green',
        'rephased (in progress)': 'yellow',
        'rephased (ok)': 'yellow',
        'failed': 'red',
        'compensate (in progress)': 'orange',
        'compensate (ok)': 'orange',
        'compensate (not ok)': 'orange',
    }
    patch = pat.Ellipse((x0 + .5 * width, y0), width, height, fill=True,
                        lw=0.5, fc=colors[field_map.get('status',
                                                        to_numpy=False)],
                        ec='k')
    return patch


def _plot_section(linac, ax, x_axis='z_abs'):
    """Add light grey rectangles behind the plot to show the sections."""
    dict_x_axis = {
        'last_elt_of_sec': lambda sec: sec[-1][-1],
        'z_abs': lambda elt: linac.get('z_abs', elt=elt, pos='out'),
        'elt_idx': lambda elt: elt.get('elt_idx') + 1,
    }
    x_ax = [0]
    for i, section in enumerate(linac.elts.by_section_and_lattice):
        elt = dict_x_axis['last_elt_of_sec'](section)
        x_ax.append(dict_x_axis[x_axis](elt))

    for i in range(len(x_ax) - 1):
        if i % 2 == 1:
            ax.axvspan(x_ax[i], x_ax[i + 1], ymin=-1e8, ymax=1e8, fill=True,
                       alpha=.1, fc='k')


# =============================================================================
# Specific plots: emittance ellipse
# =============================================================================
def _compute_ellipse_parameters(d_eq):
    """
    Compute the ellipse parameters so as to plot the ellipse.

    Parameters
    ----------
    d_eq : dict
        Holds ellipe equations parameters, defined as:
            Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0

    Return
    ------
    d_plot : dict
        Holds semi axis, center of ellipse, angle.
    """
    delta = d_eq["B"]**2 - 4. * d_eq["A"] * d_eq["C"]
    tmp1 = d_eq["A"] * d_eq["E"]**2 - d_eq["C"] * d_eq["D"]**2 \
        - d_eq["B"] * d_eq["D"] * d_eq["E"] + delta * d_eq["F"]
    tmp2 = np.sqrt((d_eq["A"] - d_eq["C"])**2 + d_eq["B"]**2)

    if np.abs(d_eq["B"]) < 1e-8:
        if d_eq["A"] < d_eq["C"]:
            theta = 0.
        else:
            theta = np.pi / 2.
    else:
        theta = np.arctan((d_eq["C"] - d_eq["A"] - tmp2) / d_eq["B"])

    d_plot = {
        "a": -np.sqrt(2. * tmp1 * (d_eq["A"] + d_eq["C"] + tmp2)) / delta,
        "b": -np.sqrt(2. * tmp1 * (d_eq["A"] + d_eq["C"] - tmp2)) / delta,
        "x0": (2. * d_eq["C"] * d_eq["D"] - d_eq["B"] * d_eq["E"]) / delta,
        "y0": (2. * d_eq["A"] * d_eq["E"] - d_eq["B"] * d_eq["D"]) / delta,
        "theta": theta,
    }
    return d_plot


def plot_ellipse(axx, d_eq, **plot_kwargs):
    """Perform the proper ellipse plotting."""
    d_plot = _compute_ellipse_parameters(d_eq)
    n_points = 10001
    var = np.linspace(0., 2. * np.pi, n_points)
    ellipse = np.array([d_plot["a"] * np.cos(var), d_plot["b"] * np.sin(var)])
    rotation = np.array([[np.cos(d_plot["theta"]), -np.sin(d_plot["theta"])],
                         [np.sin(d_plot["theta"]), np.cos(d_plot["theta"])]])
    ellipse_rot = np.empty((2, n_points))

    for i in range(n_points):
        ellipse_rot[:, i] = np.dot(rotation, ellipse[:, i])

    axx.plot(d_plot["x0"] + ellipse_rot[0, :],
             d_plot["y0"] + ellipse_rot[1, :],
             lw=0., marker='o', ms=.5, **plot_kwargs)


# TODO: move dicts into the function dedicated to dicts creation
def plot_ellipse_emittance(axx, accelerator, idx, phase_space="w"):
    """Plot the emittance ellipse and highlight interesting data."""
    # Extract Twiss and emittance at the index idx
    twi = accelerator.get("twiss_" + phase_space)[idx]
    eps = accelerator.get("eps_ " + phase_space)[idx]

    # Compute ellipse dimensions; ellipse equation:
    # Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0
    d_eq = {"A": twi[2], "B": 2. * twi[0], "C": twi[1], "D": 0., "E": 0.,
            "F": -eps}

    # Plot ellipse
    colors = {"Working": "k",
              "Broken": "r",
              "Fixed": "g"}
    color = colors[accelerator.name.split(" ")[0]]
    plot_kwargs = {"c": color}
    plot_ellipse(axx, d_eq, **plot_kwargs)

    # Set proper labels
    d_xlabel = {"z": r"Position $z$ [mm]",
                "zdelta": r"Position $z$ [mm]",
                "w": r"Phase $\phi$ [deg]"}
    axx.set_xlabel(d_xlabel[phase_space])

    d_ylabel = {"z": r"Speed $z'$ [%]",
                "zdelta": r"Speed $\delta p/p$ [mrad]",
                "w": r"Energy $W$ [MeV]"}
    axx.set_ylabel(d_ylabel[phase_space])

    form = "{:.3g}"
    # Max phase
    maxi_phi = np.sqrt(eps * twi[1])
    line = axx.axvline(maxi_phi, c='b')
    axx.axhline(-twi[0] * np.sqrt(eps / twi[1]), c=line.get_color())
    axx.get_xticklabels().append(
        plt.text(1.005 * maxi_phi, .05, form.format(maxi_phi),
                 va="bottom", rotation=90.,
                 transform=axx.get_xaxis_transform(), c=line.get_color())
    )

    # Max energy
    maxi_w = np.sqrt(eps * twi[2])
    line = axx.axhline(maxi_w)
    axx.axvline(-twi[0] * np.sqrt(eps / twi[2]), c=line.get_color())
    axx.get_yticklabels().append(
        plt.text(.005, .95 * maxi_w, form.format(maxi_w), va="top",
                 rotation=0.,
                 transform=axx.get_yaxis_transform(), c=line.get_color())
    )

    axx.grid(True)


def plot_fit_progress(hist_f, l_label, nature='Relative'):
    """Plot the evolution of the objective functions w/ each iteration."""
    _, axx = _create_fig_if_not_exists(1, num=32)
    axx = axx[0]

    scales = {'Relative': lambda x: x / x[0],
              'Absolute': lambda x: x, }

    # Number of objectives, number of evaluations
    n_f = len(l_label)
    n_iter = len(hist_f)
    iteration = np.linspace(0, n_iter - 1, n_iter)

    __f = np.empty([n_f, n_iter])
    for i in range(n_iter):
        __f[:, i] = scales[nature](hist_f)[i]

    for j, label in enumerate(l_label):
        axx.plot(iteration, __f[j], label=label)

    axx.grid(True)
    axx.legend()
    axx.set_xlabel("Iteration #")
    axx.set_ylabel(f"{nature} variation of error")
    axx.set_yscale('log')
