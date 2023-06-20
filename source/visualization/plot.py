#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:35:54 2023.

@author: placais
"""

import os
import logging
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as pat

from palettable.colorbrewer.qualitative import Dark2_8
from cycler import cycler

from util import helper
from core.accelerator import Accelerator
import util.dicts_output as dic

font = {'family': 'serif', 'size': 25}
plt.rc('font', **font)
plt.rcParams['axes.prop_cycle'] = cycler(color=Dark2_8.mpl_colors)
plt.rcParams["figure.figsize"] = (19.2, 11.24)
plt.rcParams["figure.dpi"] = 100

BASE_DICT = {'x_str': 'z_abs', 'reference': 'LW', 'replot_lw': False,
             'plot_section': True, 'plot_tw': False, 'clean_fig': False,
             'sharex': True}
DICT_PLOT_PRESETS = {
    "energy": {'x_str': 'z_abs',
               'l_y_str': ["w_kin", "w_kin_err", "struct"],
               'num': 21},
    "phase": {'x_str': 'z_abs',
              'l_y_str': ["phi_abs_array", "phi_abs_array_err", "struct"],
              'num': 22},
    "cav": {'x_str': 'elt_idx',
            'l_y_str': ["v_cav_mv", "phi_s", "struct"],
            'num': 23},
    "emittance": {'x_str': 'z_abs',
                  'l_y_str': ["eps_zdelta", "struct"],
                  'num': 24},
    "twiss": {'x_str': 'z_abs',
              'l_y_str': ["alpha_w", "beta_w", "gamma_w"],
              'num': 25},
    "envelopes": {
        'x_str': 'z_abs',
        'l_y_str': ["envelope_pos_zdelta", "envelope_energy_zdelta", "struct"],
        'num': 26},
    "mismatch_factor": {'x_str': 'z_abs',
                        'l_y_str': ["mismatch_factor", "struct"],
                        'num': 27},
}
DICT_ERROR_PRESETS = {'w_kin_err': {'scale': 1., 'diff': 'simple'},
                      'phi_abs_array_err': {'scale': 1., 'diff': 'simple'},
                      }


# =============================================================================
# Front end
# =============================================================================
def plot_preset(str_preset: str, *args, **kwargs):
    """
    Plot a preset.

    Parameters
    ----------
    str_preset : str
        Key of DICT_PLOT_PRESETS.
    args : Accelerators
        Accelerators to plot. In typical usage,
        *args = (Working, Broken, Fixed.)
    kwargs : dict
        Keys overriding DICT_PLOT_PRESETS and BASE_DICT.
    """
    # From preset, add keys that are not already in kwargs
    for key, val in DICT_PLOT_PRESETS[str_preset].items():
        if key not in kwargs:
            kwargs[key] = val
    # From base dict, add keys that are not already in kwargs
    for key, val in BASE_DICT.items():
        if key not in kwargs:
            kwargs[key] = val

    # Extract data used everywhere
    x_str, l_y_str = kwargs['x_str'], kwargs['l_y_str']
    plot_tw = kwargs['plot_tw']

    fig, axx = create_fig_if_not_exists(
        len(l_y_str), sharex=True, num=kwargs['num'],
        clean_fig=kwargs['clean_fig'])
    axx[-1].set_xlabel(dic.d_markdown[x_str])

    d_colors = {}
    for ax, y_str in zip(axx, l_y_str):
        section_already_plotted = False
        for arg in args:
            if kwargs["plot_section"] and not section_already_plotted:
                _plot_section(arg, ax, x_axis=x_str)
                section_already_plotted = True
            if y_str == 'struct':
                _plot_structure(arg, ax, x_axis=x_str)
                continue
        if y_str == 'struct':   # FIXME
            continue

        x_data, y_data, l_kwargs =  \
            _concatenate_all_data(x_str, y_str, *args, plot_tw=plot_tw,
                                  reference=kwargs['reference'])
        d_colors = _plot_all_data(ax, x_data, y_data, l_kwargs, d_colors)

        # Rescale
        _autoscale_based_on(ax, str_ignore='Broken')

        ax.grid(True)
        if y_str[-3:] == 'err':
            diff = DICT_ERROR_PRESETS[y_str]['diff']
            ax.set_ylabel(dic.d_markdown['err_' + diff])
            continue

        ax.set_ylabel(dic.d_markdown[y_str])
        # TODO handle linear vs log

    axx[0].legend()

    if kwargs['save_fig']:
        fixed_lin = args[-1]
        file = os.path.join(fixed_lin.get('out_lw'), '..', f"{str_preset}.png")
        _savefig(fig, file)


def plot_evaluate(z_m: np.ndarray, reference_values: list[dict],
                  fixed_values: list[dict], acceptable_values: list[dict],
                  lin_fix: Accelerator, evaluation: str = 'test',
                  save_fig: bool = True, num: int = 0) -> None:
    """Plot data from util.evaluate."""
    x_str = 'z_abs'

    for i, (ref, fix, limits) in enumerate(zip(reference_values,
                                               fixed_values,
                                               acceptable_values)):
        n_axes = len(ref) + 1
        num += 1
        fig, axx = create_fig_if_not_exists(n_axes, sharex=True, num=num,
                                            clean_fig=True)
        axx[-1].set_xlabel(dic.d_markdown[x_str])
        # TODO : structure plot (needs a linac)

        for ax, (key, ref), fix, lim in zip(axx[:-1], ref.items(),
                                            fix.values(), limits.values()):
            ax.set_ylabel(dic.d_markdown[key])
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
        _plot_structure(lin_fix, axx[-1], x_axis=x_str)

        if save_fig:
            file = os.path.join(lin_fix.get('out_lw'), '..',
                                f"{evaluation}_{i}.png")
            _savefig(fig, file)


# =============================================================================
# Used in plot_preset
# =============================================================================
def _concatenate_all_data(x_str: str, y_str: str, *args, plot_tw: bool = False,
                          reference: str = 'self'):
    """
    Get all the data that should be plotted.

    Parameters
    ----------
    x_str : str
        Name of x data.
    y_str : str
        Name of y data.
    *args : Accelerators
        Tuple of linacs which data should be gathered.
    plot_tw : bool, optional
        If TW data should be shown. The default is False.
    reference : str, optional
        How error is calculated, see d_ref in _err. The default is 'self'.

    Returns
    -------
    x_data : list of np.ndarray
        All x data.
    y_data : list of np.ndarray
        All y_data.
    l_kwargs : list of dict
        matplotlib kwargs for every dataset.

    """
    x_data = []
    y_data = []
    l_kwargs = []

    source = 'TW.multipart'
    # FIXME Get all the data that should be plotted.
    if y_str in ['v_cav_mv', 'phi_s']:
        source = 'cav_param'

    plot_error = y_str[-3:] == 'err'
    if plot_error:
        x_data, y_data, l_kwargs = _err(x_str, y_str, *args, plot_tw=plot_tw,
                                        reference=reference)
        return x_data, y_data, l_kwargs

    for arg in args:
        x_dat, y_dat, kw = _data_from(x_str, y_str, arg)
        x_data.append(x_dat), y_data.append(y_dat), l_kwargs.append(kw)

        # TODO handle multipart or envelope
        if plot_tw:
            x_dat, y_dat, kw = _data_from(x_str, y_str, arg, source=source)
            x_data.append(x_dat), y_data.append(y_dat), l_kwargs.append(kw)

    return x_data, y_data, l_kwargs


def _data_from(x_str: str, y_str: str, arg: tuple[Accelerator | dict],
               source: str = 'LW') -> tuple[np.ndarray, np.ndarray, dict]:
    """Get data."""
    if source == 'cav_params':
        logging.critical("Legacy cav_params. How was this supposed to work?")
    getters = {
        'TW.envelope': lambda x, arg:
            _data_from_tw(x, arg.tracewin_simulation.results_envelope),
        'TW.multipart': lambda x, arg:
            _data_from_tw(x, arg.tracewin_simulation.results_multipart),
        'LW': lambda x, arg: _data_from_lw(x, arg)
    }
    getter = getters[source]

    try:
        x_dat, y_dat = getter(x_str, arg), getter(y_str, arg)
    except AttributeError:
        logging.error(f"Data {x_str} and/or {y_str} not found in {arg}.")
        x_dat, y_dat = np.full((10, 1), np.NaN), np.full((10, 1), np.NaN)

    kw = dic.d_plot_kwargs[y_str].copy()
    kw['label'] = arg.name
    if source != 'LW':
        kw['ls'] = '--'
        kw['label'] = 'TW ' + kw['label']
    return x_dat, y_dat, kw


def _data_from_lw(data_name: str, linac: Accelerator) -> Any:
    """Get the data calculated by LightWin."""
    data = linac.get(data_name, to_deg=True)
    return data


def _data_from_tw(data_name: str, tracewin_results: dict,
                  warn_missing: bool = False) -> Any:
    """Get the data calculated by TraceWin, already loaded."""
    out = None

    # Data recomputed from TW simulation
    if data_name in tracewin_results.keys():
        out = tracewin_results[data_name]

    # Raw data from TW simulation
    if data_name in dic.d_lw_to_tw.keys():
        key = dic.d_lw_to_tw[data_name]
        if key in tracewin_results.keys():
            out = tracewin_results[key]

    # If need to be rescaled or modified
    if out is not None and data_name in dic.d_lw_to_tw_func.keys():
        out = dic.d_lw_to_tw_func[data_name](out)

    # Not implemented
    if warn_missing and out is None:
        logging.warning(f"{data_name} not found for TW.")
    return out


def _err(x_str, y_str, *args, plot_tw=False, reference='self'):
    """Calculate error with a reference calculation."""
    # We expect the first arg to be the reference Accelerator
    assert args[0].get('name') == 'Working'

    d_ref = {
        'LW': lambda source: 'LW',          # Error calculated w.r.t LW
        'TW.multipart': lambda source: 'TW.multipart',   # Error calculated w.r.t TW
        'self': lambda source: source}      # LW error w.r.t LW, TW w.r.t TW

    # Set up a scale (for example if the error is very small)
    scale = DICT_ERROR_PRESETS[y_str]['scale']
    d_diff = {
        'simple': lambda y_ref, y_lin: scale * (y_ref - y_lin),
        'abs': lambda y_ref, y_lin: scale * np.abs(y_ref - y_lin),
        'rel': lambda y_ref, y_lin: scale * (y_ref - y_lin) / y_ref,
        'log': lambda y_ref, y_lin: scale * np.log10(np.abs(y_lin / y_ref)),
        }
    fun_diff = d_diff[DICT_ERROR_PRESETS[y_str]['diff']]

    x_data, y_data, l_kwargs = [], [], []
    key = y_str[:-4]
    for arg in args[1:]:
        source = 'LW'
        ref = d_ref[source](reference)
        x_ref, y_ref, _ = _data_from(x_str, key, args[0], source=ref)
        __x, __y, kw = _data_from(x_str, key, arg, source=source)

        x_data.append(__x)
        diff = None
        if __y is not None:
            if __y.shape != y_ref.shape:
                x_ref, y_ref, __x, __y = helper.resample(x_ref, y_ref,
                                                         __x, __y)
            diff = fun_diff(y_ref, __y)
        y_data.append(diff)
        l_kwargs.append(kw)

        if plot_tw:
            source = 'TW.multipart'
            ref = d_ref[source](reference)
            x_ref, y_ref, _ = _data_from(x_str, key, args[0], source=ref)
            __x, __y, kw = _data_from(x_str, key, arg, source=source)

            x_data.append(__x)
            diff = None
            if __y is not None:
                if __y.shape != y_ref.shape:
                    x_ref, y_ref, __x, __y = helper.resample(x_ref, y_ref,
                                                             __x, __y)
                diff = fun_diff(y_ref, __y)
            y_data.append(diff)
            l_kwargs.append(kw)

    return x_data, y_data, l_kwargs


def _plot_all_data(axx, x_data, y_data, l_kwargs, d_colors):
    """Plot given data on give axis. Keep same colors for LW and TW data."""
    for x, y, kw in zip(x_data, y_data, l_kwargs):
        key_color = kw['label'].removeprefix('TW ')

        if key_color in d_colors.keys():
            kw['color'] = d_colors[key_color]

        if y is None:
            continue
        line, = axx.plot(x, y, **kw)

        if key_color not in d_colors.keys():
            d_colors[key_color] = line.get_color()

    return d_colors


def _savefig(fig, filepath):
    """Save the figure."""
    fig.savefig(filepath)
    logging.debug(f"Fig. saved in {filepath}")


# =============================================================================
# Basic helpers
# =============================================================================
def create_fig_if_not_exists(axnum, sharex=False, num=1, clean_fig=False):
    """
    Check if figures were already created, create it if not.

    Parameters
    ----------
    axnum : list of int or int
        Axes indexes as understood by fig.add_subplot or number of desired
        axes.
    sharex : boolean, opt
        If x axis should be shared.
    num : int, opt
        Fig number.
    """
    if isinstance(axnum, int):
        # We make a one-column, axnum rows figure
        axnum = range(100 * axnum + 11, 101 * axnum + 11)

    if plt.fignum_exists(num):
        fig = plt.figure(num)
        axlist = fig.get_axes()

        if clean_fig:
            _clean_fig([num])
        return fig, axlist

    fig = plt.figure(num)
    axlist = []
    axlist.append(fig.add_subplot(axnum[0]))

    d_sharex = {True: axlist[0], False: None}

    for i in axnum[1:]:
        axlist.append(fig.add_subplot(i, sharex=d_sharex[sharex]))
    return fig, axlist


def _clean_fig(fignumlist):
    """Clean axis of Figs in fignumlist."""
    for fignum in fignumlist:
        fig = plt.figure(fignum)
        for axx in fig.get_axes():
            axx.cla()


# TODO still used?
def _empty_fig(fignum):
    """Return True if at least one axis of Fig(fignum) has no line."""
    out = False
    if plt.fignum_exists(fignum):
        fig = plt.figure(fignum)
        axlist = fig.get_axes()
        for axx in axlist:
            if axx.lines == []:
                out = True
    return out


def _autoscale_based_on(axx, str_ignore):
    """Rescale axis, ignoring Lines with str_ignore in their label."""
    lines = [line for line in axx.get_lines()
             if str_ignore not in line.get_label()]
    axx.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        datxy = np.vstack(line.get_data()).T
        axx.dataLim.update_from_data_xy(datxy, ignore=False)
    axx.autoscale_view()


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
    d_colors = {
        'nominal': 'green',
        'rephased (in progress)': 'yellow',
        'rephased (ok)': 'yellow',
        'failed': 'red',
        'compensate (in progress)': 'orange',
        'compensate (ok)': 'orange',
        'compensate (not ok)': 'orange',
    }
    patch = pat.Ellipse((x0 + .5 * width, y0), width, height, fill=True,
                        lw=0.5, fc=d_colors[field_map.get('status',
                                                          to_numpy=False)],
                        ec='k')
    return patch


def _plot_section(linac, ax, x_axis='z_abs'):
    """Add light grey rectangles behind the plot to show the sections."""
    dict_x_axis = {
        'last_elt_of_sec': lambda sec: sec[-1][-1],
        'z_abs': lambda elt: linac.get('z_abs')[elt.idx['s_out']],
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
            theta = np.pi/2.
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
    """The proper ellipse plotting."""
    d_plot = _compute_ellipse_parameters(d_eq)
    n_points = 10001
    var = np.linspace(0., 2. * np.pi, n_points)
    ellipse = np.array([d_plot["a"] * np.cos(var), d_plot["b"] * np.sin(var)])
    rotation = np.array([[np.cos(d_plot["theta"]), -np.sin(d_plot["theta"])],
                         [np.sin(d_plot["theta"]),  np.cos(d_plot["theta"])]])
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
    d_colors = {"Working": "k",
                "Broken": "r",
                "Fixed": "g"}
    color = d_colors[accelerator.name.split(" ")[0]]
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
    _, axx = create_fig_if_not_exists(1, num=32)
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
