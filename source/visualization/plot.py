#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:35:54 2023.

@author: placais
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as pat
from collections import OrderedDict

from palettable.colorbrewer.qualitative import Dark2_8
from cycler import cycler

import util.helper as helper
import util.tracewin_interface as tw
from util.dicts_output import d_markdown, d_plot_kwargs, d_lw_to_tw, d_scale_tw_to_lw

font = {'family': 'serif', 'size': 25}
plt.rc('font', **font)
plt.rcParams['axes.prop_cycle'] = cycler(color=Dark2_8.mpl_colors)
plt.rcParams["figure.figsize"] = (19.2, 11.24)
plt.rcParams["figure.dpi"] = 100

BASE_DICT = {'x_str': 'z_abs', 'reference': 'LW', 'replot_lw': False,
             'plot_section': True, 'plot_tw': False,
             'sharex': True}
DICT_PLOT_PRESETS = {
    "energy": {'x_str': 'z_abs',
               'l_y_str': ["w_kin", "err_abs", "struct"],
               'num': 21},
    "phase": {'x_str': 'z_abs',
              'l_y_str': ["phi_abs_array", "err_abs", "struct"],
              'num': 22},
    "cav": {'x_str': 'z_abs',
            'l_y_str': ["v_cav_mv", "k_e", "phi_s", "struct"],
            'num': 23},
    "emittance": {'x_str': 'z_abs',
                  'l_y_str': ["eps_zdelta", "struct"],
                  'num': 24},
    "twiss": {'x_str': 'z_abs',
              'l_y_str': ["alpha_w", "beta_w", "gamma_w"],
              'num': 25},
    "envelopes": {'x_str': 'z_abs',
                  'l_y_str': ["envelope_pos_w", "envelope_energy_w", "struct"],
                  'num': 26},
    "mismatch factor": {'x_str': 'z_abs',
                        'l_y_str': ["mismatch factor", "struct"],
                        'num': 27},
}

# =============================================================================
# Front end
# =============================================================================
def plot_preset(str_preset, *args, **kwargs):
    """
    Plot a preset.

    Parameters
    ----------
    str_preset : string
        Key of DICT_PLOT_PRESETS.
    args : Accelerators
        Accelerators to plot. In typical usage, *args = (Working, Broken,
                                                         Fixed.)
    kwargs : dict
        Keys overriding DICT_PLOT_PRESETS and BASE_DICT.
    """
    plt.close('all') # FIXME
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

    fig, axx = create_fig_if_not_exists(len(l_y_str), sharex=True,
                                        num=kwargs['num'])
    axx[-1].set_xlabel(d_markdown[x_str])

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

        x_data, y_data, l_kwargs =  _concatenate_all_data(x_str, y_str, *args,
                                                          plot_tw=plot_tw)
        d_colors = _plot_all_data(ax, x_data, y_data, l_kwargs, d_colors)

        ax.set_ylabel(d_markdown[y_str])
        ax.grid(True)
        # TODO handle linear vs log

    axx[0].legend()

    if kwargs['save_fig']:
        fixed_lin = args[-1]
        file = os.path.join(fixed_lin.get('out_lw'), '..', f"{str_preset}.png")
        _savefig(fig, file)


# =============================================================================
# Used in plot_preset
# =============================================================================
def _concatenate_all_data(x_str, y_str, *args, plot_tw=False):
    """Get all the data that should be plotted."""
    x_data = []
    y_data = []
    l_kwargs = []

    plot_error = y_str[:3] == 'err'
    if plot_error:
        raise IOError('Error plot to implement.')

    for arg in args:
        key = arg.name
        x_data.append(_data_from_lw(arg, x_str))
        y_data.append(_data_from_lw(arg, y_str))

        kw = d_plot_kwargs[y_str].copy()
        kw['label'] = key
        l_kwargs.append(kw)

        # TODO handle multipart or envelope
        if plot_tw:
            key = f"TW {arg.name}"
            tw_results = arg.tw_results['multipart']

            x_data.append(None)
            y_data.append(None)

            kw = d_plot_kwargs[y_str].copy()
            kw['label'], kw['ls'] = key, '--'
            l_kwargs.append(kw)

            if len(tw_results) == 0:
                continue

            x_data[-1] = _data_from_tw(tw_results, x_str)
            y_data[-1] = _data_from_tw(tw_results, y_str)


    return x_data, y_data, l_kwargs


def _data_from_lw(linac, data_str):
    """Get the data calculated by LightWin."""
    data = linac.get(data_str, to_deg=True)
    return data


def _data_from_tw(d_tw, data_str, warn_missing=True):
    """Get the data calculated by TraceWin, already loaded."""
    # Data recomputed from TW simulation
    if data_str in d_tw.keys():
        return d_tw[data_str]

    # Not implemented
    if data_str not in d_lw_to_tw.keys():
        if warn_missing:
            helper.printc("plot._data_from_tw warning: ",
                          opt_message=f"{data_str} not found for TW.")
        return None

    # Raw data from TW simulation
    key = d_lw_to_tw[data_str]
    data = d_tw[key]

    # Handle conversion issues
    if data_str in d_scale_tw_to_lw.keys():
        return d_scale_tw_to_lw[data_str] * data

    return data


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
    """Saves the figure."""
    fig.tight_layout()
    fig.savefig(filepath)
    helper.printc("plot._savefig info: ",
                  opt_message=f"Fig. saved in {filepath}")


# =============================================================================
# Basic helpers
# =============================================================================
def create_fig_if_not_exist(fignum, axnum, sharex=False, **kwargs):
    printc("plot.create_fig_if_not_exist warning: ", opt_message='deprecated.')
    kwargs['num'] = fignum
    return create_fig_if_not_exists(axnum, sharex, **kwargs)


def create_fig_if_not_exists(axnum, sharex=False, num=1):
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
    n_axes = len(axnum)
    axlist = []

    if plt.fignum_exists(num):
        fig = plt.figure(num)
        for i in range(n_axes):
            axlist.append(fig.axes[i])
        return fig, axlist

    fig = plt.figure(num)
    axlist.append(fig.add_subplot(axnum[0]))

    d_sharex = {True: axlist[0], False: None}

    for i in axnum[1:]:
        axlist.append(fig.add_subplot(i, sharex=d_sharex[sharex]))

    return fig, axlist


# TODO still used?
def clean_fig(fignumlist):
    """Clean axis of Figs in fignumlist."""
    for fignum in fignumlist:
        fig = plt.figure(fignum)
        for axx in fig.get_axes():
            axx.cla()


# TODO still used?
def empty_fig(fignum):
    """Return True if at least one axis of Fig(fignum) has no line."""
    out = False
    if plt.fignum_exists(fignum):
        fig = plt.figure(fignum)
        axlist = fig.get_axes()
        for axx in axlist:
            if axx.lines == []:
                out = True
    return out


def _autoscale_based_on(axx, lines):
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


def compare_with_tracewin(linac, x_str='z_abs', **kwargs):
    """
    Plot data calculated by TraceWin and LightWin.

    Parameters
    ----------
    linac : Accelerator
        Linac under study.
    x_str : str, opt
        To designate what the x axis should be.
    l_y_str : list, opt
        List of str to designate what y axis should be.
    filepath_ref : str, opt
        Path to the results of the TW project for error plots.
    linac_ref : Accelerator
        Reference linac for error plots.
    fignum : int, opt
        Num of fig.
    reference : str, opt
        To tell what is the reference in the error plots.
    plot_section : bool, opt
        To separate different linac sections.
    replot_lw : bool, opt
        To replot Working and Broken data every time (deactivate if you want to
        study different retuning settings on a fixed error.)
    """
    for key, val in BASE_DICT.items():
        if key not in kwargs:
            kwargs[key] = val

    if kwargs["filepath_ref"] is None:
        __f = os.path.join(linac.get('orig_dat_folder'),
                           'results/energy_ref.txt')
        assert os.path.exists(__f), f"""
        You need to run a TW reference simulation, go to Data > Save table to
        file. Default filename is {__f}.
        """
        kwargs["filepath_ref"] = __f

    # Prep some data common to all plots
    x_dat = linac.get(x_str, to_deg=True)
    elts_indexes = linac.get('s_out')

    # Prep figure and axes
    n_plots = len(kwargs["l_y_str"])
    axnum = 100 * n_plots + 11
    axnum = range(axnum, axnum + n_plots)
    _, axx = create_fig_if_not_exist(kwargs["fignum"], axnum, sharex=True)

    # Get data and kwargs
    for i, y_str in enumerate(kwargs["l_y_str"]):
        if kwargs["plot_section"]:
            plot_section(linac, axx[i], x_axis=x_str)

        if y_str == 'struct':
            plot_structure(linac, axx[i], x_axis=x_str)
            continue

        l_y_dat = []
        l_kwargs = []

        # Plot error data?
        plot_error = y_str[:3] == 'err'

        # Plot TW data?
        label_tw = 'TW'
        plot_tw = not plot_error and y_str in tw.d_tw_data_table \
            and label_tw not in axx[i].get_legend_handles_labels()[1]

        # Replot Working and Broken every time?
        label_lw = 'LW ' + linac.name
        plot_lw = not plot_error \
            and (kwargs["replot_lw"]
                 or linac.name not in ['Working', 'Broken']
                 or label_lw not in axx[i].get_legend_handles_labels()[1])

        if plot_error:
            diff = y_str[4:]
            l_y_dat.append(_err(linac, kwargs["l_y_str"][i - 1], diff=diff,
                                **kwargs))
            l_kwargs.append(
                d_plot_kwargs[kwargs["l_y_str"][i - 1]] | {
                    'label': linac.name + 'err. w/ TW'}
            )

        if plot_tw:
            l_y_dat.append(tw.load_tw_results(kwargs["filepath_ref"], y_str))
            l_kwargs.append(
                d_plot_kwargs[y_str] | {'label': label_tw, 'c': 'k', 'lw': 2.,
                                        'ls': '--'}
            )

        if plot_lw:
            # LightWin
            l_y_dat.append(linac.get(y_str, to_deg=True))
            l_kwargs.append(
                d_plot_kwargs[y_str] | {'label': label_lw}
            )

        for y_dat, other_kwargs in zip(l_y_dat, l_kwargs):
            # Downsample x or y if necessary
            x_plot, y_plot = helper.reformat(x_dat, y_dat, elts_indexes)

            axx[i].plot(x_plot, y_plot, **other_kwargs)

        axx[i].set_ylabel(d_markdown[y_str])
        axx[i].grid(True)

    axx[0].legend()
    axx[-1].set_xlabel(d_markdown[x_str])


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
        'elt': lambda elt, i: [
            {'x0': i, 'width': 1},
            [0, i]
        ]
    }

    for i, elt in enumerate(linac.elts):
        kwargs = d_x_axis[x_axis](elt, i)[0]
        ax.add_patch(d_elem_plot[elt.get('nature', to_numpy=False)](
            elt,**kwargs))

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
        'z_abs': lambda elt: linac.synch.pos['z_abs'][elt.idx['s_out']],
        'elt': lambda elt: elt.idx['element'] + 1,
    }
    x_ax = [0]
    for i, section in enumerate(linac.elements['l_sections']):
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


def plot_fit_progress(hist_f, l_label):
    """Plot the evolution of the objective functions w/ each iteration."""
    _, axx = create_fig_if_not_exist(32, [111])
    axx = axx[0]

    # Number of objectives, number of evaluations
    n_f = len(l_label)
    n_iter = len(hist_f)
    iteration = np.linspace(0, n_iter - 1, n_iter)

    __f = np.empty([n_f, n_iter])
    for i in range(n_iter):
        __f[:, i] = np.abs(hist_f[i] / hist_f[0])

    for j, label in enumerate(l_label):
        axx.plot(iteration, __f[j], label=label)

    axx.grid(True)
    axx.legend()
    axx.set_xlabel("Iteration #")
    axx.set_ylabel("Relative variation of error")
    axx.set_yscale('log')
