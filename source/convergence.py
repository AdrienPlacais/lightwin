#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:25:31 2022.

@author: placais
"""
import os
import numpy as np
import timeit as ti
import matplotlib.pyplot as plt
import constants
import accelerator as acc
from palettable.colorbrewer.qualitative import Set1_9
from cycler import cycler
font = {'family': 'serif',
        'size': 20}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=(cycler('color', Set1_9.mpl_colors)))
plt.rc('mathtext', fontset='cm')

FILEPATH = os.path.abspath(
    "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat")


def prepare(filepath, method, flag_cython, n_steps):
    """Prepare linac object, no timer."""
    constants.METHOD = method
    constants.FLAG_CYTHON = flag_cython
    constants.N_STEPS_PER_CELL = n_steps

    if flag_cython:
        constants.METHOD += '_c'
    else:
        constants.METHOD += '_p'
    linac = acc.Accelerator(filepath, "Working")
    return linac


def compute(linac):
    """Compute transfer matrix of linac."""
    linac.compute_transfer_matrices()

    # From TW
    ref_values = [np.deg2rad(126622.65), 601.16554,
                  np.array(([-0.43919173, 0.61552512],
                            [-0.083619398, -0.21324773]))]

    delta_phi = linac.synch.phi['abs_array'][-1] - ref_values[0]
    delta_W = linac.synch.energy['kin_array_mev'][-1] - ref_values[1]
    delta_MT = np.sum(linac.transf_mat['cumul'][-1] - ref_values[2])
    return abs(delta_phi) + abs(delta_W) + abs(delta_MT)


# Plot
fig = plt.figure(51)
axs = [fig.add_subplot(211), fig.add_subplot(212)]
axs[0].set_ylabel('Error')
axs[0].set_yscale('log')
axs[1].set_ylabel('Time [s]')
axs[1].set_xlabel('# of steps per cell')

# Steps to study
l_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_repeat = 20

dic_labels = {True: ' Cython', False: ' Python'}
dic_ls = {True: '--', False: '-'}


for ax in axs:
    ax.grid(True)

for meth in ['RK', 'leapfrog']:
    for flag in [False, True]:
        label = meth + dic_labels[flag]
        times = []
        errors = []

        for step in l_steps:
            linac = prepare(FILEPATH, meth, flag, step)
            errors.append(compute(linac))
            timer = ti.Timer(stmt="compute(linac)",
                             setup="from __main__ import compute",
                             globals={'linac': linac})
            tmp = timer.repeat(number=n_repeat)
            times.append(min(tmp))

        axs[0].plot(l_steps, errors, label=label, ls=dic_ls[flag], marker='o')
        axs[1].plot(l_steps, times, label=label, ls=dic_ls[flag], marker='o')

axs[0].legend()
