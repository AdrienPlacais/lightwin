#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 14:26:45 2021

@author: placais
"""

import numpy as np
from palettable.colorbrewer.qualitative import Set1_9
import helper
import debug
from constants import q_adim, m_MeV, c
import solver
import particle


def transport_beam(accelerator):
    """Compute transfer matrices by the transport method."""
    omega0_bunch = accelerator.list_of_elements[0].omega0_bunch
    synch = particle.Particle(0., 16.6, omega0_bunch)
    rand_1 = particle.Particle(-1e-2, 16.5, omega0_bunch)
    rand_2 = particle.Particle(1e-2, 16.7, omega0_bunch)
    all_part = [synch, rand_1, rand_2]
    rand_part = [rand_1, rand_2]

    for part in rand_part:
        part.compute_phase_space(synch)

    phase_space_1 = phase_space_dict_to_matrix(rand_1, rand_2)
    transfer_matrix = np.expand_dims(np.eye(2), 0)

    for elt in accelerator.list_of_elements:
        print(elt.name)
        n_steps = elt.solver_transf_mat.n_steps
        z_step = elt.solver_transf_mat.d_z

        def du_dz(z, u):
            """
            Compute derivative of system energy/time.

            Parameters
            ----------
            u: np.array(2)
                First component is e_mev(i).
                Second component is phi(i).

            Return
            ------
            v: np.array(2)
                First component is (e_mev(i+1) - e_mev(i)) / dz.
                Second component is (phi(i+1) - phi(i)) / dz.
            """
            v0 = q_adim * elt.acc_field.e_func(z, u[1])

            gamma_float = helper.mev_to_gamma(u[0], m_MeV)
            beta = helper.gamma_to_beta(gamma_float)

            v1 = elt.acc_field.n_cell * elt.omega0_bunch / (beta * c)

            return np.array(([v0, v1]))

        entrance_phase = synch.phi['rel']
        entrance_pos = synch.z['rel']
        for i in range(n_steps):
            # Compute energy, position and phase evolution during a time step
            for part in all_part:
                if i == 0:
                    # Set relative phase and positions to 0.
                    part.z['rel'] -= entrance_pos
                    part.phi['rel'] -= entrance_phase

                u_rk = np.array(([part.energy['e_mev'], part.phi['rel']]))
                delta_u = solver.rk4(u_rk, du_dz, part.z['rel'], z_step)

                part.set_energy(delta_u[0], delta_e=True)
                part.advance_phi(delta_u[1])       # TODO BUG FIXME
                part.advance_position(z_step)

            for part in rand_part:
                part.compute_phase_space(synch)

            phase_space_0 = phase_space_1
            phase_space_1 = phase_space_dict_to_matrix(rand_1, rand_2)

            new_transfer_matrix = compute_transfer_matrix(phase_space_0,
                                                          phase_space_1)
            print(i, '\n', new_transfer_matrix, '\n', synch.z['abs'])
            transfer_matrix = np.vstack((transfer_matrix, new_transfer_matrix))

        # Raise some arrays to element
        # elt.pos_m = np.array(synch.z['abs_array'])[-n_steps:]
        elt.energy['gamma_array'] \
            = np.array(synch.energy['gamma_array'])[-n_steps:]
        elt.energy['e_array_mev'] \
            = np.array(synch.energy['e_array_mev'])[-n_steps:]
        elt.transfer_matrix = transfer_matrix[-n_steps:]

    # debug.plot_transfer_matrices(accelerator, transfer_matrix)


def phase_space_dict_to_matrix(rand_1, rand_2):
    """Convert phase space dicts to a 2x2 matrix."""
    matrix = np.array(([rand_1.phase_space['z'],
                        rand_2.phase_space['z']],
                       [rand_1.phase_space['delta'],
                        rand_2.phase_space['delta']]))
    return matrix


def compute_transfer_matrix(phase_space_0, phase_space_1):
    """Compute transfer matrix by matrix inversion."""
    try:
        inv_phase_space_1 = np.linalg.inv(phase_space_1)

    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print('Singular matrix error:')
            print(phase_space_0, '\n', phase_space_1)
        else:
            raise
    transfer_matrix = phase_space_0 @ inv_phase_space_1
    transfer_matrix = np.expand_dims(transfer_matrix, 0)
    return transfer_matrix


def compute_envelope(accelerator):
    """
    Compute the z | dp/p array evolution accross the accelerator.

    Parameters
    ----------
    accelerator: Accelerator object
        Longitudinal transfer matrices must have been calculated.
    """
    flag_compare = True

    fignum = 32
    axnum = range(211, 213)
    fig, axlist = helper.create_fig_if_not_exist(fignum, axnum)
    axlist[0].set_ylabel(r'$\phi$ [deg]')
    axlist[1].set_ylabel(r'd$p/p$ [%]')
    axlist[1].set_xlabel(r'$z_s$ [m]')

    if flag_compare:
        # @TODO auto import of correct file
        filepath = '/home/placais/TraceWin/work_field_map/results/envelope.txt'
        # File must be generated from Envelope Plot -> Save chart -> DATA ASCII
        # file
        data = np.loadtxt(filepath, skiprows=2, usecols=(0, 19, 17, 16))
        col = 'k'
        ls = '--'
        axlist[0].plot(data[:, 0], data[:, 3], ls=ls, c=col, label='TW')
        axlist[1].plot(data[:, 0], data[:, 2], ls=ls, c=col)

    transfer_matrix = accelerator.transfer_matrix_cumul
    pos = accelerator.get_from_elements('pos_m')

    n = transfer_matrix.shape[0]
    gamma = accelerator.get_from_elements('gamma_array')
    beta = helper.gamma_to_beta(gamma)

    # Assumption that the frequency won't change
    lambda_rf = accelerator.list_of_elements[5].acc_field.lambda_rf

    # Vectors of transverse dynamics
    transv = np.full((4, 2, n), np.NaN)

    # Initial delta phase of 11.5deg
    if flag_compare:
        delta_phi_0_deg = data[0, 1]
        delta_p_0 = data[0, 2]
        delta_z_0 = data[0, 3]

    else:
        delta_phi_0_deg = 11.55
        delta_p_0 = 0.005
        delta_z_0 = -beta[0] * lambda_rf * delta_phi_0_deg / 360.

    transv[0, :, 0] = np.array(([+delta_z_0, +delta_p_0]))
    transv[1, :, 0] = np.array(([+delta_z_0, -delta_p_0]))
    transv[2, :, 0] = np.array(([-delta_z_0, +delta_p_0]))
    transv[3, :, 0] = np.array(([-delta_z_0, -delta_p_0]))

    # Transport beam
    for i in range(n - 1):
        for j in range(4):
            transv[j, :, i + 1] = transfer_matrix[i + 1, :, :] \
                @ transv[j, :, 0]

    # Convert delta_z in phase
    col = (Set1_9.colors[0][0] / 256., Set1_9.colors[0][1] / 256.,
           Set1_9.colors[0][1] / 256.)
    ls = '-'
    label = ['LightWin', '']
    for i in range(2):
        axlist[i].plot(pos, np.max(transv[:, i, :], axis=0),
                       ls=ls, c=col, label=label[i])
        axlist[i].plot(pos, np.min(transv[:, i, :], axis=0),
                       ls=ls, c=col)
        axlist[i].grid(True)
    axlist[0].legend()
