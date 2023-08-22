#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:37:30 2023.

@author: placais

This module holds two functions to create `ListOfElements` with the proper
input synchronous particle and beam properties.

The first one, `new_list_of_elements`, is called within the `Accelerator` class
and generate a full `ListOfElements` from scratch.

The second one, `subset_of_pre_existing_list_of_elements`, is called within the
`Fault` class and generates a `ListOfElements` that contains only a fraction of
the linac.

TODO : also handle `.dst` file in `subset_of_pre_existing_list_of_elements`.

Maybe it will be necessary to handle cases where the synch particle is not
perfectly on the axis?

"""
import os
import logging

import numpy as np

from core.elements import _Element
from core.particle import ParticleInitialState
from core.beam_parameters import BeamParameters
from core.list_of_elements import ListOfElements

import tracewin_utils.load
from tracewin_utils.dat_files import (
    create_structure,
    set_field_map_files_paths,
    dat_filecontent_from_smaller_list_of_elements,
)
import tracewin_utils.electric_fields
from tracewin_utils.dat_files import save_dat_filecontent_to_dat

from beam_calculation.output import SimulationOutput


# =============================================================================
# New (whole) list of elements, called from `Accelerator`
# =============================================================================
def new_list_of_elements(dat_filepath: str,
                         accelerator_path: str,
                         **kwargs: float | np.ndarray,
                         ) -> ListOfElements:
    """
    Create a new `ListOfElements`.

    Factory function called from the `Accelerator` object. It encompasses the
    full linac.

    Parameters
    ----------
    dat_filepath : str
        Path to the `.dat` file (TraceWin).
    input_particle : ParticleInitialState
        An object to hold initial energy and phase of the particle.
    input_beam : BeamParameters
        Holds the initial properties of the beam. It is pretty light, as
        Envelope1D does not need a lot of beam properties, and as the ones
        required by TraceWin are already defined in the `.ini` file.
    accelerator_path : str
        Where results should be stored.

    Returns
    -------
    list_of_elements : ListOfElements
        Contains all the `_Elements` of the linac, as well as the proper
        particle and beam properties at its entry.

    """
    dat_filepath = os.path.abspath(dat_filepath)
    logging.info("First initialisation of ListOfElements, ecompassing all "
                 "linac. Also removing Lattice and Freq commands, setting "
                 "Lattice/Section structures, _Elements names. "
                 f"Created with dat_filepath = {dat_filepath}")

    files = {
        'dat_filepath': dat_filepath,
        'dat_content': tracewin_utils.load.dat_file(dat_filepath),
        'field_map_folder': None,
        'out_path': accelerator_path
    }

    elts, field_map_folder = _dat_filepath_to_plain_list_of_elements(files)
    files['field_map_folder'] = field_map_folder

    input_particle = _new_input_particle(**kwargs)
    input_beam = _new_beam_parameters(**kwargs)
    list_of_elements = ListOfElements(elts=elts,
                                      input_particle=input_particle,
                                      input_beam=input_beam,
                                      files=files,
                                      first_init=True)
    tracewin_utils.electric_fields.set_all_electric_field_maps(
        field_map_folder, list_of_elements.by_section_and_lattice)
    return list_of_elements


def _new_input_particle(w_kin: float,
                        phi_abs: float,
                        z_in: float,
                        **kwargs: np.ndarray) -> ParticleInitialState:
    """Create a `ParticleInitialState` for a full `ListOfElements`."""
    input_particle = ParticleInitialState(w_kin=w_kin,
                                          phi_abs=phi_abs,
                                          z_in=z_in,
                                          synchronous=True)
    return input_particle


def _new_beam_parameters(sigma_in_zdelta: np.ndarray,
                         **kwargs: float) -> BeamParameters:
    """
    Generate a `BeamParameters` objet for the linac entry.


    Called from `Accelerator.__init__`. The returned `input_beam` is the object
    required by `new_list_of_elements.__init__`.

    """
    input_beam = BeamParameters()
    input_beam.create_phase_spaces('zdelta')
    input_beam.zdelta.tm_cumul = np.eye(2)
    input_beam.zdelta.sigma_in = sigma_in_zdelta
    return input_beam


def _dat_filepath_to_plain_list_of_elements(
        files: dict[str, str | list[list[str]] | None],
) -> list[_Element]:
    """
    Convert the content of the `.dat` file to a plain list of `_Element`s.

    Parameters
    ----------
    files : dict[str, str | list[list[str]] | None]
        Must contain filepath to `.dat` and content of this file as returned
        by tracewin_utils.load.dat_file.

    Returns
    -------
    elts : list[_Element]
        Plain list of _Element (not yet a `ListOfElements` object).

    """
    elts = create_structure(files['dat_content'])
    elts, field_map_folder = set_field_map_files_paths(
        elts,
        default_field_map_folder=os.path.dirname(files['dat_filepath'])
    )
    return elts, field_map_folder


# =============================================================================
# Partial list of elements, called from `Fault`
# =============================================================================
def subset_of_pre_existing_list_of_elements(
    elts: list[_Element],
    simulation_output: SimulationOutput,
    files_from_full_list_of_elements: dict[str, str | list[list[str]]],
) -> ListOfElements:
    """
    Create a `ListOfElements` which is a subset of a previous one.

    Factory function used during the fitting process from a `Fault` object.
    During this optimisation process, we compute the propagation of the beam
    only on the smallest possible subset of the linac.

    It creates the proper `input_particle` and `input_beam` objects. In
    contrary to `new_list_of_elements` function, `input_beam` must contain
    information on the transverse plane if beam propagation is performed with
    TraceWin.

    Parameters
    ----------
    elts : list[_Element]
        A plain list containing the `_Element` objects that the object should
        contain.
    simulation_output : SimulationOutput
        Holds the results of the pre-existing `ListOfElements`.

    Returns
    -------
    list_of_elements : ListOfElements
        Contains all the `_Elements` that will be recomputed during the
        optimisation, as well as the proper particle and beam properties at its
        entry.

    """
    logging.info(f"Initalisation of ListOfElements from already initialized "
                 f"elements: {elts[0]} to {elts[-1]}.")

    input_elt, input_pos = _get_initial_element(elts, simulation_output)
    kwargs = {'elt': input_elt,
              'pos': input_pos,
              'to_numpy': False,
              'phase_space': None}
    input_particle = _subset_input_particle(simulation_output, **kwargs)
    input_beam: BeamParameters = _subset_beam_parameters(simulation_output,
                                                         **kwargs)

    logging.warning("The `phase_info` dict, which handles how and if cavities "
                    "are rephased in the `.dat` file, is hard-coded. It should"
                    " take config_manager.PHI_ABS_FLAG as input.")

    files = _subset_files_dictionary(
        elts,
        files_from_full_list_of_elements,
    )

    list_of_elements = ListOfElements(elts=elts,
                                      input_particle=input_particle,
                                      input_beam=input_beam,
                                      files=files,
                                      first_init=False)

    return list_of_elements


def _subset_files_dictionary(
    elts: list[_Element],
    files_from_full_list_of_elements: dict[str, str | list[list[str]]],
    tmp_folder: str = 'tmp',
    tmp_dat: str = 'tmp.dat',
) -> dict[str, str | list[list[str]]]:
    """Set the new `.dat` file containing only `_Element` of `elts`."""
    dirname = files_from_full_list_of_elements['out_path']
    dat_filepath = os.path.join(dirname, tmp_folder, tmp_dat)

    dat_content = dat_filecontent_from_smaller_list_of_elements(
        files_from_full_list_of_elements['dat_content'],
        elts,
    )

    field_map_folder = files_from_full_list_of_elements['field_map_folder']

    files = {'dat_filepath': dat_filepath,
             'dat_content': dat_content,
             'field_map_folder': field_map_folder,
             'out_path': os.path.dirname(dat_filepath)}

    os.mkdir(os.path.join(dirname, tmp_folder))
    save_dat_filecontent_to_dat(dat_content, dat_filepath)
    return files


def _delta_phi_for_tracewin(phi_at_entry_of_compensation_zone: float) -> float:
    """
    Give new absolute phases for `TraceWin`.

    In `TraceWin`, the absolute phase at the entrance of the compensation zone
    is 0, while it is not in the rest of the code. Hence we must rephase the
    cavities in the subset.

    """
    phi_at_linac_entry = 0.
    delta_phi_bunch = phi_at_entry_of_compensation_zone - phi_at_linac_entry
    return delta_phi_bunch


def _get_initial_element(elts: list[_Element],
                         simulation_output: SimulationOutput
                         ) -> tuple[_Element | str, str]:
    """Set the `_Element` from which we should take energy, phase, etc."""
    input_elt, input_pos = elts[0], 'in'
    try:
        _ = simulation_output.get('w_kin', elt=input_elt)
    except AttributeError:
        logging.warning("First element of new list of elements is not in the"
                        " given SimulationOutput. I will consider that the "
                        "last element of the SimulationOutput if the first of"
                        "of the new ListOfElements.")
        input_elt, input_pos = 'last', 'out'
    return input_elt, input_pos


def _subset_input_particle(simulation_output: SimulationOutput,
                           **kwargs: _Element | str | bool | None
                           ) -> ParticleInitialState:
    """Create `ParticleInitialState` for an incomplete list of `_Element`s."""
    w_kin, phi_abs, z_abs = simulation_output.get('w_kin', 'phi_abs', 'z_abs',
                                                  **kwargs)
    input_particle = ParticleInitialState(w_kin, phi_abs, z_abs,
                                          synchronous=True)
    return input_particle


def _subset_beam_parameters(simulation_output: SimulationOutput,
                            **kwargs: _Element | str | bool | None
                            ) -> BeamParameters:
    """Create `BeamParameters` for an incomplete list of `_Element`s."""
    z_abs, gamma_kin, beta_kin = simulation_output.get(
        *('z_abs', 'gamma', 'beta'), **kwargs)
    input_beam = BeamParameters(z_abs=z_abs,
                                gamma_kin=gamma_kin,
                                beta_kin=beta_kin)

    phase_spaces = _required_phase_spaces(simulation_output.is_3d,
                                          simulation_output.is_multiparticle)
    quantities = _required_quantities()
    beam_param_kwargs = _get_quantities_from_phase_spaces(
        phase_spaces,
        quantities,
        simulation_output.beam_parameters,
        **kwargs)
    input_beam.create_phase_spaces(*phase_spaces, **beam_param_kwargs)

    return input_beam


def _required_phase_spaces(is_3d: bool, is_multiparticle: bool) -> tuple[str]:
    """Give necessary phase spaces according to `SimulationOutput` flags."""
    phase_spaces = ('z', 'zdelta')
    if is_3d:
        phase_spaces = ('x', 'y', 't', 'z', 'zdelta')
    if is_multiparticle:
        phase_spaces = ('x', 'y', 't', 'z', 'zdelta', 'x99', 'y99', 'wphi99')
    return phase_spaces


def _required_quantities() -> tuple[str]:
    """Give quantities to set."""
    return ('eps', 'alpha', 'beta', 'tm_cumul')


def _get_quantities_from_phase_spaces(phase_spaces: tuple[str],
                                      quantities: tuple[str],
                                      full_beam_parameters: BeamParameters,
                                      **kwargs: _Element | str | bool | None
                                      ) -> dict[str, dict[str, float | None]]:
    """Get desired quantities at proper place in every phase space."""
    beam_param_kwargs = {phase_space_name: None
                         for phase_space_name in phase_spaces}

    for phase_space_name in phase_spaces:
        phase_space = getattr(full_beam_parameters, phase_space_name)

        beam_param_kwargs[phase_space_name] = {
            quantity: phase_space.get(quantity, **kwargs)
            for quantity in quantities}
    return beam_param_kwargs
