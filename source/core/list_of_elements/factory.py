#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds two functions to create :class:`.ListOfElements`.

Their main goal is to initialize it with the proper input synchronous particle
and beam properties.
The first one, :func:`new_list_of_elements`, is called within the
:class:`.Accelerator` class and generate a full :class:`.ListOfElements` from
scratch.
The second one, :func:`subset_of_pre_existing_list_of_elements`, is called
within the :class:`.Fault` class and generates a :class:`.ListOfElements` that
contains only a fraction of the linac.

.. todo::
    Also handle `.dst` file in `subset_of_pre_existing_list_of_elements`.

.. todo::
    Maybe it will be necessary to handle cases where the synch particle is not
    perfectly on the axis?

"""
import os
import logging
from typing import Any

import numpy as np

from core.instruction import Instruction
from core.instructions_factory import InstructionsFactory
from core.elements.element import Element
from core.commands.command import Command
from core.particle import ParticleInitialState

from core.beam_parameters.factory import InitialBeamParametersFactory

from core.list_of_elements.list_of_elements import ListOfElements
from core.beam_parameters.initial_beam_parameters import InitialBeamParameters
import tracewin_utils.load
from tracewin_utils.dat_files import (
    create_structure,
    dat_filecontent_from_smaller_list_of_elements,
)
from tracewin_utils.dat_files import save_dat_filecontent_to_dat

from beam_calculation.output import SimulationOutput
import config_manager as con


# =============================================================================
# New (whole) list of elements, called from `Accelerator`
# =============================================================================
def new_list_of_elements(
        dat_filepath: str,
        accelerator_path: str,
        input_beam_factory: InitialBeamParametersFactory,
        instruction_factory: InstructionsFactory | None = None,
        **kwargs: Any,
) -> ListOfElements:
    """
    Create a new :class:`.ListOfElements`, encompassing a full linac.

    Factory function called from the :class:`.Accelerator` object.

    Parameters
    ----------
    dat_filepath : str
        Path to the ``.dat`` file (TraceWin).
    accelerator_path : str
        Where results should be stored.

    Returns
    -------
    list_of_elements : ListOfElements
        Contains all the :class:`.Elements` of the linac, as well as the proper
        particle and beam properties at its entry.

    """
    dat_filepath = os.path.abspath(dat_filepath)
    logging.info("First initialisation of ListOfElements, ecompassing all "
                 f"linac. Created with {dat_filepath = }")

    files = {
        'dat_filepath': dat_filepath,
        'dat_content': tracewin_utils.load.dat_file(dat_filepath),
        'out_path': accelerator_path,
        'elts_n_cmds': list[Instruction],
    }

    if instruction_factory is None:
        instructions = _dat_filepath_to_plain_list_of_elements(files)
    else:
        dat_filecontent = files['dat_content']
        instructions = instruction_factory.run(dat_filecontent)
    elts = [elt for elt in instructions if isinstance(elt, Element)]

    files['elts_n_cmds'] = instructions

    input_particle = _new_input_particle(**kwargs)
    input_beam: InitialBeamParameters
    input_beam = input_beam_factory.factory_new(sigma_in=kwargs['sigma_in'],
                                                w_kin=kwargs['w_kin'])

    tm_cumul_in = np.eye(6)
    list_of_elements = ListOfElements(
        elts=elts,
        input_particle=input_particle,
        input_beam=input_beam,
        tm_cumul_in=tm_cumul_in,
        files=files,
        first_init=True)
    return list_of_elements


def _new_input_particle(w_kin: float,
                        phi_abs: float,
                        z_in: float,
                        **kwargs: np.ndarray) -> ParticleInitialState:
    """Create a :class:`.ParticleInitialState` for a full list of elements."""
    input_particle = ParticleInitialState(w_kin=w_kin,
                                          phi_abs=phi_abs,
                                          z_in=z_in,
                                          synchronous=True,)
    return input_particle


# future deprecation
def _dat_filepath_to_plain_list_of_elements(
        files: dict[str, Any]
) -> list[Instruction]:
    """
    Convert the content of the ``.dat`` file to a list of elements/commands.

    Parameters
    ----------
    files : dict[str, str | list[list[str]] | None]
        Must contain filepath to ``.dat`` and content of this file as returned
        by :func:`tracewin_utils.load.dat_file`.

    Returns
    -------
    instructions : list[Instruction]
        List containing all objects from the ``.dat`` file.

    """
    instructions = create_structure(freq_bunch=con.F_BUNCH_MHZ, **files)
    return instructions


# =============================================================================
# Partial list of elements, called from `Fault`
# =============================================================================
def subset_of_pre_existing_list_of_elements(
    elts: list[Element],
    simulation_output: SimulationOutput,
    files_from_full_list_of_elements: dict[str, str | list[list[str]]],
    input_beam_factory: InitialBeamParametersFactory,
) -> ListOfElements:
    """
    Create a :class:`.ListOfElements` which is a subset of a previous one.

    Factory function used during the fitting process, called by a
    :class:`.Fault` object. During this optimisation process, we compute the
    propagation of the beam only on the smallest possible subset of the linac.

    It creates the proper :class:`.ParticleInitialState` and
    :class:`.BeamParameters` objects. In contrary to
    :func:`new_list_of_elements`, the :class:`.BeamParameters` must contain
    information on the transverse plane if beam propagation is performed with
    :class:`.TraceWin`.

    Parameters
    ----------
    elts : list[Element]
        A plain list containing the elements objects that the object should
        contain.
    simulation_output : SimulationOutput
        Holds the results of the pre-existing list of elements.

    Returns
    -------
    list_of_elements : ListOfElements
        Contains all the elements that will be recomputed during the
        optimisation, as well as the proper particle and beam properties at its
        entry.

    """
    logging.info(f"Initalisation of ListOfElements from already initialized "
                 f"elements: {elts[0]} to {elts[-1]}.")

    input_elt, input_pos = _get_initial_element(elts, simulation_output)
    get_kw = {'elt': input_elt,
              'pos': input_pos,
              'to_numpy': False,
              }
    input_particle = _subset_input_particle(simulation_output, **get_kw)
    input_beam = input_beam_factory.factory_subset(simulation_output, get_kw)

    logging.warning("The phase_info dict, which handles how and if cavities "
                    "are rephased in the .dat file, is hard-coded. It should"
                    " take config_manager.PHI_ABS_FLAG as input.")

    files = _subset_files_dictionary(
        elts,
        files_from_full_list_of_elements,
    )

    transfer_matrix = simulation_output.transfer_matrix
    assert transfer_matrix is not None
    tm_cumul_in = transfer_matrix.cumulated[0]

    list_of_elements = ListOfElements(
        elts=elts,
        input_particle=input_particle,
        input_beam=input_beam,
        tm_cumul_in=tm_cumul_in,
        files=files,
        first_init=False)

    return list_of_elements


def _subset_files_dictionary(
    elts: list[Element],
    files_from_full_list_of_elements: dict[str, Any],
    tmp_folder: str = 'tmp',
    tmp_dat: str = 'tmp.dat',
) -> dict[str, str | list[list[str]]]:
    """Set the new ``.dat`` file containing only elements of ``elts``."""
    dirname = files_from_full_list_of_elements['out_path']
    assert isinstance(dirname, str)
    dat_filepath = os.path.join(dirname, tmp_folder, tmp_dat)

    original_instructions = files_from_full_list_of_elements['elts_n_cmds']
    assert isinstance(original_instructions, list)
    assert all(isinstance(elt, (Element, Command))
               for elt in original_instructions)
    dat_content, instructions = dat_filecontent_from_smaller_list_of_elements(
        files_from_full_list_of_elements['elts_n_cmds'],
        elts,
    )

    files = {'dat_filepath': dat_filepath,
             'dat_content': dat_content,
             'elts_n_cmds': instructions,
             'out_path': os.path.dirname(dat_filepath)}

    os.mkdir(os.path.join(dirname, tmp_folder))
    save_dat_filecontent_to_dat(dat_content, dat_filepath)
    return files


def _delta_phi_for_tracewin(phi_at_entry_of_compensation_zone: float) -> float:
    """
    Give new absolute phases for :class:`.TraceWin`.

    In TraceWin, the absolute phase at the entrance of the compensation zone
    is 0, while it is not in the rest of the code. Hence we must rephase the
    cavities in the subset.

    """
    phi_at_linac_entry = 0.
    delta_phi_bunch = phi_at_entry_of_compensation_zone - phi_at_linac_entry
    return delta_phi_bunch


def _get_initial_element(elts: list[Element],
                         simulation_output: SimulationOutput
                         ) -> tuple[Element | str, str]:
    """Set the element from which we should take energy, phase, etc."""
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
                           **kwargs: Any
                           ) -> ParticleInitialState:
    """Create `ParticleInitialState` for an incomplete list of `Element`s."""
    w_kin, phi_abs, z_abs = simulation_output.get('w_kin', 'phi_abs', 'z_abs',
                                                  **kwargs)
    input_particle = ParticleInitialState(w_kin, phi_abs, z_abs,
                                          synchronous=True)
    return input_particle
