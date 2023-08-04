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

"""
import os.path
import logging

import numpy as np

from core.elements import _Element
from core.particle import ParticleInitialState
from core.beam_parameters import BeamParameters
from core.list_of_elements import ListOfElements

import tracewin_utils.load
from tracewin_utils.dat_files import (create_structure,
                                      set_field_map_files_paths)
import tracewin_utils.electric_fields

from beam_calculation.output import SimulationOutput


def new_list_of_elements(dat_filepath: str,
                         input_particle: ParticleInitialState,
                         input_beam: BeamParameters,
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

    Returns
    -------
    list_of_elements : ListOfElements
        Contains all the `_Elements` of the linac, as well as the proper
        particle and beam properties at its entry.

    """
    logging.info("First initialisation of ListOfElements, ecompassing "
                 + "all linac. Also removing Lattice and Freq "
                 + "commands, setting Lattice/Section structures, "
                 + "_Elements names.")

    dat_filepath = os.path.abspath(dat_filepath)
    dat_information = {
        'path': dat_filepath,
        'content': tracewin_utils.load.dat_file(dat_filepath),
        'field_map_folder': None,
    }

    elts, field_map_folder = _dat_filepath_to_plain_list_of_elements(
        dat_information)
    dat_information['field_map_folder'] = field_map_folder

    list_of_elements = ListOfElements(elts=elts,
                                      input_particle=input_particle,
                                      input_beam=input_beam,
                                      dat_information=dat_information,
                                      first_init=True)
    tracewin_utils.electric_fields.set_all_electric_field_maps(
        field_map_folder, list_of_elements.by_section_and_lattice)
    return list_of_elements


def _dat_filepath_to_plain_list_of_elements(
        dat_information: dict[str, str | list[list[str]] | None],
) -> list[_Element]:
    """
    Convert the content of the `.dat` file to a plain list of `_Element`s.

    Parameters
    ----------
    dat_information : dict[str, str | list[list[str]] | None]
        Must contain filepath to `.dat` and content of this file as returned
        by tracewin_utils.load.dat_file.

    Returns
    -------
    elts : list[_Element]
        Plain list of _Element (not yet a `ListOfElements` object).

    """
    elts = create_structure(dat_information['content'])
    elts, field_map_folder = set_field_map_files_paths(
        elts,
        default_field_map_folder=os.path.dirname(dat_information['path'])
    )
    return elts, field_map_folder


def subset_of_pre_existing_list_of_elements(
    elts: list[_Element],
    simulation_output: SimulationOutput,
) -> ListOfElements:
    """
    Create a `ListOfElements` which is a subset of a previous one.

    Factory function used during the fitting process from a `Fault` object.
    During this optimisation process, we compute the propagation of the beam
    only in the smallest possible subset of the linac.

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
    logging.warning("Check how TraceWin will deal with incomplete Lattices.")

    input_elt, input_pos = elts[0], 'in'
    try:
        _ = simulation_output.get('w_kin', elt=input_elt)
    except AttributeError:
        logging.warning("First element of new list of elements is not in the"
                        " given SimulationOutput. I will consider that the "
                        "last element of the SimulationOutput if the first of"
                        "of the new ListOfElements.")
        input_elt, input_pos = 'last', 'out'

    kwargs = {'elt': input_elt,
              'pos': input_pos,
              'to_numpy': False}

    w_kin, phi_abs = simulation_output.get('w_kin', 'phi_abs', **kwargs)
    input_particle = ParticleInitialState(w_kin, phi_abs, synchronous=True)

    input_beam: BeamParameters = simulation_output.beam_parameters.subset(
        *('x', 'y', 'z', 'zdelta'), elt=input_elt, pos=input_pos)
    if np.any(np.isnan(input_beam.zdelta.tm_cumul)):
        logging.error("Previous transfer matrix was not calculated.")

    list_of_elements = ListOfElements(elts=elts,
                                      input_particle=input_particle,
                                      input_beam=input_beam,
                                      first_init=False)

    return list_of_elements