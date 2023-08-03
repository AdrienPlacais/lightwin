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
import logging

import numpy as np

from core.elements import _Element
from core.particle import ParticleInitialState
from core.beam_parameters import BeamParameters
from core.list_of_elements import ListOfElements

from beam_calculation.output import SimulationOutput


def new_list_of_elements(elts: list[_Element],
                         input_particle: ParticleInitialState,
                         input_beam: BeamParameters,
                         ) -> ListOfElements:
    """
    Create a new `ListOfElements`.

    Factory function called from the `Accelerator` object. It encompasses the
    full linac.

    Parameters
    ----------
    elts : list[_Element]
        A plain list containing all the `_Element` objects of the linac.
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
    list_of_elements = ListOfElements(elts=elts,
                                      input_particle=input_particle,
                                      input_beam=input_beam,
                                      first_init=True)
    return list_of_elements


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
