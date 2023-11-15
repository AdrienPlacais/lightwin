#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is specific to hold mismatches.

It has its own module as this quantity is pretty specific.

"""
import logging

import numpy as np

from optimisation.objective.objective import Objective

from core.elements.element import Element
from core.beam_parameters.helper import mismatch_from_arrays

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput


class MinimizeMismatch(Objective):
    """Minimize a mismatch factor."""

    def __init__(self,
                 name: str,
                 weight: float,
                 get_key: str,
                 get_kwargs: dict[str, Element | str | bool],
                 reference: SimulationOutput,
                 descriptor: str | None = None,
                 ) -> None:
        """
        Set complementary :func:`get` flags, reference value.

        Parameters
        ----------
        get_key : str
            Must contain 'twiss' plus the name of a phase-space, or simply
            'twiss' and the phase-space is defined in ``get_kwargs``.
        get_kwargs : dict[str, Element | str | bool]
            Keyword arguments for the :func:`get` method. We do not check its
            validity, but in general you will want to define the keys ``elt``
            and ``pos``. You should also define the ``phase_space`` key if it
            is not defined in the ``get_key``.
        reference : SimulationOutput
            The reference simulation output from which the Twiss parameters
            will be taken.

        """
        if 'twiss' not in get_key:
            logging.warning("The get_key should contain 'twiss'. Taking "
                            "'twiss' and setting phase space to zdelta.")
            get_key = 'twiss'
            get_kwargs['phase_space'] = 'zdelta'
        self.get_key = get_key
        self.get_kwargs = get_kwargs
        super().__init__(name,
                         weight,
                         descriptor=descriptor,
                         ideal_value=0.)
        self._twiss_ref = self._twiss_getter(reference)

    def _base_str(self) -> str:
        """Return a base text for output."""
        message = f"{self.name:>23}"

        elt = str(self.get_kwargs.get('elt', 'NA'))
        message += f" @elt {elt:>5}"

        pos = str(self.get_kwargs.get('pos', 'NA'))
        message += f" ({pos:>3}) | {self.weight:>5} | "
        return message

    def __str__(self) -> str:
        """Give objective information value."""
        return self._base_str() + f"{self.ideal_value:>10}"

    def current_value(self, simulation_output: SimulationOutput) -> str:
        res = self._compute_residues(simulation_output)
        message = self._base_str() + f"{'NA':>10} | {res:>10}"
        return message

    def _twiss_getter(self, simulation_output: SimulationOutput
                      ) -> np.ndarray:
        """Get desired value using :func:`SimulationOutput.get` method."""
        return simulation_output.get(self.get_key, **self.get_kwargs)

    def evaluate(self, simulation_output: SimulationOutput) -> float:
        twiss_fix = self._twiss_getter(simulation_output)
        return self._compute_residues(twiss_fix)

    def _compute_residues(self, twiss_fix: np.ndarray) -> float:
        """Compute the residues."""
        res = mismatch_from_arrays(self._twiss_ref, twiss_fix)[0]
        return self.weight * res
