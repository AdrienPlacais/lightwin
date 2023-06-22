#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:56:20 2023.

@author: placais

A class to uniformly store the outputs from the different simulation tools:
    envelope1d
    tracewin_envelope
    tracewin_multiparticle
"""
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from core.particle import ParticleFullTrajectory
from core.elements import _Element
from core.list_of_elements import ListOfElements
from core.emittance import beam_parameters_all, mismatch_factor
from util.helper import recursive_items, recursive_getter


@dataclass
class SimulationOutput:
    """Stores the information that is needed for a fit."""

    z_abs: np.ndarray | None = None
    synch_trajectory: ParticleFullTrajectory | None = None

    cav_params: list[dict | None] | None = None
    phi_s: list[float] | None = None
    rf_fields: list[dict] | None = None

    r_zz_elt: list[np.ndarray] | None = None
    tm_cumul: np.ndarray | None = None

    eps_zdelta: np.ndarray | None = None
    twiss_zdelta: np.ndarray | None = None
    sigma_matrix: np.ndarray | None = None
    mismatch_factor: list[float | None] | None = None

    element_to_index: Callable[[str | _Element, str | None], int | slice] \
        | None = None

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True,
            elt: _Element | None = None, pos: str | None = None,
            **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys: str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        elt : _Element | None, optional
            If provided, return the attributes only at the considered _Element.
        pos : 'in' | 'out' | None
            If you want the attribute at the entry, exit, or in the whole
            _Element.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        The attributes.

        """
        val = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

            if None not in (self.element_to_index, elt):
                idx = self.element_to_index(elt, pos)
                val[key] = val[key][idx]

        out = [np.array(val[key]) if to_numpy and not isinstance(val[key], str)
               else val[key]
               for key in keys]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def compute_complementary_data(self, elts: ListOfElements,
                                   ref_twiss_zdelta: np.ndarray | None = None
                                   ) -> None:
        """
        Compute some other indirect quantities.

        Parameters
        ----------
        elts : ListOfElements
            Must be a full ListOfElements, containing all the _Elements of the
            linac.
        ref_twiss_zdelta : np.ndarray | None, optional
            A reference array of Twiss parameters. If provided, it allows the
            calculation of the mismatch factor. The default is None.

        """
        self.z_abs = elts.get('abs_mesh', remove_first=True)
        self.synch_trajectory.compute_complementary_data()

        self.beam_param = beam_parameters_all(self.eps_zdelta,
                                              self.twiss_zdelta,
                                              self.synch_trajectory.gamma)
        mism = None
        if ref_twiss_zdelta is not None:
            self.mismatch_factor = self._compute_mismatch(ref_twiss_zdelta)
            mism = self.mismatch_factor

        if mism is not None:
            self.beam_param["mismatch_factor"] = mism

    def _compute_mismatch(self, ref_twiss_zdelta: np.ndarray) -> np.ndarray:
        """Compute the mismatch between reference and broken linac."""
        mism = mismatch_factor(ref_twiss_zdelta, self.get("twiss_zdelta"),
                               transp=True)
        return mism
