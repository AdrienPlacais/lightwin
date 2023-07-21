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
import logging
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import pandas as pd

from core.particle import ParticleFullTrajectory
from core.elements import _Element
from core.list_of_elements import ListOfElements
from core.beam_parameters import BeamParameters
from util.helper import recursive_items, recursive_getter, range_vals


@dataclass
class SimulationOutput:
    """Stores the information that is needed for a fit."""

    z_abs: np.ndarray | None = None
    synch_trajectory: ParticleFullTrajectory | None = None

    cav_params: dict[str, float | None] | None = None
    rf_fields: list[dict] | None = None

    r_zz_elt: list[np.ndarray] | None = None
    beam_parameters: BeamParameters | None = None

    element_to_index: Callable[[str | _Element, str | None], int | slice] \
        | None = None

    in_tw_fashion: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Save complementary data, such as _Elements indexes."""
        self.elt_idx: list[int]
        if self.cav_params is None:
            logging.error("Failed to init SimulationOutput.elt_idx as "
                          ".cav_params was not provided.")
        else:
            self.elt_idx = [
                i for i, _ in enumerate(self.cav_params['v_cav_mv'], start=1)
            ]

    def __str__(self) -> str:
        """Give a resume of the data that is stored."""
        out = "SimulationOutput:\n"
        out += "\t" + range_vals("z_abs", self.z_abs)
        out += self.synch_trajectory.__str__()
        out += self.beam_parameters.__str__()
        return out

    def has(self, key: str) -> bool:
        """
        Tell if the required attribute is in this class.

        We also call the beam_parameters.has, as it is designed to handle the
        alias (twiss_zdelta <-> zdelta.twiss).
        """
        return key in recursive_items(vars(self)) \
            or self.beam_parameters.has(key)

    def get(self, *keys: str, to_numpy: bool = True,
            elt: _Element | None = None, pos: str | None = None,
            none_to_nan: bool = False,
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
        out : Any
            Attribute(s) value(s).

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
                idx = self.element_to_index(elt=elt, pos=pos)
                val[key] = val[key][idx]

        out = [np.array(val[key])
               if to_numpy and not isinstance(val[key], str)
               else val[key]
               for key in keys]

        if none_to_nan:
            if not to_numpy:
                logging.error(f"{none_to_nan = } while {to_numpy = }, which "
                              "is not supported.")
            out = [val.astype(float) for val in out]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    # in reality, kwargs can be of SimulationOutput type
    def compute_complementary_data(self, elts: ListOfElements,
                                   ref_simulation_output: Any = None,
                                   **kwargs: Any
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
        if self.z_abs is None:
            self.z_abs = elts.get('abs_mesh', remove_first=True)
        self.synch_trajectory.compute_complementary_data()

        # self.beam_parameters.compute_full(self.synch_trajectory.gamma)
        if ref_simulation_output is not None:
            ref_twiss_zdelta = \
                ref_simulation_output.beam_parameters.zdelta.twiss
            self.beam_parameters.compute_mismatch(ref_twiss_zdelta)

        # self.in_tw_fashion = tracewin.interface.output_data_in_tw_fashion()
        logging.critical("data_in_tw_fashion is bugged")
