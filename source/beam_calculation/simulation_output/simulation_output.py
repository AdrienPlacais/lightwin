#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define a class to store outputs from different :class:`.BeamCalculator`.

.. todo::
    Clarify difference cav_params vs rf_fields

.. todo::
    Do I really need the `r_zz_elt` key??

.. todo::
    Do I really need z_abs? Envelope1D does not uses it while TraceWin does.

.. todo::
    Transfer matrices are stored in :class:`.TransferMatrix`, but also in
    :data:`.BeamParameters.zdelta`.

"""
from dataclasses import dataclass
import datetime
import logging
import os.path
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.beam_parameters.beam_parameters import (
    BeamParameters,
    mismatch_from_objects,
)
from core.elements.element import Element
from core.list_of_elements.list_of_elements import ListOfElements
from core.particle import ParticleFullTrajectory
from core.transfer_matrix.transfer_matrix import TransferMatrix
from util.helper import recursive_items, recursive_getter, range_vals


@dataclass
class SimulationOutput:
    """
    Stores the information produced by a `BeamCalculator`.

    Used for fitting, post-processing, plotting.

    Attributes
    ----------
    out_folder : str
        Results folder used by the :class:`.BeamCalculator` that created this.
    is_multiparticle : bool
        Tells if the simulation is a multiparticle simulation.
    is_3d : bool
        Tells if the simulation is in 3D.
    synch_trajectory : ParticleFullTrajectory | None
        Holds energy, phase of the synchronous particle.
    cav_params : dict[str, float | None] | None
        Holds amplitude, synchronous phase, absolute phase, relative phase of
        cavities.
    rf_fields : list[dict] | None
        Holds amplitude, synchronous phase, absolute phase, relative phase of
        cavities.
    beam_parameters : BeamParameters | None
        Holds emittance, Twiss parameters, envelopes in the various phase
        spaces.
    element_to_index : Callable[[str | Element, str | None], int | slice] | None
        Takes an :class:`.Element`, its name, 'first' or 'last' as argument,
        and returns corresponding index. Index should be the same in all the
        arrays attributes of this class: ``z_abs``, ``beam_parameters``
        attributes, etc.  Used to easily `get` the desired properties at the
        proper position.
    transfer_matrix : TransferMatrix
         Holds absolute and relative transfer matrices in all planes.
    z_abs : np.ndarray | None, optional
        Absolute position in the linac in m. The default is None.
    in_tw_fashion : pd.DataFrame | None, optional
        A way to output the :class:`.SimulationOutput` in the same way as the
        ``Data`` tab of TraceWin. The default is None.
    r_zz_elt : list[np.ndarray] | None, optional
        Cumulated transfer matrices in the [z-delta] plane. The default is
        None.

    """

    out_folder: str
    is_multiparticle: bool
    is_3d: bool

    synch_trajectory: ParticleFullTrajectory | None

    cav_params: dict[str, float | None] | None
    rf_fields: list[dict] | None

    beam_parameters: BeamParameters | None

    element_to_index: Callable[[str | Element, str | None], int | slice] \
        | None

    transfer_matrix: TransferMatrix | None = None
    z_abs: np.ndarray | None = None
    in_tw_fashion: pd.DataFrame | None = None
    r_zz_elt: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Save complementary data, such as `Element` indexes."""
        self.elt_idx: list[int]
        if self.cav_params is None:
            logging.error("Failed to init SimulationOutput.elt_idx as "
                          ".cav_params was not provided.")
        else:
            self.elt_idx = [
                i for i, _ in enumerate(self.cav_params['v_cav_mv'], start=1)
            ]
        self.out_path: str | None = None

    def __str__(self) -> str:
        """Give a resume of the data that is stored."""
        out = "SimulationOutput:\n"
        out += "\t" + range_vals("z_abs", self.z_abs)
        out += self.synch_trajectory.__str__()
        out += self.beam_parameters.__str__()
        return out

    @property
    def beam_calculator_information(self) -> str:
        """Use ``out_path`` to retrieve info on :class:`BeamCalculator`."""
        if self.out_path is None:
            return self.out_folder
        return get_nth_parent(self.out_path, nth=2)

    def has(self, key: str) -> bool:
        """
        Tell if the required attribute is in this class.

        We also call the :meth:`.BeamParameters.has`, as it is designed to
        handle the alias (such as ``twiss_zdelta`` <=> ``zdelta.twiss``).

        """
        return key in recursive_items(vars(self)) \
            or self.beam_parameters.has(key)

    def get(self, *keys: str, to_numpy: bool = True, to_deg: bool = False,
            elt: Element | None = None, pos: str | None = None,
            none_to_nan: bool = False, **kwargs: str | bool | None) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        to_deg : bool, optional
            To apply np.rad2deg function over every ``key`` containing the
            string.
        elt : Element | None, optional
            If provided, return the attributes only at the considered element.
        pos : 'in' | 'out' | None
            If you want the attribute at the entry, exit, or in the whole
            element.
        none_to_nan : bool, optional
            To convert None to np.NaN. The default is False.
        **kwargs : str | bool | None
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

            val[key] = recursive_getter(key, vars(self), to_numpy=False,
                                        **kwargs)

            if val[key] is None:
                continue

            if to_deg and 'phi' in key:
                val[key] = _to_deg(val[key])

            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

            if None not in (self.element_to_index, elt):
                return_elt_idx = False
                if key in ('v_cav_mv', 'phi_s'):
                    return_elt_idx = True
                idx = self.element_to_index(elt=elt, pos=pos,
                                            return_elt_idx=return_elt_idx)
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
            Must be a full :class:`.ListOfElements`, containing all the
            elements of the linac.
        ref_twiss_zdelta : np.ndarray | None, optional
            A reference array of Twiss parameters. If provided, it allows the
            calculation of the mismatch factor. The default is None.

        """
        if self.z_abs is None:
            self.z_abs = elts.get('abs_mesh', remove_first=True)
        self.synch_trajectory.compute_complementary_data()

        # self.beam_parameters.compute_full(self.synch_trajectory.gamma)
        if ref_simulation_output is not None:
            phase_spaces, set_transverse_as_average = ('zdelta',), False
            if 't' in self.beam_parameters.__dir__():
                phase_spaces = ('zdelta', 'x', 'y')
                set_transverse_as_average = True

            mismatch_from_objects(
                ref_simulation_output.beam_parameters,
                self.beam_parameters,
                *phase_spaces,
                set_transverse_as_average=set_transverse_as_average)

        # self.in_tw_fashion = tracewin.interface.output_data_in_tw_fashion()
        # FIXME
        logging.warning("data_in_tw_fashion is bugged")


def _to_deg(val: np.ndarray | list | float | None
            ) -> np.ndarray | list | float | None:
    """Convert the ``val[key]`` into deg if it is not None."""
    if val is None:
        return None
    if isinstance(val, list):
        return [np.rad2deg(angle) if angle is not None else None
                for angle in val]
    return np.rad2deg(val)


def get_nth_parent(filepath: str, nth: int) -> str:
    """Return the path of current folder + n."""
    path_as_list = list(Path(filepath).parts)
    new_path_as_list = path_as_list[-nth:]
    new_path = os.path.join(*new_path_as_list)
    return new_path
