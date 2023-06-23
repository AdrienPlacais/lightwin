#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021.

@author: placais

TODO : Check if _check_consistency_phases message still relatable
TODO : compute_transfer_matrices: simplify, add a calculation of missing phi_0
at the end
"""
import os.path
import logging
from typing import Any
import numpy as np


import config_manager as con
import tracewin.interface
import tracewin.load
from beam_calculation.output import SimulationOutput
from util.helper import recursive_items, recursive_getter
from core import particle
from core.elements import _Element, FieldMapPath
from core.list_of_elements import ListOfElements, elt_at_this_s_idx, \
    equiv_elt


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, dat_filepath: str, project_folder: str,
                 name: str) -> None:
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name
        self.simulation_output: SimulationOutput

        # Prepare files and folders
        self.files = {
            'dat_filepath': os.path.abspath(dat_filepath),
            'orig_dat_folder': os.path.abspath(os.path.dirname(dat_filepath)),
            'project_folder': project_folder,
            'dat_filecontent': None,
            'field_map_folder': None,
            'out_lw': None,
            'out_tw': None}

        # Load dat file, clean it up (remove comments, etc), load elements
        dat_filecontent = tracewin.load.dat_file(dat_filepath)
        elts = tracewin.interface.create_structure(dat_filecontent)
        elts = self._handle_paths_and_folders(elts)

        self.elts = ListOfElements(elts, w_kin=con.E_MEV, phi_abs=0.,
                                   first_init=True)

        tracewin.interface.set_all_electric_field_maps(
            self.files, self.elts.by_section_and_lattice)
        self.elts.set_absolute_positions()
        self.elts.set_indexes()

        self.files['dat_filecontent'] = dat_filecontent

        self.synch = particle.ParticleInitialState(
            w_kin=con.E_MEV,
            phi_abs=0.,
            synchronous=True
        )

        self._special_getters = self._create_special_getters()
        self._check_consistency_phases()

        self._l_cav = self.elts.l_cav

    @property
    def l_cav(self):
        """Shortcut to easily get list of cavities."""
        return self.elts.l_cav

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True,
            elt: str | _Element | None = None, **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys: str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        elt : str | _Element | None, optional
            If provided, and if the desired keys are in SimulationOutput, the
            attributes will be given over the _Element only. You can provide an
            _Element name, such as `QP1`. If the given _Element is not in the
            Accelerator.ListOfElements, the _Element with the same name that is
            present in this list will be used.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}

        for key in keys:
            if key in self._special_getters:
                val[key] = self._special_getters[key](self)
                if elt is not None:
                    # TODO
                    logging.error("Get attribute by elt not implemented with "
                                  "special getters.")
                continue

            if not self.has(key):
                val[key] = None
                continue

            if elt is not None and (isinstance(elt, str)
                                    or elt not in self.elts):
                elt = self.equiv_elt(elt)

            val[key] = recursive_getter(key, vars(self), to_numpy=False,
                                        elt=elt, **kwargs)

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list) else val
                   for val in out]

        if len(keys) == 1:
            return out[0]
        return tuple(out)

    # TODO add linac name in the subproject folder name
    def _handle_paths_and_folders(self, elts: list[_Element]
                                  ) -> list[_Element]:
        """Make paths absolute, create results folders."""
        # First we take care of where results will be stored
        i = 0

        # i is useful if  you launch several simulations
        out_base = os.path.join(self.files['project_folder'], f"{i:06d}")

        while os.path.exists(out_base):
            i += 1
            out_base = os.path.join(self.files['project_folder'], f"{i:06d}")
        os.makedirs(out_base)
        self.files['out_lw'] = os.path.join(out_base, 'LW')
        self.files['out_tw'] = os.path.join(out_base, 'TW')

        # Now we handle where to look for the field maps
        field_map_basepaths = [basepath
                               for basepath in elts
                               if isinstance(basepath, FieldMapPath)]
        # FIELD_MAP_PATH are not physical elements, so we remove them
        for basepath in field_map_basepaths:
            elts.remove(basepath)

        # If no FIELD_MAP_PATH command was provided, we take field maps in the
        # .dat dir
        if len(field_map_basepaths) == 0:
            field_map_basepaths = [FieldMapPath(
                ['FIELD_MAP_PATH', self.files['orig_dat_folder']])]

        # If more than one field map folder is provided, raise an error
        msg = "Change of base folder for field maps currently not supported."
        assert len(field_map_basepaths) == 1, msg

        # Convert FieldMapPath objects into absolute paths
        field_map_basepaths = [
            os.path.join(self.files['orig_dat_folder'],
                         fm_path.path)
            for fm_path in field_map_basepaths]
        self.files['field_map_folder'] = field_map_basepaths[0]

        return elts

    def _create_special_getters(self) -> dict:
        """Create a dict of aliases that can be accessed w/ the get method."""
        # TODO also remove the M_ij?
        _special_getters = {
            'M_11': lambda self: self.simulation_output.tm_cumul[:, 0, 0],
            'M_12': lambda self: self.simulation_output.tm_cumul[:, 0, 1],
            'M_21': lambda self: self.simulation_output.tm_cumul[:, 1, 0],
            'M_22': lambda self: self.simulation_output.tm_cumul[:, 1, 1],
            'element number': lambda self: self.get('elt_idx') + 1,
        }
        return _special_getters

    def _check_consistency_phases(self) -> None:
        """Check that both TW and LW use absolute or relative phases."""
        flags_absolute = []
        for cav in self.l_cav:
            flags_absolute.append(cav.get('abs_phase_flag'))

        if con.FLAG_PHI_ABS and False in flags_absolute:
            logging.warning(
                "You asked LW a simulation in absolute phase, while there "
                + "is at least one cavity in relative phase in the .dat file "
                + "used by TW. Results won't match if there are faulty "
                + "cavities.")
        elif not con.FLAG_PHI_ABS and True in flags_absolute:
            logging.warning(
                "You asked LW a simulation in relative phase, while there "
                + "is at least one cavity in absolute phase in the .dat file "
                + "used by TW. Results won't match if there are faulty "
                + "cavities.")

    def keep_this(self, simulation_output: SimulationOutput,
                  ref_twiss_zdelta: np.ndarray | None = None) -> None:
        """
        Compute some complementary data and save it as an attribute.

        Parameters
        ----------
        simulation_output : SimulationOutput
            Class that holds all the relatable data created by the
            BeamCalculator.
        ref_twiss_zdelta : np.ndarray | None, optional
            A reference array of Twiss parameters. If provided, it allows the
            calculation of the mismatch factor. The default is None.

        """
        simulation_output.compute_complementary_data(self.elts,
                                                     ref_twiss_zdelta)

        # FIXME This is about storing parameters, not outputs
        for elt, rf_field, cav_params in zip(self.elts,
                                             simulation_output.rf_fields,
                                             simulation_output.cav_params):
            elt.keep_rf_field(rf_field, cav_params)

        self.simulation_output = simulation_output

    def elt_at_this_s_idx(self, s_idx: int, show_info: bool = False
                          ) -> _Element | None:
        """Give the element where the given index is."""
        return elt_at_this_s_idx(self.elts, s_idx, show_info)

    def equiv_elt(self, elt: _Element | str, to_index: bool = False
                  ) -> _Element | int | None:
        """Return an element from self.elts with the same name."""
        return equiv_elt(self.elts, elt, to_index)
