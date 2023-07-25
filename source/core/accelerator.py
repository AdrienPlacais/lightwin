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
import pandas as pd

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

    def __init__(self, name: str, dat_file: str, project_folder: str,
                 beam_calc_path: str, beam_calc_post_path: str | None,
                 ) -> None:
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name
        self.simulation_outputs: dict[str, SimulationOutput] = {}
        self.data_in_tw_fashion: pd.DataFrame

        # Prepare files and folders
        self.files = {
            'dat_filepath': dat_file,
            'orig_dat_folder': os.path.dirname(dat_file),
            'project_folder': project_folder,
            'dat_filecontent': None,
            'field_map_folder': None,
            'beam_calc_path': beam_calc_path,
            'beam_calc_post_path': beam_calc_post_path}

        # Load dat file, clean it up (remove comments, etc), load elements
        dat_filecontent = tracewin.load.dat_file(dat_file)
        elts = tracewin.interface.create_structure(dat_filecontent)
        elts = self._set_field_map_files_paths(elts)

        self.elts = ListOfElements(elts, w_kin=con.E_MEV, phi_abs=0.,
                                   first_init=True)

        tracewin.interface.set_all_electric_field_maps(
            self.files, self.elts.by_section_and_lattice)

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

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
            elt: str | _Element | None = None, **kwargs: bool | str) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        none_to_nan : bool, optional
            To convert None to np.NaN. The default is False.
        elt : str | _Element | None, optional
            If provided, and if the desired keys are in SimulationOutput, the
            attributes will be given over the _Element only. You can provide an
            _Element name, such as `QP1`. If the given _Element is not in the
            Accelerator.ListOfElements, the _Element with the same name that is
            present in this list will be used.
        **kwargs : bool | str
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
                                        none_to_nan=False,
                                        elt=elt, **kwargs)

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list) else val
                   for val in out]
            if none_to_nan:
                out = [val.astype(float) for val in out]

        if len(keys) == 1:
            return out[0]
        return tuple(out)

    def _set_field_map_files_paths(self, elts: list[_Element]
                                   ) -> list[_Element]:
        """Load FIELD_MAP_PATH, remove it from the list of elements."""
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
        # FIXME this won't work with new simulation output
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

    def keep_settings(self, simulation_output: SimulationOutput) -> None:
        """Save cavity parameters in _Elements and new .dat file."""
        for i, (elt, rf_field) in enumerate(zip(self.elts,
                                                simulation_output.rf_fields)):
            v_cav_mv = simulation_output.cav_params['v_cav_mv'][i]
            phi_s = simulation_output.cav_params['phi_s'][i]
            elt.keep_rf_field(rf_field, v_cav_mv, phi_s)

        self._store_settings_in_dat(save=True)

    def elt_at_this_s_idx(self, s_idx: int, show_info: bool = False
                          ) -> _Element | None:
        """Give the element where the given index is."""
        return elt_at_this_s_idx(self.elts, s_idx, show_info)

    def equiv_elt(self, elt: _Element | str, to_index: bool = False
                  ) -> _Element | int | None:
        """Return an element from self.elts with the same name."""
        return equiv_elt(self.elts, elt, to_index)

    def _store_settings_in_dat(self, save: bool = True) -> None:
        """Update the dat file, save it if asked."""
        tracewin.interface.update_dat_with_fixed_cavities(
            self.get('dat_filecontent', to_numpy=False),
            self.elts,
            self.get('field_map_folder')
        )

        if save:
            dat_filepath = os.path.join(
                self.get('beam_calc_path'),
                os.path.basename(self.get('dat_filepath')))
            self.files['dat_filepath'] = dat_filepath
            with open(self.get('dat_filepath'), 'w') as file:
                for line in self.files['dat_filecontent']:
                    file.write(' '.join(line) + '\n')
            logging.info(f"New dat saved in {self.get('dat_filepath')}")


def accelerator_factory(files: dict[str, str], beam_calculator: dict[str, Any],
                        beam: dict[str, Any],
                        wtf: dict[str, Any] | None = None,
                        beam_calculator_post: dict[str, Any] | None = None,
                        **kwargs
                        ) -> list[Accelerator]:
    """Create the required Accelerators as well as their output folders."""
    n_simulations = 1
    if wtf is not None:
        n_simulations = len(wtf['failed']) + 1

    beam_calc_paths, beam_calc_post_paths = _generate_folders_tree_structure(
        project_folder=files['project_folder'],
        n_simulations=n_simulations,
        tool=beam_calculator['tool'],
        post_tool=beam_calculator_post['tool']
        if beam_calculator_post is not None else None
    )
    names = ['Broken' if i > 0 else 'Working' for i in range(n_simulations)]

    accelerators = [Accelerator(name,
                                **files,
                                beam_calc_path=beam_calc_path,
                                beam_calc_post_path=beam_calc_post_path)
                    for name, beam_calc_path, beam_calc_post_path
                    in zip(names, beam_calc_paths, beam_calc_post_paths)]
    return accelerators


def _generate_folders_tree_structure(project_folder: str,
                                     n_simulations: int,
                                     tool: str, post_tool: str | None = None
                                     ) -> tuple[list[str], list[str | None]]:
    """
    Create the proper folders for every Accelerator.

    The default structure is:

    where_original_dat_is/
        YYYY.MM.DD_HHhMM_SSs_MILLIms/              <- project_folder
            000000_ref/                            <- fault_scenario_path
                beam_calculation_toolname/         <- beam_calc
                (beam_calculation_post_toolname)/  <- beam_calc_post
            000001/
                beam_calculation_toolname/
                (beam_calculation_post_toolname)/
            000002/
                beam_calculation_toolname/
                (beam_calculation_post_toolname)/
            etc
    """
    fault_scenario_paths = [os.path.join(project_folder, f"{i:06d}")
                            for i in range(n_simulations)]
    fault_scenario_paths[0] += '_ref'

    base_beam_calc = f"beam_calculation_{tool}"
    beam_calc_paths = [os.path.join(fault_scenar, base_beam_calc)
                       for fault_scenar in fault_scenario_paths]
    _ = [os.makedirs(beam_calc_path) for beam_calc_path in beam_calc_paths]

    beam_calc_post_paths = [None for fault_scenar in fault_scenario_paths]
    if post_tool is not None:
        base_beam_calc_post = f"beam_calculation_post_{post_tool}"
        beam_calc_post_paths = [os.path.join(fault_scenar, base_beam_calc_post)
                                for fault_scenar in fault_scenario_paths]
        _ = [os.makedirs(beam_calc_path)
             for beam_calc_path in beam_calc_post_paths]

    return beam_calc_paths, beam_calc_post_paths
