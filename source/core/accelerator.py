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
from typing import TypeVar, Type, Any
import numpy as np

from constants import c
import config_manager as con
import util.tracewin_interface as tw
import util.converters as convert
from util.helper import recursive_items, recursive_getter
from core import particle
from core import elements
from core.list_of_elements import ListOfElements
from core.emittance import beam_parameters_all


_AccelT = TypeVar("_AccelT", bound="Accelerator")


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
        dat_filecontent, l_elts = tw.load_dat_file(dat_filepath)
        l_elts = self._handle_paths_and_folders(l_elts)
        l_elts, l_secs, l_latts, freqs = _sections_lattices(l_elts)

        # Create a the list containing all the elements
        self.elts = ListOfElements(l_elts, w_kin=con.E_MEV, phi_abs=0.,
                                   idx_in=0)

        self.elements = {'l_lattices': l_latts, 'l_sections': l_secs}

        tw.load_filemaps(self.files, self.elements['l_sections'], freqs,
                         con.F_BUNCH_MHZ)
        tw.give_name(l_elts)
        self.files['dat_filecontent'] = dat_filecontent

        # Set indexes and absolute position of the different elements
        last_idx = self._set_indexes_and_abs_positions()

        # Create synchronous particle
        reference = bool(name == 'Working')
        self.synch = particle.Particle(0., con.E_MEV, n_steps=last_idx,
                                       synchronous=True, reference=reference)

        # Transfer matrices
        self.transf_mat = {
            'tm_cumul': np.full((last_idx + 1, 2, 2), np.NaN),
            'tm_indiv': np.full((last_idx + 1, 2, 2), np.NaN),
            'first_calc?': True,
        }
        self.transf_mat['tm_indiv'][0, :, :] = np.eye(2)

        # Beam parameters: emittance, Twiss
        self.beam_param = {
            "twiss": {"twiss_zdelta": np.full((last_idx + 1, 3), np.NaN),
                      "twiss_z": np.full((last_idx + 1, 3), np.NaN),
                      "twiss_w": np.full((last_idx + 1, 3), np.NaN)},
            "eps": {"eps_zdelta": np.full((last_idx + 1), np.NaN),
                    "eps_z": np.full((last_idx + 1), np.NaN),
                    "eps_w": np.full((last_idx + 1), np.NaN)},
            "envelopes": {
                "envelopes_zdelta": np.full((last_idx + 1, 2), np.NaN),
                "envelopes_z": np.full((last_idx + 1, 2), np.NaN),
                "envelopes_w": np.full((last_idx + 1, 2), np.NaN)},
            "mismatch factor": np.full((last_idx + 1), np.NaN),
            "sigma matrix": np.full((last_idx + 1), np.NaN),
        }
        # Define some shortcuts
        self._d_special_getters = self._create_special_getters()

        self.synch.init_abs_z(self.get('abs_mesh', remove_first=True))
        # Check that LW and TW computes the phases in the same way (abs or rel)
        self._check_consistency_phases()

        self.tw_results = {'envelope': {}, 'multipart': {}, 'cav_param': {},
                           'transf_mat': np.empty((0))}

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: tuple[str], to_numpy: bool = True, **kwargs: dict
            ) -> Any:
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if key in self._d_special_getters:
                val[key] = self._d_special_getters[key](self)
                continue

            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), to_numpy=False,
                                        **kwargs)

        # Convert to list, and to numpy array if necessary
        out = [val[key] for key in keys]
        if to_numpy:
            # VisibleDeprecationWarning with Python < 3.11?
            # Error with Python >= 3.11?
            # Try setting to_numpy to False.
            out = [np.array(val) if isinstance(val, list)
                   else val
                   for val in out]

        # Return as tuple or single value
        if len(keys) == 1:
            return out[0]
        # implicit else
        return tuple(out)

    # TODO add linac name in the subproject folder name
    def _handle_paths_and_folders(self, l_elts: list[elements._Element]
                                  ) -> list[elements._Element]:
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
                               for basepath in l_elts
                               if isinstance(basepath, elements.FieldMapPath)]
        # FIELD_MAP_PATH are not physical elements, so we remove them
        for basepath in field_map_basepaths:
            l_elts.remove(basepath)

        # If no FIELD_MAP_PATH command was provided, we take field maps in the
        # .dat dir
        if len(field_map_basepaths) == 0:
            field_map_basepaths = [elements.FieldMapPath(
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

        return l_elts

    def _create_special_getters(self) -> dict:
        """Create a dict of aliases that can be accessed w/ the get method."""
        _d_special_getters = {
            'alpha_zdelta': lambda self:
                self.beam_param['twiss']['twiss_zdelta'][:, 0],
            'beta_zdelta': lambda self:
                self.beam_param['twiss']['twiss_zdelta'][:, 1],
            'gamma_zdelta': lambda self:
                self.beam_param['twiss']['twiss_zdelta'][:, 2],
            'alpha_z': lambda self:
                self.beam_param['twiss']['twiss_z'][:, 0],
            'beta_z': lambda self:
                self.beam_param['twiss']['twiss_z'][:, 1],
            'gamma_z': lambda self:
                self.beam_param['twiss']['twiss_z'][:, 2],
            'alpha_w': lambda self:
                self.beam_param['twiss']['twiss_w'][:, 0],
            'beta_w': lambda self:
                self.beam_param['twiss']['twiss_w'][:, 1],
            'gamma_w': lambda self:
                self.beam_param['twiss']['twiss_w'][:, 2],
            'envelope_pos_zdelta': lambda self:
                self.beam_param['envelopes']['envelopes_zdelta'][:, 0],
            'envelope_energy_zdelta': lambda self:
                self.beam_param['envelopes']['envelopes_zdelta'][:, 1],
            'envelope_pos_z': lambda self:
                self.beam_param['envelopes']['envelopes_z'][:, 0],
            'envelope_energy_z': lambda self:
                self.beam_param['envelopes']['envelopes_z'][:, 1],
            'envelope_pos_w': lambda self:
                self.beam_param['envelopes']['envelopes_w'][:, 0],
            'envelope_energy_w': lambda self:
                self.beam_param['envelopes']['envelopes_w'][:, 1],
            'M_11': lambda self: self.transf_mat['tm_cumul'][:, 0, 0],
            'M_12': lambda self: self.transf_mat['tm_cumul'][:, 0, 1],
            'M_21': lambda self: self.transf_mat['tm_cumul'][:, 1, 0],
            'M_22': lambda self: self.transf_mat['tm_cumul'][:, 1, 1],
            'element number': lambda self: self.get('elt_idx') + 1,
        }
        return _d_special_getters

    def _set_indexes_and_abs_positions(self) -> int:
        """Init solvers, set indexes and absolute positions of elements."""
        pos = {'in': 0., 'out': 0.}
        idx = {'in': 0, 'out': 0}

        for elt in self.elts:
            elt.init_solvers()

            pos['in'] = pos['out']
            pos['out'] += elt.length_m
            elt.solver_param['abs_mesh'] = elt.get('rel_mesh') + pos['in']

            idx['in'] = idx['out']
            idx['out'] += elt.get('n_steps')
            elt.idx['s_in'], elt.idx['s_out'] = idx['in'], idx['out']
        return idx['out']

    def _check_consistency_phases(self) -> None:
        """Check that both TW and LW use absolute or relative phases."""
        cavities = self.elements_of(nature='FIELD_MAP')
        flags_absolute = []
        for cav in cavities:
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

    def store_results(self, results: dict, l_elts: list[elements._Element, ...]
                      ) -> None:
        """
        We save data into the appropriate objects.

        In particular:
            energy and phase into accelerator.synch
            rf field parameters, element transfer matrices into Elements
            global transfer matrices into Accelerator

        This function is called when the fitting is not required/is already
        finished.
        """
        idx_in = l_elts[0].idx['s_in']
        idx_out = l_elts[-1].idx['s_out'] + 1

        # Go across elements
        for elt, rf_field, cav_params in zip(l_elts, results['rf_fields'],
                                             results["cav_params"]):
            idx_abs = range(elt.idx['s_in'] + 1, elt.idx['s_out'] + 1)
            idx_rel = range(elt.idx['s_in'] + 1 - idx_in,
                            elt.idx['s_out'] + 1 - idx_in)
            transf_mat_elt = results["tm_cumul"][idx_rel]

            self.transf_mat['tm_indiv'][idx_abs] = transf_mat_elt
            elt.keep_rf_field(rf_field, cav_params)

        # Save into Accelerator
        self.transf_mat['tm_cumul'][idx_in:idx_out] = results["tm_cumul"]
        self.beam_param["sigma matrix"] = results["sigma matrix"]
        # Mismatch will be None straight out of
        # ListOfElements._pack_into_single_dict method
        # We add it manually to results dict during the fitting process
        # (FaultScenario)
        mism = results["mismatch factor"]
        if mism is not None:
            self.beam_param["mismatch factor"] = mism

        # Save into Particle
        self.synch.keep_energy_and_phase(results, range(idx_in, idx_out))

        # Save into Accelerator
        gamma = convert.energy(np.array(results["w_kin"]), "kin to gamma")
        d_beam_param = beam_parameters_all(results['eps_zdelta'],
                                           results['twiss_zdelta'], gamma)

        # Go across beam parameters (Twiss, emittance, long. envelopes)
        for item1 in self.beam_param.items():
            if not isinstance(item1[1], dict):
                continue
            # Go across phase spaces (z-z', z-delta, w-phi)
            for item2 in item1[1].items():
                item2[1][idx_in:idx_out] = d_beam_param[item1[0]][item2[0]]

    def elements_of(self, nature: str,
                    sub_list: list[elements._Element] | None = None
                    ) -> list[elements._Element]:
        """Return a list of elements of nature 'nature'."""
        if sub_list is None:
            sub_list = self.elts
        list_of = list(filter(lambda elt: elt.get('nature') == nature,
                              sub_list))
        return list_of

    def where_is_this_index(self, idx: int, show_info: bool = False
                            ) -> elements._Element | None:
        """Give the element where the given index is."""
        found, elt = False, None

        for elt in self.elts:
            if idx in range(elt.idx['s_in'], elt.idx['s_out']):
                found = True
                break

        if show_info:
            if found:
                print(f"Mesh index {idx} is in {elt.get('elt_info')}.")
                print(f"Indexes of this elt: {elt.get('idx')}.\n")
            else:
                print(f"Mesh index {idx} does not belong to any element.")
        return elt if found else None

    def simulate_in_tracewin(self, ini_path: str, **kwargs: str) -> None:
        """Compute this linac with TraceWin."""
        if 'Broken' in self.name:
            logging.warning("simulating broken linac in TW useless.")

        os.makedirs(self.get('out_tw'))
        path_cal = self.get("out_tw")
        dat_file = self.get("dat_filepath")
        for path in [ini_path, path_cal, dat_file]:
            path = os.path.abspath(path)
        tw.run_tw(ini_path, path_cal, dat_file, **kwargs)

    def store_tracewin_results(self) -> None:
        """Take the results created by TraceWin."""
        folder = self.get('out_tw')

        self.tw_results['envelope'] = tw.get_multipart_tw_results(
            folder, filename='tracewin.out')

        self.tw_results['multipart'] = tw.get_multipart_tw_results(
            folder, filename='partran1.out')

        _, _, self.tw_results['transf_mat'] = tw.get_transfer_matrices(
            folder, filename='Transfer_matrix1.dat')

        self.tw_results['cav_param'] = tw.get_tw_cav_param(
            folder, filename='Cav_set_point_res.dat')

    def precompute_some_tracewin_results(self) -> None:
        """Use TW result dicts to compute other useful results."""
        for dic in [self.tw_results['envelope'], self.tw_results['multipart']]:
            dic['gamma'] = 1. + dic['gama-1']
            dic['w_kin'] = convert.energy(dic['gamma'], "gamma to kin")
            dic['beta'] = convert.energy(dic['w_kin'], "kin to beta")
            dic['lambda'] = c / 162e6

            num = dic['beta'].shape[0]
            dic['phi_abs_array'] = np.full((num), 0.)
            for i in range(1, num):
                delta_z = dic['z(m)'][i] - dic['z(m)'][i - 1]
                beta_i = dic['beta'][i]
                delta_phi = 2. * np.pi * delta_z / (dic['lambda'] * beta_i)
                dic['phi_abs_array'][i] = dic['phi_abs_array'][i - 1] \
                    + np.rad2deg(delta_phi)

            # Transverse emittance, used in evaluate
            dic['et'] = 0.5 * (dic['ex'] + dic['ey'])

            # Twiss parameters, used in evaluate
            for _eps, size, disp, twi in zip(
                    ['ex', 'ey', 'ezdp'],
                    ['SizeX', 'SizeY', 'SizeZ'],
                    ["sxx'", "syy'", 'szdp'],
                    ['twiss_x', 'twiss_y', 'twiss_zdp']):
                eps = dic[_eps] / (dic['gamma'] * dic['beta'])
                alpha = -dic[disp] / eps
                beta = dic[size]**2 / eps
                if _eps == 'ezdp':
                    beta /= 10.
                gamma = (1. + alpha**2) / beta
                dic[twi] = np.column_stack((alpha, beta, gamma))

    def resample_tw_results(self, ref_linac: Type[_AccelT]) -> None:
        """Re-evaluate all quantities at the same z as the ref linac."""
        d_tw = self.tw_results['multipart']
        z_fix = d_tw['z(m)'].copy()

        z_ref = ref_linac.tw_results['multipart']['z(m)']

        for key in d_tw:
            d_tw[key] = np.interp(z_ref, z_fix, d_tw[key])


def _sections_lattices(l_elts: list[elements._Element]) -> tuple[
        list[elements._Element], list[elements._Element],
        list[elements._Element], dict]:
    """Gather elements by section and lattice."""
    l_elts, d_struct = _prepare_sections_and_lattices(l_elts)

    lattices = []
    sections = []
    j = 0
    for i in range(d_struct['n_sections']):
        lattices = []
        n_lattice = d_struct['lattices'][i].n_lattice
        while j < d_struct['indices'][i]:
            lattices.append(l_elts[j:j + n_lattice])
            j += n_lattice
        sections.append(lattices)

    shift_lattice = 0
    j = 0
    for i, sec in enumerate(sections):
        for j, lattice in enumerate(sec):
            for elt in lattice:
                elt.idx['section'] = i
                elt.idx['lattice'] = j + shift_lattice
                elt.idx['elt_idx'] = l_elts.index(elt)
        shift_lattice += j + 1
    lattices = []
    for sec in sections:
        lattices += sec
    return l_elts, sections, lattices, d_struct['frequencies']


def _prepare_sections_and_lattices(l_elts: list[elements._Element]) -> tuple[
        list[elements._Element], dict]:
    """
    Save info on the accelerator structure.

    In particular: in every section, the number of elements per lattice,
    the rf frequency of cavities, the position of the delimitations between
    two sections.
    """
    lattices = [lattice
                for lattice in l_elts
                if isinstance(lattice, elements.Lattice)
                ]
    frequencies = [frequency
                   for frequency in l_elts
                   if isinstance(frequency, elements.Freq)
                   ]

    n_sections = len(lattices)
    assert n_sections == len(frequencies)

    idx_of_section_changes = []
    n_secs_before_current = 0
    for i in range(n_sections):
        latt, freq = lattices[i], frequencies[i]

        idx_latt = l_elts.index(latt)
        idx_freq = l_elts.index(freq)
        assert idx_freq - idx_latt == 1

        idx_of_section_changes.append(idx_latt - 2 * n_secs_before_current)
        n_secs_before_current += 1

        l_elts.pop(idx_freq)
        l_elts.pop(idx_latt)
    idx_of_section_changes.pop(0)
    idx_of_section_changes.append(len(l_elts))

    return l_elts, {'lattices': lattices, 'frequencies': frequencies,
                    'indices': idx_of_section_changes,
                    'n_sections': n_sections}
