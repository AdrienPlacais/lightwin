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
import numpy as np
import tracewin_interface as tw
import particle
from constants import E_MEV, FLAG_PHI_ABS
import elements
from emittance import beam_parameters_zdelta, beam_parameters_all, \
    mismatch_factor
from helper import kin_to_gamma


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, dat_filepath, name):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.elements['list'].
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name

        # Load dat file, clean it up (remove comments, etc), load elements
        dat_filecontent, l_elts = tw.load_dat_file(dat_filepath)
        l_elts, l_secs, l_latts, freqs = _sections_lattices(l_elts)

        self.elements = {'n': len(l_elts),
                         'list': l_elts,
                         'l_lattices': l_latts,
                         'l_sections': l_secs}

        tw.load_filemaps(dat_filepath, dat_filecontent,
                         self.elements['l_sections'], freqs)
        tw.give_name(self.elements['list'])

        self.files = {'project_folder': os.path.dirname(dat_filepath),
                      'dat_filecontent': dat_filecontent,
                      'results_folder':
                          os.path.dirname(dat_filepath) + '/results_lw/'}

        # Set indexes and absolute position of the different elements
        last_idx = self._set_indexes_and_abs_positions()

        # Create synchronous particle
        reference = bool(name == 'Working')
        self.synch = particle.Particle(0., E_MEV, n_steps=last_idx,
                                       synchronous=True, reference=reference)
        self.synch.init_abs_z(self.elements['list'])

        # Transfer matrices
        self.transf_mat = {
            'cumul': np.full((last_idx + 1, 2, 2), np.NaN),
            'indiv': np.full((last_idx + 1, 2, 2), np.NaN),
            'first_calc?': True,
        }
        self.transf_mat['indiv'][0, :, :] = np.eye(2)

        # Beam parameters: emittance, Twiss
        self.beam_param = {
            "twiss": {"zdelta": np.full((last_idx + 1, 3), np.NaN),
                      "z": np.full((last_idx + 1, 3), np.NaN),
                      "w": np.full((last_idx + 1, 3), np.NaN)},
            "eps": {"zdelta": np.full((last_idx + 1), np.NaN),
                    "z": np.full((last_idx + 1), np.NaN),
                    "w": np.full((last_idx + 1), np.NaN)},
            "enveloppes": {"zdelta": np.full((last_idx + 1, 2), np.NaN),
                           "z": np.full((last_idx + 1, 2), np.NaN),
                           "w": np.full((last_idx + 1, 2), np.NaN)},
            "mismatch factor": np.full((last_idx + 1), np.NaN)
        }

        # Check that LW and TW computes the phases in the same way (abs or rel)
        self._check_consistency_phases()

    def _set_indexes_and_abs_positions(self):
        """Init solvers, set indexes and absolute positions of elements."""
        pos = {'in': 0., 'out': 0.}
        idx = {'in': 0, 'out': 0}

        for elt in self.elements['list']:
            elt.init_solvers()

            pos['in'] = pos['out']
            pos['out'] += elt.length_m
            elt.pos_m['abs'] = elt.pos_m['rel'] + pos['in']

            idx['in'] = idx['out']
            idx['out'] += elt.tmat['solver_param']['n_steps']
            elt.idx['s_in'], elt.idx['s_out'] = idx['in'], idx['out']
        return idx['out']

    def _check_consistency_phases(self):
        """Check that both TW and LW use absolute or relative phases."""
        cavities = self.elements_of(nature='FIELD_MAP')
        flags_absolute = []
        for cav in cavities:
            flags_absolute.append(cav.acc_field.phi_0['abs_phase_flag'])

        if FLAG_PHI_ABS and False in flags_absolute:
            print("Warning: you asked LW a simulation in absolute phase,",
                  "while there is at least one cavity in relative phase in",
                  "the .dat file used by TW. Results won't match if there",
                  "are faulty cavities.\n")
        elif not FLAG_PHI_ABS and True in flags_absolute:
            print("Warning: you asked LW a simulation in relative phase,",
                  "while there is at least one cavity in absolute phase in",
                  "the .dat file used by TW. Results won't match if there",
                  "are faulty cavities.\n")

    def compute_transfer_matrices(self, l_elts=None, d_fits=None,
                                  flag_transfer_data=True):
        """
        Compute the transfer matrices of Accelerator's elements.

        Parameters
        ----------
        l_elts : list of Elements, optional
            List of elements from which you want the transfer matrices. Default
            is None. In this case, MT calculated for the whole linac.
        d_fits: dict, optional
            Dict to where norms and phases of compensating cavities are stored.
            If the dict is None, we take norms and phases from cavity objects.
        flag_transfer_data : boolean, optional
            If True, we save the energies, transfer matrices, etc that are
            calculated in the routine.
        """
        if l_elts is None:
            l_elts = self.elements['list']

        # Prepare lists to store each element's results
        l_elt_results = []
        l_rf_fields = []

        # Initial phase and energy values:
        idx_in = l_elts[0].idx['s_in']
        w_kin = self.synch.energy['kin_array_mev'][idx_in]
        phi_abs = self.synch.phi['abs_array'][idx_in]

        # Compute transfer matrix and acceleration in each element
        for elt in l_elts:
            elt_results, rf_field = \
                self._proper_transf_mat(elt, phi_abs, w_kin, d_fits)

            # Store this element's results
            l_elt_results.append(elt_results)
            l_rf_fields.append(rf_field)

            # If there is nominal cavities in the recalculated zone during a
            # fit, we remove the associated rf fields and phi_s_rad
            if (not flag_transfer_data) \
                and (d_fits is not None) \
                    and (elt.info['status'] == 'nominal'):
                l_rf_fields[-1] = None
                l_elt_results[-1]['phi_s_rad'] = None

            # Update energy and phase
            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        # We store all relevant data in results: evolution of energy, phase,
        # transfer matrices, emittances, etc
        results = self._pack_into_single_dict(l_elt_results, l_rf_fields,
                                              idx_in)

        if flag_transfer_data:
            self._definitive_save_into_accelerator_element_and_synch_objects(
                results, l_elts)

        return results

    def _proper_transf_mat(self, elt, phi_abs, w_kin, d_fits):
        """Get the proper arguments and call the elt.calc_transf_mat."""
        if elt.info['nature'] != 'FIELD_MAP' or elt.info['status'] == 'failed':
            rf_field = None
            elt_results = elt.calc_transf_mat(w_kin)

        else:
            # Here we determine if we take the rf field parameters from an
            # optimisation algorithm or from the Element.Rf_Field object
            if d_fits is not None \
               and elt.info['status'] == 'compensate (in progress)':
                d_fit_elt = {'flag': True,
                             'phi': d_fits['l_phi'].pop(0),
                             'k_e': d_fits['l_k_e'].pop(0)}
            else:
                d_fit_elt = d_fits

            rf_field = elt.rf_param(self.synch, phi_abs, w_kin, d_fit_elt)
            elt_results = elt.calc_transf_mat(w_kin, **rf_field)

        return elt_results, rf_field

    # Could be function instead of method
    def _indiv_to_cumul_transf_mat(self, l_r_zz_elt, idx_in, n_steps):
        """Compute cumulated transfer matrix."""
        # Compute transfer matrix of l_elts
        arr_r_zz_cumul = np.full((n_steps, 2, 2), np.NaN)

        # If we are at the start of the linac, initial transf mat is unity
        if idx_in == 0:
            arr_r_zz_cumul[0] = np.eye(2)
        else:
            # Else we take the tm at the start of l_elts
            arr_r_zz_cumul[0] = self.transf_mat['cumul'][idx_in]
            assert ~np.isnan(arr_r_zz_cumul[0]).any(), \
                "Previous transfer matrix was not calculated."

        for i in range(1, n_steps):
            arr_r_zz_cumul[i] = l_r_zz_elt[i - 1] @ arr_r_zz_cumul[i - 1]

        return arr_r_zz_cumul

    # Could be function instead of method
    # FIXME could be simpler
    def _pack_into_single_dict(self, l_elt_results, l_rf_fields, idx_in):
        """
        We store energy, transfer matrices, phase, etc into the results dict.

        This dict is used in the fitting process.
        """
        # To store results
        results = {
            "phi_s_rad": [],
            "cav_params": [],
            "w_kin": [self.synch.energy['kin_array_mev'][idx_in]],
            "phi_abs": [self.synch.phi['abs_array'][idx_in]],
            "r_zz_elt": [],         # List of numpy arrays
            "r_zz_cumul": None,     # (n, 2, 2) numpy array
            "rf_fields": [],        # List of dicts
            "d_zdelta": None,
        }

        for elt_results, rf_field in zip(l_elt_results, l_rf_fields):
            results["rf_fields"].append(rf_field)
            results["cav_params"].append(elt_results["cav_params"])
            if rf_field is not None:
                results["phi_s_rad"].append(
                    elt_results['cav_params']['phi_s_rad'])

            r_zz_elt = [elt_results['r_zz'][i, :, :]
                        for i in range(elt_results['r_zz'].shape[0])]
            results["r_zz_elt"].extend(r_zz_elt)

            l_phi_abs = [phi_rel + results["phi_abs"][-1]
                         for phi_rel in elt_results['phi_rel']]
            results["phi_abs"].extend(l_phi_abs)

            results["w_kin"].extend(elt_results['w_kin'].tolist())

        results["r_zz_cumul"] = self._indiv_to_cumul_transf_mat(
            results["r_zz_elt"], idx_in, len(results["w_kin"]))

        results["d_zdelta"] = beam_parameters_zdelta(results["r_zz_cumul"])
        return results

    def _definitive_save_into_accelerator_element_and_synch_objects(
            self, results, l_elts):
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
            transf_mat_elt = results["r_zz_cumul"][idx_rel]

            self.transf_mat["indiv"][idx_abs] = transf_mat_elt
            elt.keep_mt_and_rf_field(transf_mat_elt, rf_field, cav_params)

        # Save into Accelerator
        self.transf_mat['cumul'][idx_in:idx_out] = results["r_zz_cumul"]

        # Save into Particle
        self.synch.keep_energy_and_phase(results, range(idx_in, idx_out))

        # Save into Accelerator
        gamma = kin_to_gamma(np.array(results["w_kin"]))
        d_beam_param = beam_parameters_all(results["d_zdelta"], gamma)

        # Go across beam parameters (Twiss, emittance, long. enveloppes)
        for item1 in self.beam_param.items():
            if not isinstance(item1[1], dict):
                continue
            # Go across phase spaces (z-z', z-delta, w-phi)
            for item2 in item1[1].items():
                item2[1][idx_in:idx_out] = d_beam_param[item1[0]][item2[0]]

    def get_from_elements(self, attribute, key=None, other_key=None):
        """
        Return attribute from all elements in list_of_elements.

        Parameters
        ----------
        attribute : string
            Name of the desired attribute.
        key : string, optional
            If attribute is a dict, key must be provided. Default is None.
        other_key : string, optional
            If attribute[key] is a dict, a second key must be provided. Default
            is None.
        """
        list_of_keys = vars(self.elements['list'][0])
        data_nature = str(type(list_of_keys[attribute]))

        dict_data_getter = {
            "<class 'dict'>": lambda elt_dict, key: elt_dict[key],
            "<class 'electric_field.RfField'>": lambda elt_class, key:
                vars(elt_class)[key],
            "<class 'numpy.ndarray'>": lambda data_new, key: data_new,
            "<class 'float'>": lambda data_new, key: data_new,
            "<class 'str'>": lambda data_new, key: data_new,
        }
        fun = dict_data_getter[data_nature]

        data_out = []
        for elt in self.elements['list']:
            list_of_keys = vars(elt)
            piece_of_data = fun(list_of_keys[attribute], key)
            if isinstance(piece_of_data, dict):
                piece_of_data = piece_of_data[other_key]
            data_out.append(piece_of_data)

        # Concatenate into a single matrix
        if attribute == 'transfer_matrix':
            data_array = data_out[0]
            for i in range(1, len(data_out)):
                data_array = np.vstack((data_array, data_out[i]))
        # Transform into array
        else:
            data_array = np.array(data_out)
        return data_array

    def elements_of(self, nature, sub_list=None):
        """
        Return a list of elements of nature 'nature'.

        Parameters
        ----------
        nature : string
            Nature of the elements you want, eg FIELD_MAP or DRIFT.
        sub_list : list, optional
            List of elements (eg module) if you want the elements only in this
            module.

        Returns
        -------
        list_of : list of Element
            List of all the Elements which have a nature 'nature'.
        """
        if sub_list is None:
            sub_list = self.elements['list']
        list_of = list(filter(lambda elt: elt.info['nature'] == nature,
                              sub_list))
        return list_of

    def where_is(self, elt, nature=False):
        """
        Determine where is elt in list_of_elements.

        If nature = True, elt is the idx-th element of his nature.

        Parameters
        ----------
        elt : Element
            Element you want the position of.
        nature : bool, optional
            Allow to count only the elt's nature (eg QUAD). The default is
            False.

        Returns
        -------
        idx : int
            Position of elt in list_of_elements, or in the list of elements of
            it's nature if nature is True.

        """
        if nature:
            print('Warning, calling where_is with argument nature.')
            idx = self.elements_of(nature=elt.info['nature']).index(elt)
        else:
            print('Warning, where_is should be useless now.')
            idx = self.elements['list'].index(elt)

        return idx

    def where_is_this_index(self, idx, showinfo=False):
        """Give an equivalent index."""
        for elt in self.elements['list']:
            if idx in range(elt.idx['s_in'], elt.idx['s_out']):
                break
        if showinfo:
            print('Synch index', idx, 'is in:', elt.info)
            print('Indexes of this elt:', elt.idx, '\n\n')
        return elt

    def compute_mismatch(self, ref_linac):
        """Compute mismatch factor between this non-nominal linac and a ref."""
        assert self.name != 'Working'
        assert ref_linac.name == 'Working'
        self.beam_param["mismatch factor"] = \
                mismatch_factor(ref_linac.beam_param["twiss"]["z"],
                                self.beam_param["twiss"]["z"], transp=True)


def _prepare_sections_and_lattices(l_elts):
    """
    Save info on the accelerator structure.

    In particular: in every section, the number of elements per lattice,
    the rf frequency of cavities, the position of the delimitations between
    two sections.
    """
    lattices = [
        lattice
        for lattice in l_elts
        if isinstance(lattice, elements.Lattice)
    ]
    frequencies = [
        frequency
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


def _sections_lattices(l_elts):
    """Gather elements by section and lattice."""
    l_elts, dict_struct = _prepare_sections_and_lattices(l_elts)

    lattices = []
    sections = []
    j = 0
    for i in range(dict_struct['n_sections']):
        lattices = []
        n_lattice = dict_struct['lattices'][i].n_lattice
        while j < dict_struct['indices'][i]:
            lattices.append(l_elts[j:j + n_lattice])
            j += n_lattice
        sections.append(lattices)

    shift_lattice = 0
    for i, sec in enumerate(sections):
        for j, lattice in enumerate(sec):
            for k, elt in enumerate(lattice):
                elt.idx['section'] = (i, j, k)
                elt.idx['lattice'] = (j + shift_lattice, k)
                elt.idx['element'] = l_elts.index(elt)
        shift_lattice += j + 1
    lattices = []
    for sec in sections:
        lattices += sec
    return l_elts, sections, lattices, dict_struct['frequencies']
