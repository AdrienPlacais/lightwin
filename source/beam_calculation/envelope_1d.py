#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:24:37 2023.

@author: placais

"""
import logging
from functools import partial
from typing import Callable
from dataclasses import dataclass

from core.particle import ParticleFullTrajectory
from core.elements import _Element
from core.list_of_elements import (ListOfElements, indiv_to_cumul_transf_mat,
                                   equiv_elt)
from core.accelerator import Accelerator
from core.beam_parameters import BeamParameters

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.output import SimulationOutput
from beam_calculation.single_element_envelope_1d_parameters import (
    SingleElementEnvelope1DParameters)

from optimisation.set_of_cavity_settings import SetOfCavitySettings


@dataclass
class Envelope1D(BeamCalculator):
    """The fastest beam calculator, adapted to high energies."""

    flag_phi_abs: bool
    flag_cython: bool
    n_steps_per_cell: int
    method: str

    def __post_init__(self):
        """Set the proper motion integration function, according to inputs."""
        self.id = self.__repr__()
        if self.flag_cython:
            try:
                import core.transfer_matrices_c as transf_mat
            except ModuleNotFoundError:
                logging.error("Cython version of transfer_matrices was not "
                              + "compiled. Check util/setup.py.")
                raise ModuleNotFoundError("Cython not compiled.")
        else:
            import core.transfer_matrices_p as transf_mat
        self.transf_mat_module = transf_mat

    def run(self, elts: ListOfElements) -> SimulationOutput:
        """
        Compute beam propagation in 1D, envelope calculation.

        Parameters
        ----------
        elts : ListOfElements
            List of elements in which the beam must be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        return self.run_with_this(set_of_cavity_settings=None, elts=elts)

    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings | None,
                      elts: ListOfElements) -> SimulationOutput:
        """
        Envelope 1D calculation of beam in `elts`, with non-nominal settings.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings | None
            The new cavity settings to try. If it is None, then the cavity
            settings are taken from the FieldMap objects.
        elts : ListOfElements
            List of elements in which the beam must be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        single_elts_results = []
        rf_fields = []

        w_kin = elts.w_kin_in
        phi_abs = elts.phi_abs_in

        for elt in elts:
            cavity_settings = set_of_cavity_settings.get(elt) \
                if isinstance(set_of_cavity_settings, SetOfCavitySettings) \
                else None

            rf_field_kwargs = elt.rf_param(self.id, phi_abs, w_kin,
                                           cavity_settings)
            elt_results = \
                elt.beam_calc_param[self.id].transf_mat_function_wrapper(
                    w_kin, elt.is_accelerating(), elt.get('status'),
                    **rf_field_kwargs)

            single_elts_results.append(elt_results)
            rf_fields.append(rf_field_kwargs)

            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        simulation_output = self._generate_simulation_output(
            elts, single_elts_results, rf_fields)
        return simulation_output

    def init_solver_parameters(self, accelerator: Accelerator) -> None:
        """
        Create the number of steps, meshing, transfer functions for elts.

        The solver parameters are stored in self.parameters. As for now, for
        memory purposes, only one set of solver parameters is stored. In other
        words, if you compute the transfer matrices of several ListOfElements
        back and forth, the solver paramters will be re-initialized each time.

        Parameters
        ----------
        accelerator : Accelerator
            Accelerator object which ListOfElements must be initialized.

        """
        elts = accelerator.elts
        kwargs = {'n_steps_per_cell': self.n_steps_per_cell,
                  'method': self.method,
                  'transf_mat_module': self.transf_mat_module,
                  }
        for elt in elts:
            elt.beam_calc_param[self.id] = SingleElementEnvelope1DParameters(
                length_m=elt.get('length_m', to_numpy=False),
                is_accelerating=elt.is_accelerating(),
                n_cells=elt.get('n_cell', to_numpy=False),
                **kwargs)

        position = 0.
        index = 0
        for elt in elts:
            position, index = \
                elt.beam_calc_param[self.id].set_absolute_meshes(position,
                                                                 index)

    def _generate_simulation_output(self, elts: ListOfElements,
                                    single_elts_results: list[dict],
                                    rf_fields: list[dict]
                                    ) -> SimulationOutput:
        """Transform the outputs of BeamCalculator to a SimulationOutput."""
        w_kin = [energy
                 for results in single_elts_results
                 for energy in results['w_kin']
                 ]
        w_kin.insert(0, elts.w_kin_in)

        phi_abs_array = [elts.phi_abs_in]
        for elt_results in single_elts_results:
            phi_abs = [phi_rel + phi_abs_array[-1]
                       for phi_rel in elt_results['phi_rel']]
            phi_abs_array.extend(phi_abs)
        synch_trajectory = ParticleFullTrajectory(w_kin=w_kin,
                                                  phi_abs=phi_abs_array,
                                                  synchronous=True)

        cav_params = [results['cav_params']
                      for results in single_elts_results]
        phi_s = [cav_param['phi_s']
                 for cav_param in cav_params if cav_param is not None]

        r_zz_elt = [
            results['r_zz'][i, :, :]
            for results in single_elts_results
            for i in range(results['r_zz'].shape[0])
        ]
        tm_cumul = indiv_to_cumul_transf_mat(elts.tm_cumul_in, r_zz_elt,
                                             len(w_kin))

        beam_params = BeamParameters(tm_cumul)

        element_to_index = self._generate_element_to_index_func(elts)

        simulation_output = SimulationOutput(
            synch_trajectory=synch_trajectory,
            cav_params=cav_params,
            phi_s=phi_s,
            r_zz_elt=r_zz_elt,
            rf_fields=rf_fields,
            beam_parameters=beam_params,
            element_to_index=element_to_index
        )
        return simulation_output

    def _generate_element_to_index_func(self, elts: ListOfElements
                                        ) -> Callable[[_Element, str | None],
                                                      int | slice]:
        """Create the func to easily get data at proper mesh index."""
        shift = elts[0].beam_calc_param[self.id].s_in
        return partial(_element_to_index, _elts=elts, _shift=shift,
                       _solver_id=self.id)


def _element_to_index(_elts: ListOfElements, _shift: int,
                      _solver_id: str, elt: _Element | str,
                      pos: str | None = None) -> int | slice:
    """
    Convert element + pos into a mesh index.

    Parameters
    ----------
    _elts : ListOfElements
        List of Elements where elt should be. Must be set by a
        functools.partial.
    shift : int
        Mesh index of first _Element. Used when the first _Element of _elts is
        not the first of the Accelerator. Must be set by functools.partial.
    elt : _Element | str
        Element of which you want the index.
    pos : 'in' | 'out' | None, optional
        Index of entry or exit of the _Element. If None, return full
        indexes array. The default is None.

    """
    if isinstance(elt, str):
        elt = equiv_elt(elts=_elts, elt=elt)
    elif elt not in _elts:
        logging.warning(f"Required element {elt} belongs to another "
                        "ListOfElements, which is questionable in this "
                        "context.")

    beam_calc_param = elt.beam_calc_param[_solver_id]
    if pos is None:
        return slice(beam_calc_param.s_in - _shift,
                     beam_calc_param.s_out - _shift + 1)
    elif pos == 'in':
        return beam_calc_param.s_in - _shift
    elif pos == 'out':
        return beam_calc_param.s_out - _shift
    else:
        logging.error(f"{pos = }, while it must be 'in', 'out' or None")
