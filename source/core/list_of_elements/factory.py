"""Define an object to create :class:`.ListOfElements`.

Its main goal is to initialize :class:`.ListOfElements` with the proper input
synchronous particle and beam properties.
:meth:`.whole_list_run` is called within the :class:`.Accelerator` and generate
a full :class:`.ListOfElements` from scratch.

:meth:`.subset_list_run` is called within :class:`.Fault` and generates a
:class:`.ListOfElements` that contains only a fraction of the linac.

.. todo::
    Also handle ``.dst`` file in :meth:`.subset_list_run`.

.. todo::
    Maybe it will be necessary to handle cases where the synch particle is not
    perfectly on the axis?

.. todo::
    Find a smart way to sublass :class:`.ListOfElementsFactory` according to
    the :class:`.BeamCalculator`... Loading field maps not necessary with
    :class:`.TraceWin` for example.

.. todo::
    The ``elements_to_dump`` key should be in the configuration file

"""

import logging
from abc import ABCMeta
from pathlib import Path
from typing import Any

import numpy as np

import tracewin_utils.load
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.beam_parameters.factory import InitialBeamParametersFactory
from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap
from core.instruction import Instruction
from core.instructions_factory import InstructionsFactory
from core.list_of_elements.list_of_elements import ListOfElements
from core.particle import ParticleInitialState
from tracewin_utils.dat_files import (
    dat_filecontent_from_smaller_list_of_elements,
    export_dat_filecontent,
)


class ListOfElementsFactory:
    """Factory class to create list of elements from different contexts."""

    def __init__(
        self,
        is_3d: bool,
        is_multipart: bool,
        freq_bunch_mhz: float,
        default_field_map_folder: Path,
        load_field_maps: bool = True,
        field_maps_in_3d: bool = False,
        load_cython_field_maps: bool = False,
        elements_to_dump: ABCMeta | tuple[ABCMeta, ...] = (),
    ):
        """Declare and create some mandatory factories.

        .. note::
            For now, we have only one ``input_beam`` parameters, we create only
            one :class:`.ListOfElements`. Hence we create in the most general
            way possible.
            We instantiate :class:`.InitialBeamParametersFactory` with
            ``is_3d=True`` and ``is_multipart=True`` because it will work with
            all the :class:`.BeamCalculator` objects -- some phase-spaces may
            be created but never used though.

        Parameters
        ----------
        phi_s_definition : str, optional
            Definition for the synchronous phases that will be used. Allowed
            values are in
            :var:`util.synchronous_phases.SYNCHRONOUS_PHASE_FUNCTIONS`. The
            default is ``'historical'``.

        """
        self.initial_beam_factory = InitialBeamParametersFactory(
            # Useless with Envelope1D
            is_3d=True,
            # Useless with Envelope1D, Envelope3D, TraceWin if partran = 0
            is_multipart=True,
        )

        self.instructions_factory = InstructionsFactory(
            freq_bunch_mhz,
            default_field_map_folder,
            load_field_maps,
            field_maps_in_3d,
            load_cython_field_maps,
            elements_to_dump=elements_to_dump,
        )

    def whole_list_run(
        self,
        dat_file: Path,
        accelerator_path: Path,
        **kwargs: Any,
    ) -> ListOfElements:
        """Create a new :class:`.ListOfElements`, encompassing a full linac.

        Factory function called from within the :class:`.Accelerator` object.

        Parameters
        ----------
        dat_file : Path
            Absolute path to the ``.dat`` file.
        accelerator_path : Path
            Absolute path where results for each :class:`.BeamCalculator` will
            be stored.

        Returns
        -------
        list_of_elements : ListOfElements
            Contains all the :class:`.Elements` of the linac, as well as the
            proper particle and beam properties at its entry.

        """
        logging.info(
            "First initialisation of ListOfElements, ecompassing all linac. "
            f"Created with {dat_file = }"
        )

        dat_filecontent = tracewin_utils.load.dat_file(dat_file, keep="all")
        files = {
            "dat_file": dat_file,
            "dat_content": dat_filecontent,
            "accelerator_path": accelerator_path,
            "elts_n_cmds": list[Instruction],
        }

        instructions = self.instructions_factory.run(dat_filecontent)
        elts = [x for x in instructions if isinstance(x, Element)]

        files["elts_n_cmds"] = instructions

        input_particle = self._whole_list_input_particle(**kwargs)
        input_beam = self.initial_beam_factory.factory_new(
            sigma_in=kwargs["sigma_in"], w_kin=kwargs["w_kin"]
        )

        tm_cumul_in = np.eye(6)
        list_of_elements = ListOfElements(
            elts=elts,
            input_particle=input_particle,
            input_beam=input_beam,
            tm_cumul_in=tm_cumul_in,
            files=files,
            first_init=True,
        )
        return list_of_elements

    def _whole_list_input_particle(
        self, w_kin: float, phi_abs: float, z_in: float, **kwargs: np.ndarray
    ) -> ParticleInitialState:
        """Create a :class:`.ParticleInitialState` for full list of elts."""
        input_particle = ParticleInitialState(
            w_kin=w_kin,
            phi_abs=phi_abs,
            z_in=z_in,
            synchronous=True,
        )
        return input_particle

    def subset_list_run(
        self,
        elts: list[Element],
        simulation_output: SimulationOutput,
        files_from_full_list_of_elements: dict[
            str, Path | str | list[list[str]]
        ],
    ) -> ListOfElements:
        """
        Create a :class:`.ListOfElements` which is a subset of a previous one.

        Factory function used during the fitting process, called by a
        :class:`.Fault` object. During this optimisation process, we compute
        the propagation of the beam only on the smallest possible subset of the
        linac.

        It creates the proper :class:`.ParticleInitialState` and
        :class:`.BeamParameters` objects. In contrary to
        :func:`new_list_of_elements`, the :class:`.BeamParameters` must contain
        information on the transverse plane if beam propagation is performed
        with :class:`.TraceWin`.

        Parameters
        ----------
        elts : list[Element]
            A plain list containing the elements objects that the object should
            contain.
        simulation_output : SimulationOutput
            Holds the results of the pre-existing list of elements.
        files_from_full_list_of_elements : dict
            The :attr:`.ListOfElements.files` from the full
            :class:`.ListOfElements`.

        Returns
        -------
        list_of_elements : ListOfElements
            Contains all the elements that will be recomputed during the
            optimisation, as well as the proper particle and beam properties at
            its entry.

        """
        logging.info(
            "Initalisation of ListOfElements from already initialized"
            f" elements: {elts[0]} to {elts[-1]}."
        )

        input_elt, input_pos = self._get_initial_element(
            elts, simulation_output
        )
        get_kw = {
            "elt": input_elt,
            "pos": input_pos,
            "to_numpy": False,
        }
        input_particle = self._subset_input_particle(
            simulation_output, **get_kw
        )
        input_beam = self.initial_beam_factory.factory_subset(
            simulation_output, get_kw
        )

        files = self._subset_files_dictionary(
            elts,
            files_from_full_list_of_elements,
        )

        transfer_matrix = simulation_output.transfer_matrix
        assert transfer_matrix is not None
        tm_cumul_in = transfer_matrix.cumulated[0]

        list_of_elements = ListOfElements(
            elts=elts,
            input_particle=input_particle,
            input_beam=input_beam,
            tm_cumul_in=tm_cumul_in,
            files=files,
            first_init=False,
        )

        return list_of_elements

    def _shift_entry_phase(
        self, cavities: list[FieldMap], delta_phi: float
    ) -> None:
        """Shift the entry phase in the given cavities.

        This is mandatory for TraceWin solver, as the absolute phase at the
        entrance of the first element is always 0.0.

        """
        pass

    def _subset_files_dictionary(
        self,
        elts: list[Element],
        files_from_full_list_of_elements: dict[str, Any],
        tmp_folder: Path | str = Path("tmp"),
        tmp_dat: Path | str = Path("tmp.dat"),
    ) -> dict[str, Path | list[list[str]]]:
        """Set the new ``.dat`` file containing only elements of ``elts``."""
        accelerator_path = files_from_full_list_of_elements["accelerator_path"]
        out = accelerator_path / tmp_folder
        out.mkdir(exist_ok=True)
        dat_file = out / tmp_dat

        original_instructions = files_from_full_list_of_elements["elts_n_cmds"]
        assert isinstance(original_instructions, list)
        dat_content, instructions = (
            dat_filecontent_from_smaller_list_of_elements(
                files_from_full_list_of_elements["elts_n_cmds"],
                elts,
            )
        )

        files = {
            "dat_file": dat_file,
            "dat_content": dat_content,
            "elts_n_cmds": instructions,
            "accelerator_path": accelerator_path / tmp_folder,
        }

        export_dat_filecontent(dat_content, dat_file)
        return files

    def _delta_phi_for_tracewin(
        self, phi_at_entry_of_compensation_zone: float
    ) -> float:
        """
        Give new absolute phases for :class:`.TraceWin`.

        In TraceWin, the absolute phase at the entrance of the compensation
        zone is 0, while it is not in the rest of the code. Hence we must
        rephase the cavities in the subset.

        """
        phi_at_linac_entry = 0.0
        delta_phi_bunch = (
            phi_at_entry_of_compensation_zone - phi_at_linac_entry
        )
        return delta_phi_bunch

    def _get_initial_element(
        self, elts: list[Element], simulation_output: SimulationOutput
    ) -> tuple[Element | str, str]:
        """Set the element from which we should take energy, phase, etc."""
        input_elt, input_pos = elts[0], "in"
        try:
            _ = simulation_output.get("w_kin", elt=input_elt)
        except AttributeError:
            logging.warning(
                "First element of new list of elements is not in "
                "the given SimulationOutput. I will consider "
                "that the last element of the SimulationOutput is "
                "the first of the new ListOfElements."
            )
            input_elt, input_pos = "last", "out"
        return input_elt, input_pos

    def _subset_input_particle(
        self, simulation_output: SimulationOutput, **kwargs: Any
    ) -> ParticleInitialState:
        """Create input particle for subset of list of elements."""
        w_kin, phi_abs, z_abs = simulation_output.get(
            "w_kin", "phi_abs", "z_abs", **kwargs
        )
        input_particle = ParticleInitialState(
            w_kin, phi_abs, z_abs, synchronous=True
        )
        return input_particle
