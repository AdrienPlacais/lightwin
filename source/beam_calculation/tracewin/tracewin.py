#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we define :class:`TraceWin`, to call it from command line.

It inherits from :class:`.BeamCalculator` base class.  It solves the motion of
the particles in envelope or multipart, in 3D. In contrary to
:class:`.Envelope1D` solver, it is not a real solver but an interface with
``TraceWin`` which must be installed on your machine.

.. warning::
    For now, :class:`TraceWin` behavior with relative phases is undetermined.
    You should ensure that you are working with *absolute* phases, i.e. that
    last argument of ``FIELD_MAP`` commands is ``1``.
    You can run a simulation with :class:`.Envelope1D` solver and
    ``flag_phi_abs= True``. The ``.dat`` file created in the ``000001_ref``
    folder should be the original ``.dat`` but converted to absolute phases.

.. todo::
    Already written elsewhere in the code, but a script to convert ``.dat``
    between the different phases would be good.

.. todo::
    Allow TW to work with relative phases. Will have to handle ``rf_fields``
    too.

"""
import logging
from pathlib import Path
import shutil
import subprocess

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from beam_calculation.tracewin.element_tracewin_parameters_factory import (
    ElementTraceWinParametersFactory)
from beam_calculation.tracewin.simulation_output_factory import (
    SimulationOutputFactoryTraceWin,
)
from core.accelerator.accelerator import Accelerator
from core.elements.field_maps.field_map import FieldMap
from core.list_of_elements.list_of_elements import ListOfElements
from failures.set_of_cavity_settings import SetOfCavitySettings
from tracewin_utils.interface import beam_calculator_to_command


class TraceWin(BeamCalculator):
    """
    A class to hold a TW simulation and its results.

    The simulation is not necessarily runned.

    Attributes
    ----------
    executable : str
        Path to the TraceWin executable.
    ini_path : str
        Path to the ``.ini`` TraceWin file.
    base_kwargs : dict[str, str | bool | int | None | float]
        TraceWin optional arguments. Override what is defined in ``.ini``, but
        overriden by arguments from :class:`ListOfElements` and
        :class:`SimulationOutput`.
    _tracewin_command : list[str] | None, optional
        Attribute to hold the value of the base command to call TraceWin.
    id : str
        Complete name of the solver.
    out_folder : str
        Name of the results folder (not a complete path, just a folder name).
    load_results : Callable
        Function to call to get the output results.
    path_cal : str
        Name of the results folder. Updated at every call of the
        :func:`init_solver_parameters` method, using
        ``Accelerator.accelerator_path`` and ``self.out_folder`` attributes.
    dat_file :
        Base name for the ``.dat`` file. ??

    """

    def __init__(self,
                 executable: Path,
                 ini_path: Path,
                 base_kwargs: dict[str, str | int | float | bool | None],
                 out_folder: Path | str,
                 default_field_map_folder: Path | str,
                 cal_file: Path | None = None,
                 ) -> None:
        """Define some other useful methods, init variables."""
        self.executable = executable
        self.ini_path = ini_path.resolve().absolute()
        self.base_kwargs = base_kwargs
        self.cal_file = cal_file

        filename = Path('tracewin.out')
        if self.is_a_multiparticle_simulation:
            filename = Path('partran1.out')
        self._filename = filename
        super().__init__(out_folder, default_field_map_folder)

        logging.warning("TraceWin solver currently cannot work with relative "
                        "phases (last arg of FIELD_MAP should be 1). You "
                        "should check this, because I will not.")
        self.path_cal: Path
        self.dat_file: Path
        self._tracewin_command: list[str] | None = None

    def _set_up_specific_factories(self) -> None:
        """Set up the factories specific to the :class:`.BeamCalculator`.

        This method is called in the :meth:`super().__post_init__`, hence it
        appears only in the base :class:`.BeamCalculator`.

        """
        self.beam_calc_parameters_factory = ElementTraceWinParametersFactory()

        self.simulation_output_factory = SimulationOutputFactoryTraceWin(
            self.is_a_3d_simulation,
            self.is_a_multiparticle_simulation,
            self.id,
            self.out_folder,
            self._filename,
            self.beam_calc_parameters_factory,
        )

    def _tracewin_base_command(self,
                               accelerator_path: Path,
                               **kwargs
                               ) -> tuple[list[str], Path]:
        """Define the 'base' command for TraceWin.

        This part of the command is the same for every :class:`ListOfElements`
        and every :class:`Fault`. It sets the TraceWin executable, the ``.ini``
        file.  It also defines ``base_kwargs``, which should be the same for
        every calculation.
        Finally, it sets ``path_cal``.
        But this path is more :class:`ListOfElements`
        dependent...
        ``Accelerator.accelerator_path`` + ``out_folder``
            (+ ``fault_optimisation_tmp_folder``)

        """
        kwargs = kwargs.copy()
        for key, val in self.base_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val

        path_cal = accelerator_path / self.out_folder
        if not path_cal.is_dir():
            path_cal.mkdir()

        _tracewin_command = beam_calculator_to_command(
            self.executable,
            self.ini_path,
            path_cal,
            **kwargs,
        )
        return _tracewin_command, path_cal

    def _tracewin_full_command(
        self,
        elts: ListOfElements,
        set_of_cavity_settings: SetOfCavitySettings | None,
        **kwargs,
    ) -> tuple[list[str], Path]:
        """Set the full TraceWin command.

        It contains the 'base' command, which includes every argument that is
        common to every calculation with this :class:`BeamCalculator`: path to
        ``.ini`` file, to executable...

        It contains the :class:`ListOfElements` command: path to the ``.dat``
        file, initial energy and beam properties.

        It can contain some :class:`SetOfCavitySettings` commands: ``ele``
        arguments to modify some cavities tuning.

        """
        accelerator_path = elts.files['accelerator_path']
        assert isinstance(accelerator_path, Path)
        command, path_cal = self._tracewin_base_command(accelerator_path,
                                                        **kwargs)
        command.extend(elts.tracewin_command)

        if set_of_cavity_settings is not None:
            command.extend(set_of_cavity_settings.tracewin_command(
                delta_phi_bunch=elts.input_particle.phi_abs
            ))
        return command, path_cal

    # TODO what is specific_kwargs for? I should just have a function
    # set_of_cavity_settings_to_kwargs
    def run(self, elts: ListOfElements, **specific_kwargs) -> None:
        """
        Run TraceWin.

        Parameters
        ----------
        elts : ListOfElements
        List of :class:`Element` s in which you want the beam propagated.
        **specific_kwargs : dict
            ``TraceWin`` optional arguments. Overrides what is defined in
            ``base_kwargs`` and ``.ini``.

        """
        return self.run_with_this(set_of_cavity_settings=None, elts=elts,
                                  **specific_kwargs)

    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings | None,
                      elts: ListOfElements,
                      **specific_kwargs
                      ) -> SimulationOutput:
        """
        Perform a simulation with new cavity settings.

        Calling it with ``set_of_cavity_settings = None`` is the same as
        calling the plain :func:`run` method.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings | None
            Holds the norms and phases of the compensating cavities.
        elts : ListOfElements
            List of elements in which the beam should be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        if specific_kwargs not in (None, {}):
            logging.critical(f"{specific_kwargs = }: deprecated.")

        if specific_kwargs is None:
            specific_kwargs = {}

        rf_fields = []
        for elt in elts:
            if isinstance(set_of_cavity_settings, SetOfCavitySettings):
                cavity_settings = set_of_cavity_settings.get(elt)
                if cavity_settings is not None:
                    rf_fields.append({'k_e': cavity_settings.k_e,
                                      'phi_0_abs': cavity_settings.phi_0_abs,
                                      'phi_0_rel': cavity_settings.phi_0_rel})
                    continue

            if isinstance(elt, FieldMap):
                rf_fields.append(
                    {'k_e': elt.rf_field.k_e,
                     'phi_0_abs': elt.rf_field.phi_0['phi_0_abs'],
                     'phi_0_rel': elt.rf_field.phi_0['phi_0_rel']})
                continue
            rf_fields.append({})

        command, path_cal = self._tracewin_full_command(
            elts,
            set_of_cavity_settings,
            **specific_kwargs)
        is_a_fit = set_of_cavity_settings is not None
        exception = _run_in_bash(command, output_command=not is_a_fit)

        simulation_output = self._generate_simulation_output(elts, path_cal,
                                                             rf_fields,
                                                             exception)
        return simulation_output

    def post_optimisation_run_with_this(
        self,
        optimized_cavity_settings: SetOfCavitySettings,
        full_elts: ListOfElements,
        **specific_kwargs
    ) -> SimulationOutput:
        """
        Run TraceWin with optimized cavity settings.

        After the optimisation, we want to re-run TraceWin with the new
        settings. However, we need to tell it that the linac is bigger than
        during the optimisation. Concretely, it means:
            * rephasing the cavities in the compensation zone
            * updating the ``index`` ``n`` of the cavities in the ``ele[n][v]``
              command.

        Note that at this point, the ``.dat`` has not been updated yet.

        Parameters
        ----------
        optimized_cavity_settings : SetOfCavitySettings
            Optimized parameters.
        full_elts : ListOfElements
            Contains the full linac.

        Returns
        -------
        simulation_output : SimulationOutput
            Necessary information on the run.

        """
        optimized_cavity_settings.re_set_elements_index_to_absolute_value()
        full_elts.store_settings_in_dat(full_elts.files['dat_filepath'])

        simulation_output = self.run_with_this(optimized_cavity_settings,
                                               full_elts,
                                               **specific_kwargs)
        return simulation_output

    def init_solver_parameters(self, accelerator: Accelerator) -> None:
        """
        Set the ``path_cal`` variable.

        We also set the ``_tracewin_command`` attribute to None, as it must be
        updated when ``path_cal`` changes.

        .. note::
            In contrary to :class:`.Element1D` and :class:`.Element3D`, this
            routine does not set parameters for the :class:`.BeamCalculator`.
            As a matter of a fact, TraceWin is a standalone code and does not
            need out solver parameters.
            However, if we want to save the meshing used by TraceWin, we will
            have to use the :class:`.ElementTraceWinParametersFactory` later.

        """
        self.path_cal = Path(accelerator.get('accelerator_path'),
                             self.out_folder)
        assert self.path_cal.is_dir()

        self._tracewin_command = None

        if self.cal_file is None:
            return
        assert self.cal_file.is_file()
        shutil.copy(self.cal_file, self.path_cal)
        logging.debug(f"Copied {self.cal_file = } in {self.path_cal = }.")

    @property
    def is_a_multiparticle_simulation(self) -> bool:
        """Tell if you should buy Bitcoins now or wait a few months."""
        if 'partran' in self.base_kwargs:
            return self.base_kwargs['partran'] == 1
        return Path(self.path_cal, 'partran1.out').is_file()

    @property
    def is_a_3d_simulation(self) -> bool:
        """Tell if the simulation is in 3D."""
        return True


# =============================================================================
# Bash
# =============================================================================
def _run_in_bash(command: list[str],
                 output_command: bool = True,
                 output_error: bool = False) -> bool:
    """Run given command in bash."""
    output = "\n\t".join(command)
    if output_command:
        logging.info(f"Running command:\n\t{output}")

    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    exception = process.wait()

    # exception = False
    # for line in process.stdout:
        # if output_error:
            # print(line)
        # exception = True

    if exception != 0 and output_error:
        logging.warning("A message was returned when executing following "
                        f"command:\n\t{stderr}")
    return exception != 0
