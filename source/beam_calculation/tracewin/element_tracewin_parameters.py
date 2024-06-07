"""Store the solver parameters for :class:`.TraceWin`.

.. note::
    TraceWin does not need this to run. It is a placeholder, to keep
    consistency with :class:`.Envelope1D` and :class:`.Envelope3D`.
    Also useful to compare :class:`.SimulationOutput` created by this solver
    and others.

"""

import logging
from typing import Any, Sequence

import numpy as np

from beam_calculation.parameters.element_parameters import (
    ElementBeamCalculatorParameters,
)


class ElementTraceWinParameters(ElementBeamCalculatorParameters):
    """Hold meshing and indexes of elements.

    Unnecessary for TraceWin, but useful to link the meshing in TraceWin to
    other simulations. Hence, it is not created by the init_solver_parameters
    as for Envelope1D!!
    Instead, meshing is deducted from the TraceWin output files.

    """

    def __init__(
        self,
        length_m: float,
        z_of_this_element_from_tw: np.ndarray,
        s_in: int,
        s_out: int,
    ) -> None:
        """Instantiate object.

        Parameters
        ----------
        self :
            self
        length_m : float
            length_m
        z_of_this_element_from_tw : np.ndarray
            z_of_this_element_from_tw
        s_in : int
            s_in
        s_out : int
            s_out

        """
        self.n_steps = z_of_this_element_from_tw.shape[0]
        self.abs_mesh = z_of_this_element_from_tw
        self.rel_mesh = self.abs_mesh - self.abs_mesh[0]

        if np.abs(length_m - self.rel_mesh[-1]) > 1e-2:
            logging.debug(
                "Mismatch between length of the linac in the .out file and "
                "what is expected. Maybe an error was raised during execution "
                "of TraceWin and the .out file is incomplete? In this case, "
                "check _add_dummy_data in tracewin module."
            )

        self.s_in = s_in
        self.s_out = s_out

    def re_set_for_broken_cavity(self) -> None:
        """Do nothing."""
        pass

    def transf_mat_function_wrapper(self, *args, **kwargs) -> dict:
        """Do nothing."""
        raise NotImplementedError("maybe should be @abstractmethod also.")

    def _transfer_matrix_arguments(self) -> Sequence[Any]:
        """Do nothing."""
        return None, None
