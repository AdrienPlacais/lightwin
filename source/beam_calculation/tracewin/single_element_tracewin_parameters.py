#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:31:18 2023.

@author: placais

This module holds `SingleElementTraceWinParameters`, that inherits from the
Abstract Base Class `SingleElementCalculatorParameters`.
It is currently not necessary.

"""
import logging
import numpy as np

from beam_calculation.single_element_beam_calculator_parameters import (
    SingleElementCalculatorParameters)


class SingleElementTraceWinParameters(SingleElementCalculatorParameters):
    """
    Holds meshing and indexes of _Elements.

    Unnecessary for TraceWin, but useful to link the meshing in TraceWin to
    other simulations. Hence, it is not created by the init_solver_parameters
    as for Envelope1D!!
    Instead, meshing is deducted from the TraceWin output files.
    """

    def __init__(self, length_m: float, z_of_this_element_from_tw: np.ndarray,
                 s_in: int, s_out: int) -> None:
        self.n_steps = z_of_this_element_from_tw.shape[0]
        self.abs_mesh = z_of_this_element_from_tw
        self.rel_mesh = self.abs_mesh - self.abs_mesh[0]

        if np.abs(length_m - self.rel_mesh[-1]) > 1e-2:
            logging.error("Mismatch between length of the linac in the `.out` "
                          "file and what is expected. Maybe an error was "
                          "raised during execution of `TraceWin` and the "
                          "`.out` file is incomplete? In this case, check "
                          "`_add_dummy_data` in `tracewin` module.")

        self.s_in = s_in
        self.s_out = s_out

    def re_set_for_broken_cavity(self) -> None:
        pass

    def transf_mat_function_wrapper(self, *args, **kwargs) -> dict:
        raise NotImplementedError("maybe should be @abstractmethod also.")
