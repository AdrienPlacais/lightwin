#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds class:`Variable`, which stores an optimisation variable.

It keeps it's name, bounds, initial value, etc.

"""
import logging
from dataclasses import dataclass

from optimisation.design_space.design_space_parameter import \
    DesignSpaceParameter


IMPLEMENTED_VARIABLES = ('k_e', 'phi_0_abs', 'phi_0_rel', 'phi_s')  #:


@dataclass
class Variable(DesignSpaceParameter):
    """
    A single variable.

    It can be a cavity amplitude, absolute phase, relative phase or synchronous
    phase with an initial value and limits.

    """

    x_0: float

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        if self.name not in IMPLEMENTED_VARIABLES:
            logging.warning(f"Variable {self.name} not tested.")
        super().__post_init__()


