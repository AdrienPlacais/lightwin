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

    def str_for_file(self, fmt: str = '.5f') -> list[str]:
        """Give what should be written in the linac design-space file."""
        data = [self.limits[0], self.x_0, self.limits[1]]
        return [f"{number:{fmt}}" for number in data]

    def str_header1_for_file(self) -> list[str]:
        """Give the first line header of the linac design-space file."""
        return [self.name, self.name, self.name]

    def str_header2_for_file(self) -> list[str]:
        """Give the second line header of the linac design-space file."""
        return ['min', 'x_0', 'max']
