#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a base class for :class:`.Variable` and :class:`.Constraint`."""
from abc import ABC
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from util.dicts_output import markdown


@dataclass
class DesignSpaceParameter(ABC):
    """Hold a single variable or constraint."""

    name: str
    element_name: str
    limits: tuple[float, float]

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        self._to_deg = False
        self._to_numpy = False

    @property
    def _fmt_limits(self) -> Sequence[float]:
        """Limits but with a better output."""
        if 'phi' in self.name:
            return np.rad2deg(self.limits)
        return self.limits

    @property
    def _fmt_x_0(self) -> float:
        """Initial value but with a better output."""
        assert hasattr(self, 'x_0'), "This design space parameter has no " \
            "attribute x_0. Maybe you took a Contraint for a Variable?"
        x_0 = getattr(self, 'x_0')
        if 'phi' in self.name:
            return np.rad2deg(x_0)
        return x_0

    def __str__(self) -> str:
        """Output parameter name and limits."""
        out = f"{markdown[self.name]:25} | {self.element_name:15} | "
        if hasattr(self, 'x_0'):
            out += f"{self._fmt_x_0:>8.3f} | "
        else:
            out += "{' ':<8} | "
        out += f"limits={self._fmt_limits[0]:>9.3f} "
        out += f"{self._fmt_limits[1]:>9.3f}"

        return out

    @classmethod
    def header_of__str__(cls) -> str:
        """Give information on what :func:`__str__` is about."""
        header = f"{cls.__name__:<25} | {'Element':<15} | {'x_0':<8} | "
        header += f"{'Lower lim':<9} | {'Upper lim':<9}"
        return header

    def to_dict(self,
                *to_get: str,
                missing_value: float | None = None,
                prepend_parameter_name: bool = False
                ) -> dict[str, float | None | tuple[float, float] | str]:
        """Convert important data to dict to convert it later in pandas df."""
        out = {attribute: getattr(self, attribute, missing_value)
               for attribute in to_get}
        if not prepend_parameter_name:
            return out
        return {f"{self.name}: {key}": value for key, value in out.items()}
