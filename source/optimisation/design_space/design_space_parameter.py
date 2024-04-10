#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a base class for :class:`.Variable` and :class:`.Constraint`."""
import math
from abc import ABC
from dataclasses import dataclass
from typing import Self

import pandas as pd

from util.dicts_output import markdown


@dataclass
class DesignSpaceParameter(ABC):
    """
    Hold a single variable or constraint.

    Attributes
    ----------
    name : str
        Name of the parameter. Must be compatible with the
        :meth:`.SimulationOutput.get` method, and be in
        :data:`.IMPLEMENTED_VARIABLES` or :data:`.IMPLEMENTED_CONSTRAINTS`.
    element_name : str
        Name of the element concerned by the parameter.
    limits : tuple[float, float]
        Lower and upper bound for the variable. ``np.NaN`` deactivates a bound.

    """

    name: str
    element_name: str
    limits: tuple[float, float]

    @classmethod
    def from_floats(
        cls, name: str, element_name: str, x_min: float, x_max: float
    ) -> Self:
        """Initialize object with ``x_min``, ``x_max`` instead of ``limits``.

        Parameters
        ----------
        name : str
            Name of the parameter. Must be compatible with the
            :meth:`.SimulationOutput.get` method, and be in
            :data:`.IMPLEMENTED_VARIABLES` or :data:`.IMPLEMENTED_CONSTRAINTS`.
        element_name : str
            Name of the element concerned by the parameter.
        x_min : float
            Lower limit. ``np.NaN`` to deactivate lower bound.
        x_max : float
            Upper limit. ``np.NaN`` to deactivate lower bound.

        Returns
        -------
        Self
            A DesignSpaceParameter with limits = (x_min, x_max).

        """
        return cls(name, element_name, (x_min, x_max))

    @classmethod
    def from_pd_series(
        cls, name: str, element_name: str, pd_series: pd.Series
    ) -> Self:
        """Init object from a pd series (file import)."""
        x_min = pd_series.loc[f"{name}: x_min"]
        x_max = pd_series.loc[f"{name}: x_max"]
        return cls.from_floats(name, element_name, x_min, x_max)

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        self._to_deg = False
        self._to_numpy = False

    @property
    def x_min(self) -> float:
        """Return lower variable/constraint bound."""
        return self.limits[0]

    @property
    def x_max(self) -> float:
        """Return upper variable/constraint bound."""
        return self.limits[1]

    @property
    def _fmt_x_min(self) -> float:
        """Lower limit in deg if it is has ``'phi'`` in it's name."""
        if "phi" in self.name:
            return math.degrees(self.x_min)
        return self.x_min

    @property
    def _fmt_x_max(self) -> float:
        """Lower limit in deg if it is has ``'phi'`` in it's name."""
        if "phi" in self.name:
            return math.degrees(self.x_max)
        return self.x_max

    @property
    def _fmt_x_0(self) -> float:
        """Initial value but with a better output."""
        assert hasattr(self, "x_0"), (
            "This design space parameter has no "
            "attribute x_0. Maybe you took a Contraint for a Variable?"
        )
        x_0 = getattr(self, "x_0")
        if "phi" in self.name:
            return math.degrees(x_0)
        return x_0

    def __str__(self) -> str:
        """Output parameter name and limits."""
        out = f"{markdown[self.name]:25} | {self.element_name:15} | "
        if hasattr(self, "x_0"):
            out += f"{self._fmt_x_0:>8.3f} | "
        else:
            out += "         | "
        out += f"{self._fmt_x_min:>9.3f} | {self._fmt_x_max:>9.3f}"

        return out

    @classmethod
    def header_of__str__(cls) -> str:
        """Give information on what :func:`__str__` is about."""
        header = f"{cls.__name__:<25} | {'Element':<15} | {'x_0':<8} | "
        header += f"{'Lower lim':<9} | {'Upper lim':<9}"
        return header

    def to_dict(
        self,
        *to_get: str,
        missing_value: float | None = None,
        prepend_parameter_name: bool = False,
    ) -> dict[str, float | None | tuple[float, float] | str]:
        """Convert important data to dict to convert it later in pandas df."""
        out = {
            attribute: getattr(self, attribute, missing_value)
            for attribute in to_get
        }
        if not prepend_parameter_name:
            return out
        return {f"{self.name}: {key}": value for key, value in out.items()}
