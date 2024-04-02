#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a base class for :class:`ElementBeamCalculatorParameters`.

It is an attribute of an :class:`Element`, and holds parameters that depend on
both the :class:`Element` under study and the BeamCalculator solver that is
used.

Currently, it is used by :class:`Envelope1D` only, as :class:`TraceWin` handles
it itself.

"""
from typing import Any
from abc import ABC, abstractmethod

import numpy as np

from util.helper import recursive_items, recursive_getter


class ElementBeamCalculatorParameters(ABC):
    """Parent class to hold solving parameters. Attribute of :class:`.Element`.

    Used by :class:`.Envelope1D` and :class:`.Envelope3D`.

    """

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True,
            **kwargs: dict) -> Any:
        """Shorthand to get attributes."""
        val = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

        out = [np.array(val[key]) if to_numpy and not isinstance(val[key], str)
               else val[key]
               for key in keys]
        if len(out) == 1:
            return out[0]

        return tuple(out)

    @abstractmethod
    def re_set_for_broken_cavity(self) -> None:
        """Update solver after a cavity is broken."""
