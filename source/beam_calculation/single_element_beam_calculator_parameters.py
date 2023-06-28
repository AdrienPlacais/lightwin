#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:29:12 2023.

@author: placais

This module holds a base class for SingleElementCalculatorParameters (attribute
of an _Element).
"""
from typing import Any
from abc import ABC, abstractmethod

import numpy as np

from util.helper import recursive_items, recursive_getter


class SingleElementCalculatorParameters(ABC):
    """
    Parent class to hold solving parameters. Attribute of _Element.

    As for now, only used by Envelope1D. Useful for TraceWin? To store its
    meshing maybe?
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
        """Update solver after a cavity is  broken."""
