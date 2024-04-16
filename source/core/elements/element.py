#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define base :class:`Element`, declined in Drift, FieldMap, etc.

.. todo::
    clean the patch for the 'name'. my has and get methods do not work with
    @property

"""
import logging
from typing import Any

import numpy as np

from beam_calculation.parameters.element_parameters import (
    ElementBeamCalculatorParameters,
)
from core.electric_field import NewRfField
from core.elements.field_maps.cavity_settings import CavitySettings
from core.instruction import Instruction
from util.helper import recursive_getter, recursive_items


class Element(Instruction):
    """Generic element."""

    base_name = "ELT"
    increment_elt_idx = True
    increment_lattice_idx = True

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Init parameters common to all elements.

        Parameters
        ----------
        line : list[str]
            A line of the ``.dat`` file. If the element was given a name, it
            must not appear in ``line`` but rather in ``name``. First
            element of the list must be in :data:`.IMPLEMENTED_ELEMENTS`.
        dat_idx : int
            Position in the ``.dat`` file.
        name : str | None, optional
            Non-default name of the element, as given in the ``.dat`` file. The
            default is None, in which case an automatic name will be given
            later.

        """
        super().__init__(line, dat_idx, is_implemented=True, name=name)

        self.elt_info = {
            "nature": line[0],
        }
        self.length_m = 1e-3 * float(line[1])

        # By default, an element is non accelerating and has a dummy
        # accelerating field.
        self.new_rf_field = NewRfField()

        # TODO: init the indexes to -1 or something, to help type hinting
        # dict with pure type: int
        new_idx = {
            "elt_idx": -1,
            "lattice": -1,
            "idx_in_lattice": -1,
            "section": -1,
        }
        self.idx = self.idx | new_idx
        self.beam_calc_param: dict[str, ElementBeamCalculatorParameters] = {}

    def __str__(self) -> str:
        """Give the same name as TraceWin would."""
        return self.name

    def __repr__(self) -> str:
        """Give the same name as TraceWin would."""
        return str(self)

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(
        self, *keys: str, to_numpy: bool = True, **kwargs: bool | str | None
    ) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys: str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        **kwargs : bool | str | None
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}

        for key in keys:
            if key == "name":
                val[key] = self.name
                continue

            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

        out = [
            (
                np.array(val[key])
                if to_numpy and not isinstance(val[key], str)
                else val[key]
            )
            for key in keys
        ]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def keep_rf_field(self, *args, **kwargs) -> None:
        """Save data calculated by :func:`BeamCalculator.run_with_this`.

        .. deprecated:: 0.6.16
            Prefer :meth:`keep_cavity_settings`

        """
        logging.warning("prefer keep_cavity_settings")
        return self.keep_cavity_settings(*args, **kwargs)

    def keep_cavity_settings(
        self,
        cavity_settings: CavitySettings,
    ) -> None:
        """Save data calculated by :func:`BeamCalculator.run_with_this`."""
        raise NotImplementedError("Please override this method.")

    @property
    def is_accelerating(self) -> bool:
        """Say if this element is accelerating or not.

        Will return False by default.

        """
        return False

    @property
    def can_be_retuned(self) -> bool:
        """Tell if we can modify the element's tuning.

        Will return False by default.

        """
        return False

    def update_status(self, new_status: str) -> None:
        """Change the status of the element. To override."""
        if not self.can_be_retuned:
            logging.error(
                f"You want to give {new_status = } to the element "
                f"{self.name}, which can't be retuned. Status of "
                "elements has meaning only if they can be retuned."
            )
            return

        logging.error(
            f"You want to give {new_status = } to the element "
            f"{self.name}, which update_status method is not "
            "defined."
        )
