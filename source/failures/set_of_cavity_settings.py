#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds classes to store (compensating) cavity settings."""
from collections.abc import Sequence
from typing import Self, TypeVar

from core.elements.field_maps.cavity_settings import CavitySettings

FieldMap = TypeVar('FieldMap')


class SetOfCavitySettings(dict[FieldMap, CavitySettings]):
    """Hold several cavity settings, to try during optimisation process."""

    def __init__(self,
                 several_cavity_settings: dict[FieldMap, CavitySettings]
                 ) -> None:
        """Create the proper dictionary."""
        super().__init__(several_cavity_settings)

    @classmethod
    def for_new(cls,
                several_cavity_settings: Sequence[CavitySettings],
                compensating_cavities: Sequence[FieldMap]
                ) -> Self:
        """Create the proper dictionary."""
        zipper = zip(compensating_cavities, several_cavity_settings,
                     strict=True)
        settings = {cavity: cavity_settings
                    for cavity, cavity_settings in zipper}
        return cls(settings)

    def tracewin_command(self, delta_phi_bunch: float = 0.) -> list[str]:
        """Set TraceWin command modifier for current settings."""
        _tracewin_command = []
        for settings in self.values():
            _tracewin_command.extend(
                settings.tracewin_command(delta_phi_bunch=delta_phi_bunch)
            )
        return _tracewin_command

    def re_set_elements_index_to_absolute_value(self) -> None:
        """Update cavities index to properly set `ele[n][v]` commands.

        When switching from a sub-:class:`ListOfElements` during the
        optimisation process to the full :class:`ListOfElements` after the
        optimisation, we must update index ``n`` in the ``ele[n][v]]`` command.

        """
        for cavity, setting in self.items():
            absolute_index = cavity.idx['elt_idx']
            setting.index = absolute_index
