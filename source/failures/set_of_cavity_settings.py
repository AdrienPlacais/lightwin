#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds classes to store (compensating) cavity settings."""
from collections.abc import Sequence
from typing import Self, TypeVar

from core.elements.field_maps.cavity_settings import CavitySettings

# from core.list_of_elements.list_of_elements import ListOfElements

FieldMap = TypeVar("FieldMap")


class SetOfCavitySettings(dict[FieldMap, CavitySettings]):
    """Hold several cavity settings, to try during optimisation process."""

    def __init__(
        self, several_cavity_settings: dict[FieldMap, CavitySettings]
    ) -> None:
        """Create the proper dictionary."""
        super().__init__(several_cavity_settings)

    @classmethod
    def for_new(
        cls,
        several_cavity_settings: Sequence[CavitySettings],
        compensating_cavities: Sequence[FieldMap],
    ) -> Self:
        """Create the proper dictionary."""
        zipper = zip(
            compensating_cavities, several_cavity_settings, strict=True
        )
        settings = {
            cavity: cavity_settings for cavity, cavity_settings in zipper
        }
        return cls(settings)

    @classmethod
    def take_missing_settings_from_original_elements(
        cls, set_of_cavity_settings: Self, cavities: Sequence[FieldMap]
    ) -> Self:
        """Create an object with settings for all the field maps.

        We give each cavity settings from ``set_of_cavity_settings`` if they
        are listed in this object. If they are not, we give them their default
        :class:`.CavitySettings` (:attr:`.FieldMap.cavity_settings` attribute).
        This method is used to generate :class:`.SimulationOutput` where all
        the cavity settings are explicitly defined.

        Parameters
        ----------
        set_of_cavity_settings
            Object holding the settings of some cavities (typically, the
            settings of compensating cavities as given by an
            :class:`.OptimisationAlgorithm`).
        cavities
            All the cavities that should have :class:`.CavitySettings`
            (typically, all the cavities in a sub-:class:`.ListOfElements`
            studied during an optimisation process).

        Returns
        -------
        Self
            A :class:`.SetOfCavitySettings` with settings from
            ``set_of_cavity_settings`` or from ``cavities`` if
        """
        settings = {
            cavity: set_of_cavity_settings.get(cavity, cavity.cavity_settings)
            for cavity in cavities
        }
        return cls(settings)

    def re_set_elements_index_to_absolute_value(self) -> None:
        """Update cavities index to properly set `ele[n][v]` commands.

        When switching from a sub-:class:`ListOfElements` during the
        optimisation process to the full :class:`ListOfElements` after the
        optimisation, we must update index ``n`` in the ``ele[n][v]]`` command.

        """
        for cavity, setting in self.items():
            absolute_index = cavity.idx["elt_idx"]
            setting.index = absolute_index
