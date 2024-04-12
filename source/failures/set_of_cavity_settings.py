#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to store several :class:`.CavitySettings`.

.. todo::
    I should create a :class:`.SetOfCavitySettings` with
    :class:`.CavitySettings` for every cavity of the compensation zone.
    Mandatory to recompute the synchronous phases.

"""
from collections.abc import Sequence
from typing import Self, TypeVar

from core.elements.field_maps.cavity_settings import CavitySettings

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
        cls,
        set_of_cavity_settings: Self,
        cavities: Sequence[FieldMap],
        instantiate_new: bool = True,
    ) -> Self:
        """Create an object with settings for all the field maps.

        We give each cavity settings from ``set_of_cavity_settings`` if they
        are listed in this object. If they are not, we give them their default
        :class:`.CavitySettings` (:attr:`.FieldMap.cavity_settings` attribute).
        This method is used to generate :class:`.SimulationOutput` where all
        the cavity settings are explicitly defined.

        .. note::
            In fact, may be useless. In the future, the nominal cavities will
            also have their own :class:`.CavitySettings` in the compensation
            zone.

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
        instantiate_new
            To create new :class:`.CavitySettings` for the cavities not already
            in ``set_of_cavity_settings``. Allows to compute quantities such as
            synchronous phase without altering the original one.

        Returns
        -------
        Self
            A :class:`.SetOfCavitySettings` with settings from
            ``set_of_cavity_settings`` or from ``cavities`` if not in
            ``set_of_cavity_settings``.

        """
        complete_set_of_settings = {
            cavity: _settings_getter(
                cavity, set_of_cavity_settings, instantiate_new
            )
            for cavity in cavities
        }
        return cls(complete_set_of_settings)

    def re_set_elements_index_to_absolute_value(self) -> None:
        """Update cavities index to properly set `ele[n][v]` commands.

        When switching from a sub-:class:`ListOfElements` during the
        optimisation process to the full :class:`ListOfElements` after the
        optimisation, we must update index ``n`` in the ``ele[n][v]]`` command.

        """
        for cavity, setting in self.items():
            absolute_index = cavity.idx["elt_idx"]
            setting.index = absolute_index


def _settings_getter(
    cavity: FieldMap,
    set_of_cavity_settings: SetOfCavitySettings,
    instantiate_new: bool,
) -> CavitySettings:
    """Take the settings from the set of settings if possible.

    If ``cavity`` is not listed in ``set_of_cavity_settings``, take its nominal
    :class:`.CavitySettings` instead. In the latter case, ``instantiate_new``
    will force the creation of a new :class:`.CavitySettings` with the same
    settings.

    Parameters
    ----------
    cavity
        Cavity for which you want settings.
    set_of_cavity_settings
        Different cavity settings (a priori given by an
        :class:`OptimisationAlgorithm`).
    instantiate_new
        To force the creation of a new object; will allow to keep the original
        :class:`.CavitySettings` unaltered.

    Returns
    -------
    CavitySettings
        Cavity settings for ``cavity``.

    """
    if not instantiate_new:
        return set_of_cavity_settings.get(cavity, cavity.cavity_settings)

    return set_of_cavity_settings.get(
        cavity,
        CavitySettings.from_other_cavity_setttings(cavity.cavity_settings),
    )
