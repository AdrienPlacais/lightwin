#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define simple tests for functionality under implementation."""
from dataclasses import dataclass
from typing import Sequence, Self

import numpy as np

from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap


def assert_are_field_maps(elements: Sequence[Element], detail: str) -> None:
    """Test that all elements are field maps.

    This function exists because in first implementations of LightWin, only
    FIELD_MAP could be broken/retuned, and all FIELD_MAP were 1D along z.
    Hence there was a confustion between what is/should be a cavity, an
    accelerating elementm what could be broken, what could be used for
    compensation.
    Also useful to identify where bugs will happen when implementing tuning of
    quadrupoles, etc.

    Parameters
    ----------
    elements : Sequence[Element]
        List of elements to test.
    detail : str
        More information that will be printed if not all elements are field
        maps.

    """
    are_all = all([isinstance(element, FieldMap) for element in elements])
    if not are_all:
        msg = "At least one element here is not a FieldMap. While this "
        msg += "should be possible, implementation is not realized yet. More "
        msg += "details: " + detail
        raise NotImplementedError(msg)


@dataclass
class Generic:
    """A generic class."""

    preset: str
    property_1: np.ndarray

    def __post_init__(self):
        """Write a doc."""
        print(f"initialized generic {self} of type {type(self)}")

    @property
    def property_1_pos(self) -> np.ndarray:
        """Write a doc."""
        return self.property_1[:, 0]

    @classmethod
    def from_truc(cls, truc: str, property_2: np.ndarray) -> Self:
        """Make the doc."""
        preset = truc[::-1]
        property_1 = np.sqrt(property_2)
        return cls(preset, property_1)


@dataclass
class Initial(Generic):
    """Write a doc."""

    property_1: tuple[float, float]

    def __post_init__(self):
        """Write a doc."""
        print(f"initialized initial {self} of type {type(self)}")
        super().__post_init__()

    @property
    def property_1_pos(self) -> float:
        """Write a doc."""
        return self.property_1[0]

    # @classmethod
    # def from_truc(cls, truc: str, property_2: np.ndarray) -> Self:
    #     """Make a doc."""
    #     return super().from_truc(truc, property_2)


if __name__ == '__main__':

    property_1 = np.linspace(0, 10, 11)
    property_1 = np.vstack((
        property_1,
        property_1**2
    ))

    # gene = Generic('linac', property_1)
    # print(gene.property_1_pos)
    # init = Initial('lonac', (2., 4.))
    # print(init.property_1_pos)
    # gene = Generic.from_truc('canil', property_1**2)
    init = Initial.from_truc('calon', property_1**2)
