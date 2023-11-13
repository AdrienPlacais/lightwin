#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define simple tests for functionality under implementation."""
from typing import Sequence
from core.elements.element import Element
from core.elements.field_map import FieldMap


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
