#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define some functions to set initial values/limits in DesignSpaceFactory.

.. todo::
    check docstrings ref [1]

"""
import numpy as np
from core.elements.element import Element


def same_value_as_nominal(variable: str,
                          reference_element: Element,
                          **kwargs,
                          ) -> float:
    """Return ``variable`` value in ``reference_element``.

    This is generally a good initial value for optimisation.

    """
    reference_value = reference_element.get(variable, to_numpy=False)
    assert isinstance(reference_value, float)
    return reference_value


def phi_s_limits(reference_element: Element,
                 max_increase_sync_phase_in_percent: float,
                 max_absolute_sync_phase_in_rad: float = 0.,
                 min_absolute_sync_phase_in_rad: float = -0.5 * np.pi,
                 **kwargs,
                 ) -> tuple[float, float]:
    r"""Return classic limits for the synchronous phase.

    Minimum is ``min_absolute_sync_phase_in_rad``, which is -90 degrees by
    default. Maximum is nominal synchronous phase +
    ``max_increase_in_percent``, or ``max_absolute_sync_phase_in_rad`` which is
    0 degrees by default.

    Parameters
    ----------
    reference_element : Element
        Element in its nominal tuning.
    max_increase_in_percent : float
        Maximum increase of the synchronous phase in percent.
    max_absolute_sync_phase_in_rad : float, optional
        Maximum absolute synchronous phase in radians. The default is 0.
    min_absolute_sync_phase_in_rad : float, optional
        Minimum absolute synchronous phase in radians. The default is
        :math:`-\pi / 2`.

    Returns
    -------
    tuple[float, float]
        Lower and upper limits for the synchronous phase.

    """
    reference_phi_s = same_value_as_nominal('phi_s', reference_element)
    phi_s_min = min_absolute_sync_phase_in_rad
    phi_s_max = min(
        max_absolute_sync_phase_in_rad,
        reference_phi_s * (1. - 1e-2 * max_increase_sync_phase_in_percent))
    return (phi_s_min, phi_s_max)


def phi_0_limits(**kwargs) -> tuple[float, float]:
    r"""Return classic limits for the absolute or relative rf phase.

    Returns
    -------
    tuple[float, float]
        Always :math:`(-2\pi, 2\pi)`.

    """
    return (-2. * np.pi, 2. * np.pi)


def k_e_limits(
        reference_element: Element,
        max_decrease_k_e_in_percent: float,
        max_increase_k_e_in_percent: float,
        maximum_k_e_is_calculated_wrt_maximum_k_e_of_section: bool = False,
        reference_elements: list[Element] | None = None,
        **kwargs
        ) -> tuple[float, float]:
    r"""Get classic limits for ``k_e``.

    Parameters
    ----------
    reference_element : Element
        The nominal element.
    max_decrease_in_percent : float
        Allowed decrease in percent with respect to the nominal ``k_e``.
    max_increase_in_percent : float
        Allowed increase in percent with respect to the nominal ``k_e``.
    maximum_k_e_is_calculated_wrt_maximum_k_e_of_section : bool, optional
        Use this flag to compute allowed increase of ``k_e`` with respect to
        the maximum ``k_e`` of the section, instead of the ``k_e`` of the
        nominal cavity. This is what we used in `[1]`_. The default is False.
    reference_elements : list[Element] | None
        List of the nominal elements. Must be provided if
        ``maximum_k_e_is_calculated_wrt_maximum_k_e_of_section`` is True.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds for ``k_e``.

    # .. _[1]: doi.org/10.18429/JACoW-LINAC2022-TUPORI04
    # A. Plaçais and F. Bouly, "Cavity Failure Compensation Strategies in
    # Superconducting Linacs," in Proceedings of LINAC2022, 2022, pp. 552–555.

    """
    reference_k_e = same_value_as_nominal('k_e', reference_element)
    min_k_e = reference_k_e * (1. - 1e-2 * max_decrease_k_e_in_percent)
    max_k_e = reference_k_e * (1. + 1e-2 * max_increase_k_e_in_percent)

    if not maximum_k_e_is_calculated_wrt_maximum_k_e_of_section:
        return (min_k_e, max_k_e)

    section_idx = reference_element.idx['section']
    assert isinstance(section_idx, int)
    assert reference_elements is not None
    max_k_e_of_section = _get_maximum_k_e_of_section(section_idx,
                                                     reference_elements)
    max_k_e = max_k_e_of_section * (1. + 1e-2 * max_increase_k_e_in_percent)
    return (min_k_e, max_k_e)


def _get_maximum_k_e_of_section(section_idx: int,
                                reference_elements: list[Element],
                                ) -> float:
    """Get the maximum ``k_e`` of section."""
    elements_in_current_section = list(filter(
        lambda element: element.idx['section'] == section_idx,
        reference_elements))
    k_e_in_current_section = [element.get('k_e', to_numpy=False)
                              for element in elements_in_current_section]
    maximum_k_e = np.nanmax(k_e_in_current_section)
    return maximum_k_e
