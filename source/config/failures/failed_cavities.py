#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:49:03 2023.

@author: placais

In this module, we check the validity of the ``idx`` and ``failed`` user
inputs.

"""
import logging
import configparser


def test_failed_cavities(c_wtf: configparser.SectionProxy) -> bool:
    """
    Check that failed cavities are given.

    Required keys are ``idx`` and ``failed``.

    """
    for key in ['failed', 'idx']:
        if key not in c_wtf.keys():
            logging.error(f"You must provide {key}.")
            return False
    return _test_failed(c_wtf) and _test_idx(c_wtf)


def _test_idx(c_wtf: configparser.SectionProxy) -> bool:
    """Check that the ``idx`` was given and valid.

    If ``idx == 'cavity'``, ``failed = 5`` means that the 4th cavity will be
    broken.
    If ``idx == 'element'``, ``failed = 5`` means that the 4th element will be
    broken. If this element is not a cavity, an error will be raised later
    during execution of the code.

    This is key is also used for ``manual list``, used when the compensation
    strategy is manual.

    """
    val = c_wtf.get('idx')
    if val not in ('cavity', 'element'):
        logging.error(f"idx key is {val}, while it must be 'cavity' or "
                      "'element'.")
        return False
    return True


def _test_failed(c_wtf: configparser.SectionProxy) -> bool:
    """
    Check that ``failed`` was given.

    In the input file, one line = one simulation.

    Example
    -------
    We consider that ``idx = cavity``.

    .. code-block:: ini

        failed =
            1, 2,
            8,
            1, 2, 8

    In this case, LW will first fix the linac with the 1st and 2nd cavity
    failed. In a second time, LW will fix an error with the 8th failed cavity.
    In the last simulation, it will fix together the 1st, 2nd and 8th cavity.
    From LightWin's point of view: one line = one FaultScenario object.
    Each FaultScenario has a list of Fault objects. This is handled by
    the Faults are sorted by the FaultScenario._sort_faults method.

    See also
    --------
    :func:`failures.strategy._manual`

    """
    val = c_wtf.get('failed')
    if len(val) == 0:
        logging.warning("No fault was given.")
    return True
