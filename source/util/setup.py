#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module simply holds the function to build the Cython module(s).

To compile, go to ``LightWin/source/`` and enter:
``python3 util/setup.py build_ext --inplace``
and manually move the ``build/lib.blabla/beam_calculation/envelope_1d/transfer_
matrices_c.cpython-blabla.so`` to ``beam_calculation/envelope1d/``

.. todo::
    Auto move the ``.so``.

.. note::
    In some interpreters such as Spyder, ``.so`` are loaded at the startup
    of the software. Hence, you must restart the kernel after each compilation.

"""

from setuptools import setup
from Cython.Build import cythonize
import os.path

source_path = "/home/placais/LightWin/source"
pyx_path = os.path.join(source_path,
                        "beam_calculation/envelope_1d/transfer_matrices_c.pyx"
                        )

setup(
    ext_modules=cythonize(
        pyx_path,
        # compiler_directives={'boundscheck': False,
        #                      'nonecheck': False,
        #                      'wraparound': False}
    )
)
