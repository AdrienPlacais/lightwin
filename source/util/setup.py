#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:46:05 2022.

@author: placais

This module simply holds the function to build the Cython module(s).

To compile, go to `LightWin/source/` and enter:
``python3 util/setup.py build_ext --inplace``
and manually move the `.so` in `beam_calculation/envelope_1d` #FIXME

IMPORTANT: in some interpreters such as Spyder, `.so` are loaded at the startup
of the software. Hence, you must restart the kernel after each compilation

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
