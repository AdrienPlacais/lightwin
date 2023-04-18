#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:46:05 2022.

@author: placais
"""

from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("core/transfer_matrices_c.pyx",))
      #                       compiler_directives={'boundscheck': False,
      #                                            'nonecheck': False,
      #                                            'wraparound': False}
      #                       )
      # )
# To compile, go to LightWin/source/ and enter:
# python3 util/setup.py build_ext --inplace
# and manually move the .so in core/ #FIXME
# IMPORTANT: in some interpreters such as Spyder, .so are loaded at the startup
# of the software. Hence, you must restart the kernel after each compilation
