#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define function to build the Cython module(s).

To compile, go to ``lightwin/source/`` and enter:
``python util/setup.py build_ext --inplace``.

If it is not automatic on your platform, you must manually move the created
files to the proper location.

Unix:
    ``build/lib.linux-XXX-cpython=3XX/beam_calculation/envelope_1d/transfer_matrices_c.cpython-3XX-XXXX-linux-gnu.so``
Windows
    ``build/lib.win-XXXX-cpython-3XX/beam_calculation/envelope_1d/transfer_matrices_c.cp3XX-win_XXXX.pyd``
to ``beam_calculation/envelope_1d/``

.. todo::
    Auto move the ``.so``.

.. note::
    In some interpreters such as Spyder, ``.so`` are loaded at the startup
    of the software. Hence, you must restart the kernel after each compilation.

"""
import os
import os.path

from Cython.Build import cythonize
import numpy as np
from setuptools import setup

source_path = os.getcwd()
pyx_path = os.path.join(
    source_path, "beam_calculation/envelope_1d/transfer_matrices_c.pyx"
)

setup(
    ext_modules=cythonize(
        pyx_path,
        # compiler_directives={'boundscheck': False,
        #                      'nonecheck': False,
        #                      'wraparound': False}
    ),
    include_dirs=[np.get_include()]
)
