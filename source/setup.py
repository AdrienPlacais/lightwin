#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:46:05 2022.

@author: placais
"""

from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("transfer_matrices_c.pyx"))
