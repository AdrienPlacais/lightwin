#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:32:12 2021

@author: placais
"""

# import numpy as np
# import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import accelerator as acc

# =============================================================================
# User inputs
# =============================================================================
LINAC = acc.Accelerator()

Tk().withdraw()
# filepath = '/home/placais/LightWin/data/dummy.dat'
filepath = '/home/placais/TraceWin/work_compensation/work_compensation.dat'
# filepath = askopenfilename(filetypes=[("TraceWin file", ".dat")])
LINAC.create_struture_from_dat_file(filepath)
