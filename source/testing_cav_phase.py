#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define testing functions for the new phase cavity object."""
import logging

import numpy as np

from core.elements.field_maps.cavity_settings import CavitySettings


logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    # This is the creation of the list of elements
    fm1 = CavitySettings(
        k_e=1.,
        phi=np.pi / 2.,
        reference='phi_0_rel',
        status='nominal',
        phi_s_function=lambda x: x,
        bunch_phase_to_rf_phase=lambda x: 2. * x,
    )
    print(fm1)

    # We propagate the beam and a particle enters the cavity for the first time
    # fm1.phi_rf = 0.
    # fm1.phi_0_rel
    # fm1.phi_0_abs
    # print(fm1)
