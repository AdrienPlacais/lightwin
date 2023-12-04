#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to evaluate a :class:`.SimulationOutput`.

This object is generally created within two contexts:
    - after the resolution of a :class:`.FaultScenario`, to evaluate an
      optimisation quality;
    - at the end of the optimisation process, to evaluate the fit quality but
      with a second solver, supposed to be more precise.

"""
