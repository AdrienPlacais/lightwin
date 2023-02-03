#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:44:53 2022.

@author: placais

FIXME : removed pymoo.util.termination in import and _set_termination
"""
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem

F = get_problem("zdt3").pareto_front()
Scatter().add(F).show()
