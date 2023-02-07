#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:34:04 2023.

@author: placais
"""

import os

from util import helper


def save_files(lin, data=None, lw_fit_eval=None, flags=None):
    """Save a new dat with the new linac settings, as well as eval. files."""
    os.makedirs(lin.get('out_lw'))
    helper.printc("output.save_files info: ",
                  opt_message=f"new dat saved in {lin.get('dat_filepath')}\n")

    if data is not None:
        out = os.path.join(lin.get('out_lw'), 'data.csv')
        data.to_csv(out)
    if lw_fit_eval is not None:
        out = os.path.join(lin.get('out_lw'), 'lw_fit_eval.csv')
        lw_fit_eval.to_csv(out)
    if flags is not None:
        out = os.path.join(lin.get('out_lw'), 'flags.csv')
        flags.to_csv(out)

    with open(lin.get('dat_filepath'), 'w') as file:
        for line in lin.files['dat_filecontent']:
            file.write(' '.join(line) + '\n')


def save_project_evaluation_files(project_folder, lw_fit_evals=None,
                                  flags=None):
    """Save files that sum up the evaluations of all compensations."""
    if lw_fit_evals is not None:
        out = os.path.join(project_folder, 'all_lw_fit_evals.csv')
        lw_fit_evals.to_csv(out)
    if flags is not None:
        out = os.path.join(project_folder, 'all_flags.csv')
        flags.to_csv(out)


def format_project_evaluation_files(linacs, lw_fit_evals, flags):
    """Prepare the project evaluations files for latter treatment."""
    print("FIXME")
