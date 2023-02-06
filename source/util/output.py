#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:34:04 2023.

@author: placais
"""

import os

from util import helper


def save_files(lin, data=None, ranking=None, flags=None):
    """Save a new dat with the new linac settings."""
    os.makedirs(lin.get('out_lw'))
    helper.printc("output.save_files info: ",
                  opt_message=f"new dat saved in {lin.get('dat_filepath')}\n")

    if data is not None:
        out = os.path.join(lin.get('out_lw'), 'data.csv')
        data.to_csv(out)
    if ranking is not None:
        out = os.path.join(lin.get('out_lw'), 'ranking.csv')
        ranking.to_csv(out)
    if flags is not None:
        out = os.path.join(lin.get('out_lw'), 'flags.csv')
        flags.to_csv(out)

    with open(lin.get('dat_filepath'), 'w') as file:
        for line in lin.files['dat_filecontent']:
            file.write(' '.join(line) + '\n')
