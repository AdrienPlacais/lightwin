#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:46:38 2023

@author: placais
"""
import logging
from dataclasses import dataclass

from core.accelerator import Accelerator
from core.elements import _Element, FieldMap


@dataclass
class MyFault:
    """To handle and fix a single Fault."""
    ref: Accelerator
    fix: Accelerator
    wtf: dict
    elts: list[_Element]   # warning! update to ListOfElements after
    l_fcav: list[FieldMap]
    l_ccav: list[FieldMap]

    def __post_init__(self):
        """Set status of proper cavities."""
        self._init_status()

    def _init_status(self):
        """Update status of compensating and failed cavities."""
        for cav, status in zip([self.l_fcav, self.l_ccav],
                               ['failed', 'compensate (in progress)']):
            if cav.get('status') != 'nominal':
                logging.warning(f"Cavity {cav} is already used for another "
                                + "purpose. I will continue anyway...")
            cav.update_status(status)
