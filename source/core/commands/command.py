#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds all the commands.

"""
COMMANDS = [
    # 'END',
    # 'FIELD_MAP_PATH',
    'FREQ',
    'LATTICE',
    'LATTICE_END',
    'SUPERPOSE_MAP',
]


class Command:
    """A generic Command class."""


class End(Command):
    """The end of the linac."""

    def __init__(self, elem: list[str]) -> None:
        pass


class FieldMapPath(Command):
    """Used to get the base path of field maps."""

    def __init__(self, elem: list[str]) -> None:
        self.path = elem[1]


class Freq(Command):
    """Used to get the frequency of every Section."""

    def __init__(self, elem: list[str]) -> None:
        self.f_rf_mhz = float(elem[1])


class Lattice(Command):
    """Used to get the number of elements per lattice."""

    def __init__(self, elem: list[str]) -> None:
        self.n_lattice = int(elem[1])


class LatticeEnd(Command):
    """Dummy class."""

    def __init__(self, elem: list[str]) -> None:
        pass


class SuperposeMap(Command):
    """Dummy class."""

    def __init__(self, elem: list[str]) -> None:
        pass
