#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define utility functions to test out the ``.toml`` config file."""
import logging

from typing import Any


def check_type(instance: type | tuple[type],
               name: str,
               *args: Any,
               ) -> None:
    """
    Raise a warning if ``args`` are not all of type ``instance``.

    Not matching the provided type does not stop the program from running.

    """
    for arg in args:
        if not isinstance(arg, instance):
            logging.warning(f"{name} testing: {arg} should be a {instance}")


def dict_for_pretty_output(some_kw: dict) -> str:
    """Transform a dict in strings for nice output."""
    nice = [f"{key:>52s} = {value}" for key, value in some_kw.items()]
    return '\n'.join(nice)
