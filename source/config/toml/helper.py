#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define utility functions to test out the ``.toml`` config file."""
import logging
from pathlib import Path
from typing import Any


def check_type(
    instance: type | tuple[type],
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
    return "\n".join(nice)


def find_file(toml_folder: Path, file: str | Path) -> Path:
    """Look for the given filepath in all possible places, make it absolute.

    We sequentially check and return the first valid path:
        1. If ``file`` is a ``Path`` object, we consider that the user already
        set it as he wanted. We check if it exists.
        2. If ``file`` is in ``toml_folder``.
        3. If ``file`` is absolute.

    Parameters
    ----------
    toml_folder : Path
        Folder where the ``.toml`` configuration file is.
    file : str | Path
        Filepath to look for.

    Returns
    -------
    file : Path
        Absolute filepath, which existence has been checked.

    """
    if isinstance(file, Path):
        path = file.resolve().absolute()
        if path.is_file():
            return path

    path = (toml_folder / file).resolve().absolute()
    if path.is_file():
        return path

    path = Path(file).resolve().absolute()
    if path.is_file():
        return path

    msg = (
        f"{file = } was not found. It can be defined relative to the "
        ".toml (recommended), absolute, or relative to the execution dir"
        "of the script (not recommended)."
    )
    logging.critical(msg)
    raise FileNotFoundError(msg)

    pass
