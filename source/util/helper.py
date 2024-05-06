#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define misc helper functions.

.. todo::
    Clean this, check what is still used.

"""
import logging
from collections.abc import Generator, Iterable
from typing import Any, Iterator

import numpy as np
import pandas as pd


# =============================================================================
# For getter and setters
# =============================================================================
def recursive_items(dictionary: dict[Any, Any]) -> Iterator[str]:
    """Recursively list all keys of a possibly nested dictionary."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            yield key
            yield from recursive_items(value)
        elif hasattr(value, "has"):
            yield key
            yield from recursive_items(vars(value))
            # for ListOfElements:
            if isinstance(value, list):
                yield from recursive_items(vars(value[0]))
        else:
            yield key


def recursive_getter(
    wanted_key: str, dictionary: dict, **kwargs: bool | str | int
) -> Any:
    """Get first key in a possibly nested dictionary."""
    for key, value in dictionary.items():
        if wanted_key == key:
            return value

        if isinstance(value, dict):
            corresp_value = recursive_getter(wanted_key, value, **kwargs)
            if corresp_value is not None:
                return corresp_value

        elif hasattr(value, "get"):
            corresp_value = value.get(wanted_key, **kwargs)
            if corresp_value is not None:
                return corresp_value
    return None


# =============================================================================
# For lists manipulations
# =============================================================================
def flatten[T](nest: Iterable[T]) -> Iterator[T]:
    """Flatten nested list of lists of..."""
    for _in in nest:
        if isinstance(_in, Iterable) and not isinstance(_in, (str, bytes)):
            yield from flatten(_in)
        else:
            yield _in


def chunks[T](lst: list[T], n_size: int) -> Generator[list[T], int, None]:
    """Yield successive ``n_size``-ed chunks from ``lst``.

    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks

    """
    for i in range(0, len(lst), n_size):
        yield lst[i : i + n_size]


def remove_duplicates[T](iterable: Iterable[T]) -> Iterator[T]:
    """Create an iterator without duplicates.

    Taken from:
    https://stackoverflow.com/questions/32012878/iterator-object-for-removing-duplicates-in-python

    """
    seen = set()
    for item in iterable:
        if item in seen:
            continue
        seen.add(item)
        yield item


# =============================================================================
# Messages functions
# =============================================================================
# TODO: transform inputs into strings if they are not already strings
def printc(*args: str, color: str = "cyan") -> None:
    """Print colored messages."""
    dict_c = {
        "red": "\x1B[31m",
        "blue": "\x1b[34m",
        "green": "\x1b[32m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "normal": "\x1b[0m",
    }
    print(dict_c[color] + args[0] + dict_c["normal"], end=" ")
    for arg in args[1:]:
        print(arg, end=" ")
    print("")


def pd_output(message: pd.DataFrame, header: str = "") -> str:
    """Give a string describing a pandas dataframe."""
    tot = 100
    my_output = header + "\n" + "-" * tot + "\n" + message.to_string()
    my_output += "\n" + "-" * tot
    return my_output


def pascal_case(message: str) -> str:
    """Convert a string to Pascal case (as class names).

    .. todo::
        Second example does not work

    Examples
    --------
    >>> pascal_case("bonjoure sa_vA")
    "BonjoureSaVa"
    >>> pascal_case("BonjoureSaVa")
    "BonjoureSaVa"

    """
    return "".join(x for x in message.title() if x not in (" ", "_"))


def get_constructor(name: str, constructors: dict[str, type]) -> type:
    """Get the proper class from a string and dict of classes."""
    pascal_name = pascal_case(name)

    if pascal_name in constructors:
        return constructors[pascal_name]
    if name in constructors:
        constructor = constructors[name]
        logging.warning(
            f"{constructor = } matches the provided {name = }, but consider "
            f"calling it {pascal_name} for consistency."
        )
        return constructor
    raise KeyError(
        f"Neither {pascal_name = } nor {name = } is in {constructors = }"
    )


def get_constructors(
    names: Iterable[str], constructors: dict[str, type]
) -> Generator[type, None, None]:
    """Get several class constructors from their names."""
    return (get_constructor(name, constructors) for name in names)


# TODO: replace nan by ' ' when there is a \n in a pd DataFrame header
# def printd(message: str, header: str = '') -> None:
# """Print delimited message."""
# pd.options.display.float_format = '{:.6f}'.format
# pd.options.display.max_columns = 10
# pd.options.display.max_colwidth = 18
# pd.options.display.width = 250

# # tot = 100
# # my_output = header + "\n" + "-" * tot + "\n" + message.to_string()
# # my_output += "\n" + "-" * tot
# my_output = pd_output(message, header)
# logging.info(my_output)


def resample(
    x_1: np.ndarray, y_1: np.ndarray, x_2: np.ndarray, y_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample y_highres(olution) to x_1 or x_2 (the one with low res)."""
    assert x_1.shape == y_1.shape
    assert x_2.shape == y_2.shape

    if x_1.shape > x_2.shape:
        y_1 = np.interp(x_2, x_1, y_1)
        x_1 = x_2
        return x_1, y_1, x_2, y_2
    y_2 = np.interp(x_1, x_2, y_2)
    x_2 = x_1
    return x_1, y_1, x_2, y_2


def range_vals(name: str, data: np.ndarray | None) -> str:
    """Return formatted first and last value of the ``data`` array."""
    out = f"{name:17s}"
    if data is None:
        return out + " (not set)\n"
    out += f"{data[0]:+9.5e} -> {data[-1]:+9.5e} | {data.shape}\n"
    return out


def range_vals_object(obj: object, name: str) -> str:
    """Return first and last value of the ``name`` attr from ``obj``."""
    val = getattr(obj, name)
    out = f"{name:17s}"
    if val is None:
        return out + " (not set)\n"
    if isinstance(val, float):
        return out + f"{val} (single value)\n"
    out += f"{val[0]:+9.5e} -> {val[-1]:+9.5e} | {val.shape}\n"
    return out


# =============================================================================
# Files functions
# =============================================================================
def save_energy_phase_tm(lin: object) -> None:
    """
    Save energy, phase, transfer matrix as a function of s.

    s [m]   E[MeV]  phi[rad]  M_11    M_12    M_21    M_22

    Parameters
    ----------
    lin : Accelerator object
        Object of corresponding to desired output.
    """
    n_z = lin.get("z_abs").shape[0]
    data = np.column_stack(
        (
            lin.get("z_abs"),
            lin.get("w_kin"),
            lin.get("phi_abs_array"),
            np.reshape(lin.transf_mat["tm_cumul"], (n_z, 4)),
        )
    )
    filepath = lin.files["results_folder"] + lin.name + "_energy_phase_tm.txt"
    filepath = filepath.replace(" ", "_")
    header = (
        "s [m] \t W_kin [MeV] \t phi_abs [rad]"
        + "\t M_11 \t M_12 \t M_21 \t M_22"
    )
    np.savetxt(filepath, data, header=header)
    logging.info(f"Energy, phase and TM saved in {filepath}")


def save_vcav_and_phis(lin: object) -> None:
    """
    Output the Vcav and phi_s as a function of z.

    s [m]   V_cav [MV]  phi_s [deg]

    Parameters
    ----------
    accelerator: Accelerator object
        Object of corresponding to desired output.
    """
    printc("helper.save_vcav_and_phis warning:", "s [m] not saved.")
    # data = lin.get('abs', 'v_cav_mv', 'phi_s', to_deg=True)
    data = lin.get("v_cav_mv", "phi_s", to_deg=True)
    data = np.column_stack((data[0], data[1]))

    filepath = lin.files["results_folder"] + lin.name + "_Vcav_and_phis.txt"
    filepath = filepath.replace(" ", "_")

    header = "s [m] \t V_cav [MV] \t phi_s [deg]"
    np.savetxt(filepath, data, header=header)
    logging.info(
        "Cavities accelerating field and synch. phase saved in "
        + f"{filepath}"
    )
