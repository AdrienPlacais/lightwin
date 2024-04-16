#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define functions for TraceWin CLI."""
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from core.elements.field_maps.cavity_settings import CavitySettings
from core.elements.field_maps.field_map import FieldMap
from failures.set_of_cavity_settings import SetOfCavitySettings
from util.helper import flatten

TYPES = {
    "hide": None,
    "tab_file": str,
    "synoptic_file": str,
    "nbr_thread": int,
    "path_cal": str,
    "dat_file": str,
    "dst_file1": str,
    "dst_file2": str,
    "current1": float,
    "current2": float,
    "nbr_part1": int,
    "nbr_part2": int,
    "energy1": float,
    "energy2": float,
    "etnx1": float,
    "etnx2": float,
    "etny1": float,
    "etny2": float,
    "eln1": float,
    "eln2": float,
    "freq1": float,
    "freq2": float,
    "duty1": float,
    "duty2": float,
    "mass1": float,
    "mass2": float,
    "charge1": float,
    "charge2": float,
    "alpx1": float,
    "alpx2": float,
    "alpy1": float,
    "alpy2": float,
    "alpz1": float,
    "alpz2": float,
    "betx1": float,
    "betx2": float,
    "bety1": float,
    "bety2": float,
    "betz1": float,
    "betz2": float,
    "x1": float,
    "x2": float,
    "y1": float,
    "y2": float,
    "z1": float,
    "z2": float,
    "xp1": float,
    "xp2": float,
    "yp1": float,
    "yp2": float,
    "zp1": float,
    "zp2": float,
    "dw1": float,
    "dw2": float,
    "spreadw1": float,
    "spreadw2": float,
    "part_step": int,
    "vfac": float,
    "random_seed": int,
    "partran": int,
    "toutatis": int,
}


def variables_to_command(
    warn_skipped: bool = False, **kwargs: str | float | int
) -> Sequence[str]:
    """Generate a TraceWin command from the input dictionary.

    If the `value` of the `dict` is None, only corresponding `key` is added
    (behavior for `hide` command).

    If `value` is `np.NaN`, it is ignored.

    Else, the pair `key`-`value` is added as `key=value` string.
    """
    command = []

    for key, val in kwargs.items():
        val = _proper_type(key, val)

        if isinstance(val, float) and np.isnan(val):
            if warn_skipped:
                logging.warning(
                    f"For {key=}, I had a np.NaN value. I ignore this key."
                )
            continue

        if val is None:
            command.append(key)
            continue

        command.append(f"{key}={str(val)}")
    return command


def beam_calculator_to_command(
    executable: Path,
    ini_path: Path,
    path_cal: Path,
    **kwargs: str | int | float | bool | None,
) -> list[str]:
    """Give command calling TraceWin according to `BeamCalculator` attribs."""
    kwargs = {
        "path_cal": str(path_cal),
    } | kwargs
    command = variables_to_command(**kwargs)
    command.insert(0, str(executable))
    command.insert(1, str(ini_path))
    return command


def list_of_elements_to_command(dat_filepath: Path) -> list[str]:
    """
    Return a command from `ListOfElements` attributes.

    `ParticleInitialState` and `BeamParameters` have their own function, they
    are not called from here.

    """
    kwargs = {
        "dat_file": str(dat_filepath),
    }
    return variables_to_command(**kwargs)


def beam_parameters_to_command(
    eps_x: float,
    alpha_x: float,
    beta_x: float,
    eps_y: float,
    alpha_y: float,
    beta_y: float,
    eps_z: float,
    alpha_z: float,
    beta_z: float,
) -> list[str]:
    """Return a TraceWin command from the attributes of a `BeamParameters`."""
    kwargs = {
        "etnx1": eps_x,
        "alpx1": alpha_x,
        "betx1": beta_x,
        "etny1": eps_y,
        "alpy1": alpha_y,
        "bety1": beta_y,
        "eln1": eps_z,
        "alpz1": alpha_z,
        "betz1": beta_z,
    }
    return variables_to_command(**kwargs)


def particle_initial_state_to_command(w_kin: float) -> list[str]:
    """
    Return a TraceWin command from attributes of `ParticleInitialState`.

    We could use the `zp` command to modify the phase at the entry of the first
    element (when it is not the first element of the linac).
    We rather keep the absolute phase at the beginning of the zone to 0. and
    modify the `.dat` file in `subset_of_pre_existing_list_of_elements`
    function in order to always keep the same relative phi_0.
    """
    kwargs = {"energy1": w_kin}
    return variables_to_command(**kwargs)


def set_of_cavity_settings_to_command(
    set_of_cavity_settings: SetOfCavitySettings,
    phi_bunch_first_element: float,
    idx_first_element: int,
) -> Sequence[str]:
    """Return the ``ele`` commands for :class:`SetOfCavitySettings`.

    Parameters
    ----------
    set_of_cavity_settings : SetOfCavitySettings
        All the new cavity settings.
    phi_bunch_first_element : float
        Phase of synchronous particle at entry of first element of
        :class:`.ListOfElements` under study.
    idx_first_element : int
        Index of first element of :class:`.ListOfElements` under study.

    Returns
    -------
    Sequence[str]
        Full command that will alter the TraceWin exection to match the
        desired ``set_of_cavity_settings``.

    """
    command = [
        _cavity_settings_to_command(
            field_map,
            cavity_settings,
            delta_phi_bunch=phi_bunch_first_element,
            delta_index=idx_first_element,
        )
        for field_map, cavity_settings in set_of_cavity_settings.items()
    ]
    return [x for x in flatten(command)]


def _cavity_settings_to_command(
    field_map: FieldMap,
    cavity_settings: CavitySettings,
    delta_phi_bunch: float = 0.0,
    delta_index: int = 0,
) -> Sequence[str]:
    """Convert ``cavity_settings`` into TraceWin CLI arguments.

    Parameters
    ----------
    field_map : FieldMap
        Cavity under study.
    cavity_settings : CavitySettings
        Settings to try.
    delta_phi_bunch : float, optional
        Phase at entry of first element of :class:`.ListOfElements` under study. The default is 0.
    delta_index : int, optional
        Index of the first element of :class:`.ListOfElements` under study. The default is 0.

    Returns
    -------
    command : Sequence[str]
        Piece of command to alter ``field_map`` with ``cavity_settings``.

    """
    if cavity_settings == field_map.cavity_settings:
        return ()
    if not hasattr(cavity_settings, "phi_bunch"):
        nominal_phi_bunch = field_map.cavity_settings.phi_bunch
        cavity_settings.phi_bunch = nominal_phi_bunch
    cavity_settings.shift_phi_bunch(delta_phi_bunch, check_positive=True)

    phi_0 = cavity_settings.phi_ref
    if phi_0 is None:
        phi_0 = np.NaN
    if ~np.isnan(phi_0):
        phi_0 = math.degrees(phi_0)

    elt_idx = field_map.idx["elt_idx"]
    alter_kwargs = {"k_e": cavity_settings.k_e, "phi_0": phi_0}
    tracewin_command = _alter_element(elt_idx - delta_index, alter_kwargs)
    return tracewin_command


ARGS_POSITIONS = {
    "phi_0": 3,
    "k_e": 6,
}  #:


def _alter_element(
    index: int, alter_kwargs: dict[str, float | int]
) -> Sequence[str]:
    """Create the command piece to modify the element at ``index``.

    Parameters
    ----------
    index
        Position of the element to modify in LightWin referential (first
        element has index 0).
    alter_kwargs
        Key-pair values, where key is the LightWin name of the parameter to
        update, and value the new value to set. Key-pair value is skipped if
        value is np.NaN. Key must be in :var:`ARGS_POSITIONS`.

    Returns
    -------
    Sequence[str]
        The ``ele[i][j]=val`` command altering the given element.

    """
    for val in alter_kwargs:
        assert val is not None, "Prefer np.NaN for values to skip."
    kwargs = {
        f"ele[{index + 1}][{ARGS_POSITIONS[arg]}]": value
        for arg, value in alter_kwargs.items()
    }
    return variables_to_command(warn_skipped=False, **kwargs)


def _proper_type(
    key: str,
    value: str | int | float,
    not_in_dict_warning: bool = True,
) -> str | int | float | None:
    """Check if type of `value` is consistent and try to correct otherwise."""
    if "ele" in key:
        return value
    # no type checking for ele command!

    if key not in TYPES:
        if not_in_dict_warning:
            logging.warning(
                f"The {key = } is not understood by TraceWin, or "
                "it is not implemented yet."
            )
        return np.NaN

    my_type = TYPES[key]
    if my_type is None:
        return None

    if isinstance(value, my_type):
        return value

    logging.warning(
        f"Input value {value} is a {type(value)} while it should "
        f"be a {my_type}."
    )
    try:
        value = my_type(value)
        logging.info(f"Succesful type conversion: {value = }")
        return value

    except ValueError:
        logging.error(
            "Unsuccessful type conversion. Returning np.NaN to "
            "completely ignore key."
        )
        return np.NaN
