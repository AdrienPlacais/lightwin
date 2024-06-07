"""Define the functions to test the :class:`.BeamCalculator` arguments.

.. todo::
    Handling of arguments for TW could be cleaner

"""

import logging
import socket
import tomllib
from pathlib import Path

from config.helper import check_type

IMPLEMENTED_BEAM_CALCULATORS = ("Envelope1D", "TraceWin", "Envelope3D")  #:
IMPLEMENTED_ENVELOPE1D_METHODS = ("leapfrog", "RK4")  #:
IMPLEMENTED_ENVELOPE3D_METHODS = ("RK4",)  #:

IMPLEMENTED_TRACEWIN_ARGUMENTS = (
    "hide",
    "tab_file",
    "nbr_thread",
    "dst_file1",
    "dst_file2",
    "current1",
    "current2",
    "nbr_part1",
    "nbr_part2",
    "energy1",
    "energy2",
    "etnx1",
    "etnx2",
    "etny1",
    "etny2",
    "eln1",
    "eln2",
    "freq1",
    "freq2",
    "duty1",
    "duty2",
    "mass1",
    "mass2",
    "charge1",
    "charge2",
    "alpx1",
    "alpx2",
    "alpy1",
    "alpy2",
    "alpz1",
    "alpz2",
    "betx1",
    "betx2",
    "bety1",
    "bety2",
    "betz1",
    "betz2",
    "x1",
    "x2",
    "y1",
    "y2",
    "z1",
    "z2",
    "xp1",
    "xp2",
    "yp1",
    "yp2",
    "zp1",
    "zp2",
    "dw1",
    "dw2",
    "spreadw1",
    "spreadw2",
    "part_step",
    "vfac",
    "random_seed",
    "partran",
    "toutatis",
    "algo",
    "cancel_matching",
    "cancel_matchingP",
)


def test(tool: str, **beam_calculator_kw: bool | str | int) -> None:
    """
    Ensure that selected :class:`.BeamCalculator` will be properly initialized.
    """
    assert tool in IMPLEMENTED_BEAM_CALCULATORS
    specific_tester = {
        "Envelope1D": _test_envelope1d,
        "TraceWin": _test_tracewin,
        "Envelope3D": _test_envelope3d,
    }
    specific_tester[tool](**beam_calculator_kw)


def edit_configuration_dict_in_place(
    beam_calculator_kw: dict, config_folder: Path, **kwargs
) -> None:
    """Precompute some useful values, transform some ``str`` into ``Path``."""
    tool = beam_calculator_kw["tool"]
    specific_editer_configuration_dict_in_place = {
        "Envelope1D": _edit_configuration_dict_in_place_envelope1d,
        "TraceWin": _edit_configuration_dict_in_place_tracewin,
        "Envelope3D": _edit_configuration_dict_in_place_envelope3d,
    }
    specific_editer_configuration_dict_in_place[tool](
        beam_calculator_kw, config_folder
    )


# Comment with recommended values: leapfrog 40 and RK 20
def _test_envelope1d(
    method: str,
    flag_cython: bool,
    flag_phi_abs: bool,
    n_steps_per_cell: int,
    **beam_calculator_kw: bool | str | int,
) -> None:
    """Test the consistency for the basic :class:`.Envelope1D` beam calculator."""
    if method == "RK":
        logging.warning(
            f"{method = } is deprecated. Prefer 'RK4' (same thing but more "
            "explicit)."
        )
        method = "RK4"
    assert method in IMPLEMENTED_ENVELOPE1D_METHODS
    check_type(bool, "beam_calculator", flag_cython, flag_phi_abs)
    check_type(int, "beam_calculator", n_steps_per_cell)


def _edit_configuration_dict_in_place_envelope1d(
    beam_calculator_kw: dict, config_folder: Path
) -> None:
    """Modify the kw dict inplace."""
    if beam_calculator_kw["method"] == "RK":
        beam_calculator_kw["method"] = "RK4"


def _test_tracewin(
    config_folder: Path,
    machine_config_file: str,
    simulation_type: str,
    ini_path: str,
    path_cal: str = "",
    synoptic_file: str = "",
    partran: int | None = None,
    toutatis: int | None = None,
    machine_name: str = "",
    **beam_calculator_kw,
) -> None:
    """Test consistency for :class:`.TraceWin` beam calculator."""
    check_type(str, "beam_calculator", simulation_type, ini_path)
    _ = _find_file(config_folder, ini_path)
    _ = _test_tracewin_executable(
        config_folder, machine_config_file, simulation_type, machine_name
    )

    if path_cal:
        _ = _find_file(config_folder, path_cal)

    if synoptic_file:
        logging.error(
            "Synoptic file not implemented as I am not sure how this should "
            "work."
        )
    for val in (partran, toutatis):
        if val is not None:
            assert val in (0, 1)

    for key in beam_calculator_kw.keys():
        if "Ele" in key:
            logging.error(
                "Are you trying to use the Ele[n][v] key? Please directly "
                "modify your `.dat` to avoid clash with LightWin."
            )
            continue
        if key == "upgrade":
            logging.warning(
                "Upgrading TraceWin from LightWin is not recommended."
            )
            continue


def _test_tracewin_executable(
    config_folder: Path,
    machine_config_file: str,
    simulation_type: str,
    machine_name: str = "",
    **kwargs,
) -> Path:
    """Look for the configuration file, check if TW executable exists."""
    filepath = _find_file(config_folder, machine_config_file)
    with open(filepath, "rb") as file:
        config = tomllib.load(file)

    if not machine_name:
        machine_name = socket.gethostname()

    assert (
        machine_name in config
    ), f"{machine_name = } should be in {config.keys() = }"
    this_machine_config = config[machine_name]

    assert (
        simulation_type in this_machine_config
    ), f"{simulation_type = } was not found in {this_machine_config = }"
    executable = Path(this_machine_config[simulation_type])
    assert executable.is_file, f"{executable = } was not found"
    return executable


def _edit_configuration_dict_in_place_tracewin(
    beam_calculator_kw: dict, config_folder: Path
) -> None:
    """Transform some values.

    The arguments that will be passed to the TraceWin executable are removed
    from ``beam_calculator_kw`` and stored in ``args_for_tracewin``, which is
    an entry of ``beam_calculator_kw``.

    """
    beam_calculator_kw["executable"] = _test_tracewin_executable(
        config_folder=config_folder, **beam_calculator_kw
    )

    paths = (
        "ini_path",
        "tab_file",
        "path_cal",
        "dst_file1",
        "dst_file2",
    )
    for path in paths:
        if path not in beam_calculator_kw:
            continue
        beam_calculator_kw[path] = _find_file(
            config_folder, beam_calculator_kw[path]
        )

    args_for_tracewin = {}
    args_for_lightwin = {}
    entries_to_remove = (
        "simulation_type",
        "machine_config_file",
        "machine_name",
    )

    for key, value in beam_calculator_kw.items():
        if key in entries_to_remove:
            continue

        if key not in IMPLEMENTED_TRACEWIN_ARGUMENTS:
            args_for_lightwin[key] = value
            continue

        args_for_tracewin[key] = value

    args_for_lightwin["base_kwargs"] = args_for_tracewin

    beam_calculator_kw.clear()
    for key, value in args_for_lightwin.items():
        beam_calculator_kw[key] = value


def _find_file(config_folder: Path, file: str) -> Path:
    """Look for ``file`` as absolute and in ``config_folder``."""
    path = (config_folder / file).resolve().absolute()
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


def _test_envelope3d(
    flag_phi_abs: bool,
    n_steps_per_cell: int,
    method: str = "RK4",
    flag_cython: bool | None = None,
    **beam_calculator_kw: bool | str | int,
) -> None:
    """Check validity of arguments for :class:`.Envelope3D`."""
    if method == "RK":
        logging.warning(
            f"{method = } is deprecated. Prefer 'RK4' (same thing but more "
            "explicit)."
        )
        method = "RK4"
    assert method in IMPLEMENTED_ENVELOPE3D_METHODS
    check_type(bool, "beam_calculator", flag_phi_abs)
    check_type(int, "beam_calculator", n_steps_per_cell)

    if flag_cython is not None:
        logging.warning("Cython not implemented yet for Envelope3D.")


def _edit_configuration_dict_in_place_envelope3d(
    beam_calculator_kw: dict, config_folder: Path
) -> None:
    """Modify the kw dict inplace."""
    if beam_calculator_kw["method"] == "RK":
        beam_calculator_kw["method"] = "RK4"
