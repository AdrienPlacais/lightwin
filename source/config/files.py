#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the ``files``, set up ``logging`` module."""
import datetime
import logging
from pathlib import Path

from config.helper import find_file
from util.log_manager import set_up_logging


def test(config_folder: Path, dat_file: str, **files_kw: str | Path) -> None:
    """Check that given path exists, modify if necessary. Set up logging.

    Parameters
    ----------
    config_folder : Path
        Where the ``.toml`` is stored. Used to resolve relative paths.
    dat_file : str
        Path to the ``.dat`` file.
    files_kw : str
        Other ``files`` keyword arguments.

    """
    _ = find_file(config_folder, dat_file)


def edit_configuration_dict_in_place(
    files_kw: dict[str, str | Path],
    config_folder: Path,
) -> None:
    """Set some useful paths."""
    dat_file = files_kw["dat_file"]
    files_kw["dat_file"] = find_file(config_folder, dat_file)

    project_folder = files_kw.get("project_folder", "")
    project_path = _create_project_folders(config_folder, project_folder)
    files_kw["project_folder"] = project_path

    log_file = Path(project_path, "lightwin.log")
    set_up_logging(logfile_file=log_file)
    logging.info(f"Setting {project_path = }\nSetting {log_file = }")

    if "cal_file" in files_kw:
        files_kw["cal_file"] = Path(files_kw["cal_file"]).resolve().absolute()


def _create_project_folders(
    config_folder: Path, project_folder: str | Path = ""
) -> Path:
    """Create a folder to store outputs and log messages."""
    if isinstance(project_folder, Path):
        project_path = project_folder.resolve().absolute()
        exist_ok = True

    else:
        if project_folder:
            project_path = (config_folder / project_folder).resolve()
            exist_ok = True

        else:
            time = datetime.datetime.now().strftime("%Y.%m.%d_%Hh%M_%Ss_%fms")
            project_path = config_folder / time
            exist_ok = False

    project_path.mkdir(exist_ok=exist_ok)
    return project_path.absolute()
