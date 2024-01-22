#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""All the functions for the ``files`` key of the config file."""
import logging
from pathlib import Path
import configparser
import datetime

from util.log_manager import set_up_logging


# =============================================================================
# Front end
# =============================================================================
def test(c_files: configparser.SectionProxy) -> None:
    """Test that the provided .dat is valid."""
    passed = True
    mandatory = ["dat_file"]
    for key in mandatory:
        if key not in c_files.keys():
            logging.error(f"Key {key} is mandatory and missing.")
            passed = False

    dat_file, project_folder = _create_project_folders(
        c_files.getpath('dat_file'))

    logfile_file = Path(project_folder, 'lightwin.log')
    set_up_logging(logfile_file=logfile_file)
    logging.info(f"Setting {project_folder = }\nSetting {logfile_file = }")

    c_files['dat_file'], c_files['project_folder'] = dat_file, project_folder

    if not c_files.getpath("dat_file").is_file():
        logging.error("The provided path for the .dat does not exist.")
        passed = False

    if not passed:
        raise IOError("Error treating the files parameters.")

    logging.info(f"files parameters {c_files.name} tested with success.")


def config_to_dict(c_files: configparser.SectionProxy) -> dict:
    """Save files info into a dict."""
    files = {}
    # Special getters
    getter = {
        "dat_file": c_files.getpath,
        "project_folder": c_files.getpath,
        "cal_file": c_files.getpath,
    }
    for key in c_files.keys():
        if key in getter:
            files[key] = getter[key](key)
            continue
        files[key] = c_files.get(key)
    return files


# =============================================================================
# Handle project folders to save logs
# =============================================================================
def _create_project_folders(dat_file: Path) -> tuple[str, str]:
    """Create a folder to store outputs and log messages."""
    dat_file = dat_file.absolute()
    project_folder = Path(
        dat_file.parent,
        datetime.datetime.now().strftime('%Y.%m.%d_%Hh%M_%Ss_%fms'))
    project_folder.mkdir()
    return str(dat_file), str(project_folder)
