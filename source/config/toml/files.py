#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the ``files``, set up ``logging`` module."""
import datetime
import logging
from pathlib import Path

from util.log_manager import set_up_logging


def test(dat_file: str,
         **files_kw: str) -> None:
    """Check that given ``dat_file`` exists, set up logging."""
    dat_file_as_path, project_folder = _create_project_folders(dat_file)
    assert dat_file_as_path.is_file()
    log_file = Path(project_folder, 'lightwin.log')
    set_up_logging(logfile_file=log_file)
    logging.info(f"Setting {project_folder = }\nSetting {log_file = }")


def edit_configuration_dict_in_place(files_kw: dict[str, str | Path]) -> None:
    """Set some useful paths."""
    dat_file, project_folder = _create_project_folders(files_kw['dat_file'])
    files_kw['dat_file'] = dat_file
    files_kw['project_folder'] = project_folder

    if 'cal_file' in files_kw:
        files_kw['cal_file'] = Path(files_kw['cal_file'])


def _create_project_folders(dat_file: str) -> tuple[Path, Path]:
    """Create a folder to store outputs and log messages."""
    dat_file_as_path = Path(dat_file)

    dat_file_as_path = dat_file_as_path.absolute()
    project_folder = Path(
        dat_file_as_path.parent,
        datetime.datetime.now().strftime('%Y.%m.%d_%Hh%M_%Ss_%fms'))
    project_folder.mkdir()

    return dat_file_as_path, project_folder
