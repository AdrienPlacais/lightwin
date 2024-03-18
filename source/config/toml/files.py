#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the ``files``, set up ``logging`` module."""
import datetime
import logging
from pathlib import Path

from util.log_manager import set_up_logging


def test(dat_file: str,
         project_folder: str = '',
         **files_kw: str) -> None:
    """Check that given ``dat_file`` exists, set up logging."""
    dat_path, project_path = _create_project_folders(dat_file,
                                                     project_folder,
                                                     **files_kw)
    assert dat_path.is_file(), f"{dat_file} does not exist."

    log_file = Path(project_path, 'lightwin.log')
    set_up_logging(logfile_file=log_file)

    logging.info(f"Setting {project_path = }\nSetting {log_file = }")


def edit_configuration_dict_in_place(files_kw: dict[str, str | Path]) -> None:
    """Set some useful paths."""
    dat_file = files_kw['dat_file']
    assert isinstance(dat_file, str)
    project_folder = files_kw.get('project_folder', '')
    assert isinstance(project_folder, str)

    dat_path, project_path = _create_project_folders(dat_file, project_folder)
    files_kw['dat_file'] = dat_path
    files_kw['project_folder'] = project_path

    if 'cal_file' in files_kw:
        files_kw['cal_file'] = Path(files_kw['cal_file']).resolve().absolute()


def _create_project_folders(dat_file: str,
                            project_folder: str = '',
                            **files_kw: str) -> tuple[Path, Path]:
    """Create a folder to store outputs and log messages."""
    dat_path = Path(dat_file).resolve().absolute()

    if project_folder:
        project_path = Path(project_folder).resolve().absolute()
        exist_ok = True

    else:
        project_path = Path(
            dat_path.parent,
            datetime.datetime.now().strftime('%Y.%m.%d_%Hh%M_%Ss_%fms'))
        exist_ok = False

    project_path.mkdir(exist_ok=exist_ok)
    return dat_path, project_path
