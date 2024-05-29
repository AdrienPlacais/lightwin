"""Define functions to load and preprocess the TraceWin files."""

import itertools
import logging
import re
from collections.abc import Collection
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import Literal

import numpy as np

from core.instruction import Instruction

# Dict of data that can be imported from TW's "Data" table.
# More info in results
TRACEWIN_IMPORT_DATA_TABLE = {
    "v_cav_mv": 6,
    "phi_0_rel": 7,
    "phi_s": 8,
    "w_kin": 9,
    "beta": 10,
    "z_abs": 11,
    "phi_abs_array": 12,
}


def dat_file(
    dat_path: Path,
    *,
    keep: Literal["none", "comments", "empty lines", "all"] = "none",
    instructions_to_insert: Collection[Instruction] = (),
) -> list[list[str]]:
    """Load the dat file and convert it into a list of lines.

    Parameters
    ----------
    dat_path : Path
        Filepath to the ``.dat`` file, as understood by TraceWin.
    keep : {"none", "comments", "empty lines", "all"}, optional
        To determine which un-necessary lines in the dat file should be kept.
        The default is `'none'`.
    instructions_to_insert : Collection[Instruction], optional
        Some elements or commands that are not present in the ``.dat`` file but
        that you want to add. The default is an empty tuple.

    Returns
    -------
    dat_filecontent : list[list[str]]
        List containing all the lines of dat_path.

    """
    dat_filecontent = []

    with open(dat_path, "r", encoding="utf-8") as file:
        for line in file:
            sliced = slice_dat_line(line)

            if len(sliced) == 0:
                if keep in ("empty lines", "all"):
                    dat_filecontent.append(sliced)
                continue
            if line[0] == ";":
                if keep in ("comments", "all"):
                    dat_filecontent.append(sliced)
                continue
            dat_filecontent.append(sliced)
    if not instructions_to_insert:
        return dat_filecontent
    logging.info(
        "Will insert following instructions:\n{instructions_to_insert}"
    )
    for instruction in instructions_to_insert:
        instruction.insert(dat_filecontent=dat_filecontent)


def _strip_comments(line: str) -> str:
    """Remove comments from a line."""
    return line.split(";", 1)[0].strip()


def _split_named_elements(line: str) -> list[str]:
    """Split named elements from a line."""
    pattern = re.compile(r"(\w+)\s*:\s*(.*)")
    match = pattern.match(line)
    if match:
        return [match.group(1)] + match.group(2).split()
    return []


def _split_weighted_elements(line: str) -> list[str]:
    """Split weighted elements from a line."""
    pattern = re.compile(r"(\w+)\s*(\(\d+\.?\d*e?-?\d*\))?")
    matches = pattern.findall(line)
    result = []
    for match in matches:
        result.append(match[0])
        if match[1]:
            result.append(match[1].strip())
    return result


def slice_dat_line(line: str) -> list[str]:
    """Slices a .dat line into its components."""
    line = line.strip()
    if not line:
        return []

    if line.startswith(";"):
        return [";", line[1:].strip()]

    line = _strip_comments(line)

    named_elements = _split_named_elements(line)
    if named_elements:
        return named_elements

    if "(" in line:
        weighted_elements = _split_weighted_elements(line)
        if weighted_elements:
            return weighted_elements

    return line.split()


def table_structure_file(
    path: Path,
) -> list[list[str]]:
    """Load the file produced by ``Data`` ``Save table to file``."""
    file_content = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line_content = line.split()

            try:
                int(line_content[0])
            except ValueError:
                continue
            file_content.append(line_content)
    return file_content


def results(path: Path, prop: str) -> np.ndarray:
    """Load a property from TraceWin's "Data" table.

    Parameters
    ----------
    path : Path
        Path to results file. It must be saved from TraceWin:
        ``Data`` > ``Save table to file``.
    prop : str
        Name of the desired property. Must be in d_property.

    Returns
    -------
    data_ref: numpy array
        Array containing the desired property.

    """
    if not path.is_file():
        logging.warning(
            "Filepath to results is incorrect. Provide another one."
        )
        Tk().withdraw()
        path = Path(
            askopenfilename(filetypes=[("TraceWin energies file", ".txt")])
        )

    idx = TRACEWIN_IMPORT_DATA_TABLE[prop]

    data_ref = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                int(line.split("\t")[0])
            except ValueError:
                continue
            splitted_line = line.split("\t")
            new_data = splitted_line[idx]
            if new_data == "-":
                new_data = np.NaN
            data_ref.append(new_data)
    data_ref = np.array(data_ref).astype(float)
    return data_ref


def electric_field_1d(path: Path) -> tuple[int, float, float, np.ndarray, int]:
    """Load a 1D electric field (``.edz`` extension).

    Parameters
    ----------
    path : Path
        The path to the ``.edz`` file to load.

    Returns
    -------
    n_z : int
        Number of steps in the array.
    zmax : float
        z position of the filemap end.
    norm : float
        Electric field normalisation factor. It is different from k_e (6th
        argument of the FIELD_MAP command). Electric fields are normalised by
        k_e/norm, hence norm should be unity by default.
    f_z : np.ndarray
        Array of electric field in MV/m.
    n_cell : int
        Number of cells in the cavity.

    """
    n_z: int | None = None
    zmax: float | None = None
    norm: float | None = None

    f_z = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if i == 0:
                    line_splitted = line.split(" ")

                    # Sometimes the separator is a tab and not a space:
                    if len(line_splitted) < 2:
                        line_splitted = line.split("\t")

                    n_z = int(line_splitted[0])
                    # Sometimes there are several spaces or tabs between
                    # numbers
                    zmax = float(line_splitted[-1])
                    continue

                if i == 1:
                    norm = float(line)
                    continue

                f_z.append(float(line))
    except UnicodeDecodeError:
        logging.error(
            "File could not be loaded. Check that it is non-binary."
            "Returning nothing and trying to continue without it."
        )
        raise IOError()

    assert n_z is not None
    assert zmax is not None
    assert norm is not None
    n_cell = _get_number_of_cells(f_z)
    return n_z, zmax, norm, np.array(f_z), n_cell


def _get_number_of_cells(f_z: list[float]) -> int:
    """Count number of times the array of z-electric field changes sign.

    See `SO`_.

    .. _SO: https://stackoverflow.com/a/2936859/12188681

    """
    n_cell = len(list(itertools.groupby(f_z, lambda z: z > 0.0)))
    return n_cell


def transfer_matrices(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the transfer matrix as calculated by TraceWin."""
    transfer_matrices = []
    position_in_m = []
    elements_numbers = []

    with open(path, "r", encoding="utf-8") as file:
        lines = []
        for i, line in enumerate(file):
            lines.append(line)
            if i % 7 == 6:
                elements_numbers.append(int(lines[0].split()[1]))
                position_in_m.append(float(lines[0].split()[3]))
                transfer_matrices.append(_transfer_matrix(lines[1:]))
                lines = []
    elements_numbers = np.array(elements_numbers)
    position_in_m = np.array(position_in_m)
    transfer_matrices = np.array(transfer_matrices)
    return elements_numbers, position_in_m, transfer_matrices


def _transfer_matrix(lines: list[str]) -> np.ndarray:
    """Load a single element transfer matrix."""
    transfer_matrix = np.empty((6, 6), dtype=float)
    for i, line in enumerate(lines):
        transfer_matrix[i] = np.array(line.split(), dtype=float)
    return transfer_matrix


FIELD_MAP_LOADERS = {".edz": electric_field_1d}  #:
