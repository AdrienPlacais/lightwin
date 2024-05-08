#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define holds function to load, modify and create .dat structure files.

.. todo::
    Insert line skip at each section change in the output.dat

"""
import logging
from collections.abc import Collection, Container, Iterable, Sequence
from pathlib import Path

from core.commands.command import Command
from core.elements.element import Element
from core.instruction import Dummy, Instruction


def dat_filecontent_from_smaller_list_of_elements(
    original_instructions: Sequence[Instruction],
    elts: Collection[Element],
) -> tuple[list[list[str]], list[Instruction]]:
    """
    Create a ``.dat`` with only elements of ``elts`` (and concerned commands).

    Properties of the FIELD_MAP, i.e. amplitude and phase, remain untouched.

    """
    indexes_to_keep = [elt.get("dat_idx", to_numpy=False) for elt in elts]
    last_index = indexes_to_keep[-1] + 1

    new_dat_filecontent: list[list[str]] = []
    new_instructions: list[Instruction] = []
    for instruction in original_instructions[:last_index]:
        if not (
            _is_needed_element(instruction, indexes_to_keep)
            or _is_useful_command(instruction, indexes_to_keep)
        ):
            continue

        new_dat_filecontent.append(instruction.line)
        new_instructions.append(instruction)

    end = original_instructions[-1]
    new_dat_filecontent.append(end.line)
    new_instructions.append(end)
    return new_dat_filecontent, new_instructions


def _is_needed_element(
    instruction: Instruction, indexes_to_keep: Container[int]
) -> bool:
    """Tell if the instruction is an element that we must keep."""
    if not isinstance(instruction, Element | Dummy):
        return False
    if instruction.idx["dat_idx"] in indexes_to_keep:
        return True
    return False


def _is_useful_command(
    instruction: Instruction, indexes_to_keep: Iterable[int]
) -> bool:
    """Tell if the current command has an influence on our elements."""
    if not isinstance(instruction, Command):
        return False
    if instruction.concerns_one_of(indexes_to_keep):
        return True
    return False


def export_dat_filecontent(
    dat_content: Collection[Collection[str]], dat_path: Path
) -> None:
    """Save the content of the updated dat to a ``.dat``.

    Parameters
    ----------
    dat_content : Collection[Collection[str]]
        Content of the ``.dat``, line per line, word per word.
    dat_path : Path
        Where to save the ``.dat``.

    """
    with open(dat_path, "w", encoding="utf-8") as file:
        for line in dat_content:
            file.write(" ".join(line) + "\n")
    logging.info(f"New dat saved in {dat_path}.")
