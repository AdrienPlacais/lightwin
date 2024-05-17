#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move all images in the same folder."""

import argparse
import shutil
from pathlib import Path


def _create_output_folder(simulation_folder: Path,
                          out_folder_name: str = "images"
                          ) -> Path:
    """Create the output folder."""
    out_images = simulation_folder / out_folder_name
    if out_images.exists():
        raise FileExistsError(
            f"{out_images = } already exists. Maybe you already run this "
            "script?"
        )
    out_images.mkdir(exist_ok=False)
    return out_images

def _move_single_simulation_output_figures(simulation_folder: Path,
                                           out_images: Path,
                                           verbose: bool = False) -> None:
    """Mpve the figures of a single simulation to ``out_images``."""
    simulation_name = simulation_folder.stem
    new_filename = simulation_name

    simulation_files = simulation_folder.glob("**/*")
    images = [
        x for x in simulation_files if x.is_file() and x.suffix == ".png"
    ]

    for image in images:
        new_folder = out_images / image.stem
        new_folder.mkdir(exist_ok=True)
        out_path = (new_folder / new_filename).with_suffix(".png")
        if verbose:
            print(f"{image} -> {out_path}")
        shutil.copy(image, out_path)


def reorder_output_figures(simulation_folder: Path | str,
                           verbose: bool = False
                           ) -> None:
    """Move all the output figures in a single folder."""
    if isinstance(simulation_folder, str):
        simulation_folder = Path(simulation_folder)

    simulations = simulation_folder.glob("*/")
    simulation_folders = [x for x in simulations if not x.is_file()]

    out_images = _create_output_folder(simulation_folder)

    for simulation_folder in simulation_folders:
        _move_single_simulation_output_figures(simulation_folder,
                                               out_images,
                                               verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("reorder_output_figures")
    parser.add_argument(
        "-f",
        "--folder",
        help="Folder where to look for output figures",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print origin and destination of files",
        action="store_true",  # on/off flag
        required=False,
    )
    args = parser.parse_args()
    reorder_output_figures(args.folder, args.verbose)
