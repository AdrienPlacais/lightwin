#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define functions to compare several runs using ``evaluations.csv`` files.

.. todo::
    Set up an example.

"""

from pathlib import Path

import pandas as pd


def _to_df(
    simulation_id: str, evaluation_path: Path, evaluation_namecol: str
) -> pd.DataFrame:
    """
    Load the file and rename column header.

    Parameters
    ----------
    simulation_id : str
        Name that will be given to the simulation in the plot.
    evaluation_path : Path
        Where the ``evaluations.csv`` file is.
    evaluation_namecol : str
        Name of the column in the ``evaluations.csv`` file.

    Returns
    -------
    pd.DataFrame
        Holds all the values of ``evaluation_namecol`` in ``evaluation_path``;
        new name of the column is ``simulation_id``.

    """
    df = pd.read_csv(
        evaluation_path / "evaluations.csv", usecols=(evaluation_namecol,)
    )
    new_name = f"{simulation_id}: (mean {df.mean().iloc[0]:.2f} std {df.std().iloc[0]:.2f})"
    df.rename(columns={evaluation_namecol: new_name}, inplace=True)
    return df


def _compare_one_quantity_all_simulations(
    simulation_ids_and_paths: dict[str, Path],
    evaluation_namecol: str,
    y_label: str,
) -> None:
    """
    Load and plot all the given files.

    Parameters
    ----------
    simulation_ids_and_paths : dict[str, Path]
        Keys are name of the simulation in the plot, values are corresponding
        ``evaluations.csv`` path.
    evaluation_namecol : str
        Name of the column to take in ``evaluations.csv``.
    y_label : str
        Name of the y-axis in the plot.

    """
    all_df = [
        _to_df(study_case, path, evaluation_namecol)
        for study_case, path in simulation_ids_and_paths.items()
    ]
    df = pd.concat(all_df, axis=1)
    axes = df.plot(
        ylabel=y_label, grid=True, figsize=(18, 9), marker="o", alpha=0.7
    )
    fig = axes.get_figure()
    filename = "comparisons/" + y_label.replace(" ", "_") + ".png"
    fig.savefig(filename)


def compare_simulations(
    simulation_ids_and_paths: dict[str, Path], studies: dict[str, str]
) -> None:
    """Perform all defined studies.

    Parameters
    ----------
    simulation_ids_and_paths : dict[str, Path]
        Keys are name of the simulation in the plot, values are corresponding
        ``evaluations.csv`` path.
    studies : dict[str, str]
        Keys are the y-labels of the plots, values are corresponding column
        names in ``evaluations.csv.``

    Examples
    --------
    >>> simulation_ids_and_paths = {"leapfrog": "run_1/evaluations.csv",
                                    "downhill": "run2/evaluations.csv"}
    >>> studies = {"Lost power [W]", "lost power shall be null",
                   "Relative emittance increase", "emittance increase"}
    >>> compare_simulations(simulation_ids_and_paths, studies)

    """
    for y_label, namecol in studies.items():
        _compare_one_quantity_all_simulations(
            simulation_ids_and_paths, namecol, y_label
        )
