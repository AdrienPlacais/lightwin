"""Save all the cavity settings.

Example to tailor to your needs.

"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from core.elements.field_maps.field_map import FieldMap
from core.list_of_elements.list_of_elements import ListOfElements
from failures.fault_scenario import FaultScenario
from util.pandas_helper import to_csv


def save_cavity_settings(
    fault_scenarios: Sequence[FaultScenario], filename: Path | str
) -> None:
    """Save all the settings in a single file."""
    path = (
        fault_scenarios[0].ref_acc.elts.files["accelerator_path"].parent
        / filename
    )
    nominal_cavities = fault_scenarios[0].ref_acc.elts.l_cav
    all_df = [
        _settings_as_df(scenario.fix_acc.elts, nominal_cavities)
        for scenario in fault_scenarios
    ]
    final_df = pd.concat(all_df)
    to_csv(final_df, path)


def _settings_as_df(
    elts: ListOfElements,
    nominal_cavities: Iterable[FieldMap],
) -> pd.DataFrame:
    """Give the settings of a single scenario as a pd df."""
    cavities = elts.l_cav

    as_dict = {
        "Scenario ID": elts.files["accelerator_path"].stem,
        "Cavity name": [x.name for x in cavities],
        "Status": [x.status for x in cavities],
        "phi_0_rel nominal [deg]": _phases_in_deg(nominal_cavities),
        "k_e nominal [1]": _k_e(nominal_cavities),
        "phi_0_rel [deg]": _phases_in_deg(cavities),
        "k_e [1]": _k_e(cavities),
    }
    as_df = pd.DataFrame(as_dict)
    return as_df


def _phases_in_deg(cavities: Iterable[FieldMap]) -> np.ndarray:
    """Get all relative phases in degrees."""
    phases = np.array(
        [
            (
                x.cavity_settings.phi_0_rel
                if x.cavity_settings.phi_0_rel is not None
                else np.NaN
            )
            for x in cavities
        ]
    )
    return np.rad2deg(phases)


def _k_e(cavities: Iterable[FieldMap]) -> list[float]:
    """Return all k_e."""
    return [x.cavity_settings.k_e for x in cavities]
