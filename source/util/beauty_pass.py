"""Define utility functions to perform a "beauty pass".

After a LightWin optimisation, perform a second optimisation with TraceWin. As
for now, the implementation is kept very simple:

    - The phase of compensating cavities can be retuned at +/- ``tol_phi_deg``
    around their compensated value.
    - The amplitude of compensating cavities can be retuned at +/- ``tol_k_e``
    around their compensated value.
    - We try to keep the phase dispersion between start of compensation zone,
    and ``number_of_dsize`` lattices after.

"""

import logging
import math
from collections.abc import Collection

from core.commands.adjust import Adjust
from core.elements.diagnostic import DiagDSize3, Diagnostic
from core.elements.element import Element
from core.elements.field_maps.cavity_settings import CavitySettings
from core.instruction import Instruction
from core.list_of_elements.list_of_elements import ListOfElements
from failures.fault import Fault
from failures.fault_scenario import FaultScenario
from failures.helper import nested_containing_desired
from failures.set_of_cavity_settings import SetOfCavitySettings
from util.helper import flatten


def _cavity_settings_to_adjust(
    cavity_settings: CavitySettings,
    dat_idx: int,
    number: int,
    tol_phi_deg: float = 5,
    tol_k_e: float = 0.05,
    link_index: int = 0,
    phase_nature: str = "",
) -> tuple[Adjust, Adjust] | tuple[Adjust, Adjust, Adjust]:
    """Create a ADJUST command with small bounds around current value."""
    if not phase_nature:
        phase_nature = cavity_settings.reference
    assert (
        phase_nature != "phi_s"
    ), "Adjusting synchronous phase won't do with TraceWin."

    phase = getattr(cavity_settings, phase_nature)
    assert isinstance(phase, float)
    phase = math.degrees(phase)
    line_phi = (
        f"ADJUST {number} 3 0 {phase - tol_phi_deg} {phase + tol_phi_deg}"
    )

    k_e = cavity_settings.k_e
    line_k_e = (
        f"ADJUST {number} 5 {link_index} {k_e - tol_k_e} {k_e + tol_k_e}"
    )

    adjust_phi = Adjust(line_phi.split(), dat_idx)
    adjust_k_e = Adjust(line_k_e.split(), dat_idx)
    if not link_index:
        return adjust_phi, adjust_k_e
    line_k_g = f"ADJUST {number} 6 {link_index}"
    return adjust_phi, adjust_k_e, Adjust(line_k_g.split(), dat_idx)


def set_of_cavity_settings_to_adjust(
    set_of_cavity_settings: SetOfCavitySettings,
    number: int,
    tol_phi_deg: float = 5,
    link_k_g: bool = False,
    tol_k_e: float = 0.05,
    phase_nature: str = "phi_0_rel",
) -> list[Adjust]:
    """Create adjust commands for every compensating cavity."""
    commands = [
        _cavity_settings_to_adjust(
            cavity_settings,
            cavity.idx["dat_idx"],
            number,
            link_index=i if link_k_g else 0,
            tol_phi_deg=tol_phi_deg,
            tol_k_e=tol_k_e,
            phase_nature=phase_nature,
        )
        for i, (cavity, cavity_settings) in enumerate(
            set_of_cavity_settings.items()
        )
    ]
    return [x for x in flatten(commands)]


def elements_to_diagnostics(
    elts: ListOfElements,
    failed: Collection[Element],
    number: int,
    number_of_dsize: int,
) -> list[Diagnostic]:
    """Create the DSize3 commands that will be needed."""
    lattices = elts.by_lattice
    failed_lattices = nested_containing_desired(lattices, failed)

    first_failed, last_failed = failed_lattices[0], failed_lattices[-1]
    assert isinstance(last_failed, list)
    post_failure = lattices[lattices.index(last_failed) :]

    dzise_elements = (
        first_failed[0],
        *[lattice[0] for lattice in post_failure[:number_of_dsize]],
    )

    dsize_args = (
        (f"DSIZE3 {number} 0 0".split(), elt.idx["dat_idx"])
        for elt in dzise_elements
    )
    dsizes = [DiagDSize3(*args) for args in dsize_args]
    return dsizes


def beauty_pass_instructions(
    fault_scenario: FaultScenario,
    number_of_dsize: int,
    number: int = 666333,
    link_k_g: bool = True,
) -> list[Instruction]:
    """Perform a beauty pass."""
    if len(fault_scenario) > 1:
        raise NotImplementedError(
            "Not sure how multiple faults would interact."
        )
    fault: Fault = fault_scenario[0]
    elts = fault_scenario.fix_acc.elts
    failed = fault.failed_elements
    set_of_cavity_settings = fault.optimized_cavity_settings

    diagnostics = elements_to_diagnostics(
        elts, failed, number=number, number_of_dsize=number_of_dsize
    )

    adjusts = set_of_cavity_settings_to_adjust(
        set_of_cavity_settings, number=number, link_k_g=link_k_g
    )
    if len(adjusts) < 2:
        logging.error(
            f"Not enough DIAG_DSIZE3 in {failed = } for beauty pass."
        )
        return []
    return [*diagnostics, *adjusts]
