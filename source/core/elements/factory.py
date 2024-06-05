"""Define a class to easily create :class:`.Element` objects."""

from pathlib import Path
from typing import Any

from core.elements.aperture import Aperture
from core.elements.bend import Bend
from core.elements.diagnostic import (
    DiagAchromat,
    DiagBeta,
    DiagCurrent,
    DiagDBeta,
    DiagDCurrent,
    DiagDDivergence,
    DiagDEnergy,
    DiagDivergence,
    DiagDPhase,
    DiagDPosition,
    DiagDPSize2,
    DiagDSize,
    DiagDSize2,
    DiagDSize2FWHM,
    DiagDSize3,
    DiagDSize4,
    DiagDSizeFWHM,
    DiagDTwiss,
    DiagDTwiss2,
    DiagEmit,
    DiagEmit99,
    DiagEnergy,
    DiagHalo,
    DiagLuminosity,
    DiagPhase,
    DiagPhaseAdv,
    DiagPosition,
    DiagSeparation,
    DiagSetMatrix,
    DiagSize,
    DiagSizeFWHM,
    DiagSizeMax,
    DiagSizeMin,
    DiagSizeP,
    DiagTwiss,
    DiagWaist,
)
from core.elements.drift import Drift
from core.elements.dummy import DummyElement
from core.elements.edge import Edge
from core.elements.element import Element
from core.elements.field_maps.factory import FieldMapFactory
from core.elements.field_maps.field_map import FieldMap
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid
from core.elements.thin_steering import ThinSteering

implemented_elements = {
    "APERTURE": Aperture,
    "BEND": Bend,
    "DIAG_CURRENT": DiagCurrent,
    "DIAG_DCURRENT": DiagDCurrent,
    "DIAG_POSITION": DiagPosition,
    "DIAG_DPOSITION": DiagDPosition,
    "DIAG_DIVERGENCE": DiagDivergence,
    "DIAG_DDIVERGENCE": DiagDDivergence,
    "DIAG_SIZE_FWHM": DiagSizeFWHM,
    "DIAG_SIZE": DiagSize,
    "DIAG_SIZEP": DiagSizeP,
    "DIAG_DSIZE__FWHM": DiagDSizeFWHM,
    "DIAG_DSIZE": DiagDSize,
    "DIAG_DSIZE2_FWHM": DiagDSize2FWHM,
    "DIAG_DSIZE2": DiagDSize2,
    "DIAG_DSIZE3": DiagDSize3,
    "DIAG_DSIZE4": DiagDSize4,
    "DIAG_DPSIZE2": DiagDPSize2,
    "DIAG_PHASE": DiagPhase,
    "DIAG_ENERGY": DiagEnergy,
    "DIAG_DENERGY": DiagDEnergy,
    "DIAG_DPHASE": DiagDPhase,
    "DIAG_LUMINOSITY": DiagLuminosity,
    "DIAG_WAIST": DiagWaist,
    "DIAG_ACHROMAT": DiagAchromat,
    "DIAG_EMIT": DiagEmit,
    "DIAG_EMIT_99": DiagEmit99,
    "DIAG_HALO": DiagHalo,
    "DIAG_SET_MATRIX": DiagSetMatrix,
    "DIAG_TWISS": DiagTwiss,
    "DIAG_DTWISS": DiagDTwiss,
    "DIAG_DTWISS2": DiagDTwiss2,
    "DIAG_SEPARATION": DiagSeparation,
    "DIAG_SIZE_MAX": DiagSizeMax,
    "DIAG_SIZE_MIN": DiagSizeMin,
    "DIAG_PHASE_ADV": DiagPhaseAdv,
    "DIAG_BETA": DiagBeta,
    "DIAG_DBETA": DiagDBeta,
    "DRIFT": Drift,
    "DUMMY_ELEMENT": DummyElement,
    "EDGE": Edge,
    "FIELD_MAP": FieldMap,  # replaced in ElementFactory initialisation
    "QUAD": Quad,
    "SOLENOID": Solenoid,
    "THIN_STEERING": ThinSteering,
}  #:


class ElementFactory:
    """An object to create :class:`.Element` objects."""

    def __init__(
        self,
        default_field_map_folder: Path,
        freq_bunch_mhz: float,
        **factory_kw: Any,
    ) -> None:
        """Create a factory for the field maps."""
        field_map_factory = FieldMapFactory(
            default_field_map_folder, freq_bunch_mhz, **factory_kw
        )
        self.field_map_factory = field_map_factory
        implemented_elements["FIELD_MAP"] = field_map_factory.run

    def run(self, line: list[str], dat_idx: int, **kwargs) -> Element:
        """Call proper constructor."""
        name, line = self._personalized_name(line)
        element_constructor = _get_constructor(line[0])
        element = element_constructor(line, dat_idx, name, **kwargs)
        return element

    def _personalized_name(
        self, line: list[str]
    ) -> tuple[str | None, list[str]]:
        """Extract the user-defined name of the element if there is one.

        .. todo::
            Make this robust.

        """
        original_line = " ".join(line)
        line_delimited_with_name = original_line.split(":", maxsplit=1)

        if len(line_delimited_with_name) == 2:
            name = line_delimited_with_name[0].strip()
            cleaned_line = line_delimited_with_name[1].split()
            return name, cleaned_line

        return None, line


def _get_constructor(first_word: str) -> type:
    """Get the proper constructor."""
    key = first_word.upper()
    if key in implemented_elements:
        return implemented_elements[key]
    raise IOError(f"No Element matching {key} was found.")
