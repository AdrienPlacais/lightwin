#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define :class:`Diagnostic`.

As for now, diagnostics are not used by LightWin. However, LightWin can add
diagnostics (as well as ADJUST) to the final ``.dat`` in order to perform a
"beauty pass".

.. note::
    Functionalities still under implementation. In particular, the number of
    attributes were not checked.

"""

from core.elements.element import Element


class Diagnostic(Element):
    """A dummy object."""

    base_name = "D"
    increment_lattice_idx = False
    is_implemented = False

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx, name)
        self.length_m = 0.0
        self.number = int(line[1])


class DiagCurrent(Diagnostic):
    """Measure current."""


class DiagDCurrent(Diagnostic):
    """Measure delta current."""


class DiagPosition(Diagnostic):
    """Measure position."""


class DiagDPosition(Diagnostic):
    """Measure delta position."""


class DiagDivergence(Diagnostic):
    """Measure divergences."""


class DiagDDivergence(Diagnostic):
    """Measure delta divergences."""


class DiagSizeFWHM(Diagnostic):
    """Measure full width at half maximum."""


class DiagSize(Diagnostic):
    """Measure sizes."""


class DiagSizeP(Diagnostic):
    """Measure divergences."""


class DiagDSizeFWHM(Diagnostic):
    """Measure delta full width at half maximum."""


class DiagDSize(Diagnostic):
    """Measure delta size."""


class DiagDSize2FWHM(Diagnostic):
    """Measure delta full width at half maximum between two positions."""


class DiagDSize2(Diagnostic):
    """Measure delta size between two positions."""

    is_implemented = True
    n_attributes = (3, 4)

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx, name)
        self.x_rms_beam_delta_size = float(line[2])
        self.y_rms_beam_delta_size = float(line[3])

        if len(line) == 5:
            self.accuracy = float(line[4])


class DiagDSize3(Diagnostic):
    """Measure delta phase spread between two positions."""

    is_implemented = True
    n_attributes = (3, 4)

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx, name)
        self.rms_delta_phase_spread = float(line[2])
        self.accuracy = float(line[3])

        if len(line) == 5:
            self.low_pass_filter_frequency = float(line[4])


class DiagDPSize2(Diagnostic):
    """Measure delta divergence between two positions."""


class DiagPhase(Diagnostic):
    """Measure phase."""

    n_attributes = 2


class DiagEnergy(Diagnostic):
    """Measure energy."""

    n_attributes = 3


class DiagDEnergy(Diagnostic):
    """Measure difference between beam energy and perfect linac energy."""

    n_attributes = 3


class DiagDPhase(Diagnostic):
    """Measure difference between beam phase and perfect linac phase."""

    n_attributes = 2


class DiagLuminosity(Diagnostic):
    """Measure luminosity."""

    n_attributes = 3


class DiagWaist(Diagnostic):
    """Measure waist setting."""

    n_attributes = 4


class DiagAchromat(Diagnostic):
    """Measure achromat setting."""

    n_attributes = 5


class DiagEmit(Diagnostic):
    """Measure RMS emittance setting."""

    n_attributes = 4


class DiagEmit99(Diagnostic):
    """Measure 99% emittance setting."""

    n_attributes = 4


class DiagHalo(Diagnostic):
    """Measure halo setting."""

    n_attributes = 4


class DiagSetMatrix(Diagnostic):
    """Measure transfer matrix setting."""

    n_attributes = 6


class DiagTwiss(Diagnostic):
    """Measure beam Twiss parameters settings."""

    n_attributes = 7


class DiagDTwiss(Diagnostic):
    """Make equal two beam Twiss parameters between two positions or more."""

    n_attributes = 7


class DiagDTwiss2(Diagnostic):
    """Make equal transverse Twiss parameters at diagnostic position."""

    n_attributes = 3


class DiagSeparation(Diagnostic):
    """Measure beam separation setting."""

    n_attributes = 6


class DiagSizeMax(Diagnostic):
    """Limit beam size max."""

    n_attributes = 6


class DiagSizeMin(DiagSizeMax):
    """Limit beam size min."""


class DiagPhaseAdv(Diagnostic):
    """Measure beam phase advance."""

    n_attributes = 4


class DiagBeta(Diagnostic):
    """Measure beam beta."""

    n_attributes = 6


class DiagDBeta(Diagnostic):
    """Measure delta beam beta."""

    n_attributes = 4


IMPLEMENTED_DIAGNOSTICS = {
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
}
