#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create :class:`.CavitySettings` from various contexts."""
from abc import ABC
import math
from collections.abc import Sequence
from typing import Callable

import numpy as np

from core.elements.field_maps.cavity_settings import CavitySettings


class ICavitySettingsFactory(ABC):
    """Base class to create :class:`CavitySettings` objects."""

    def __init__(self, freq_bunch_mhz: float) -> None:
        """Instantiate factory, with attributes common to all cavities."""
        self.freq_bunch_mhz = freq_bunch_mhz

    def from_line_in_dat_file(self,
                              line: list[str],
                              set_sync_phase: bool = False,
                              ) -> CavitySettings:
        """Create the cavity settings as read in the ``.dat`` file."""
        k_e = float(line[6])
        phi_0 = math.radians(float(line[3]))
        reference = self._reference(bool(int(line[10])), set_sync_phase)
        status = 'nominal'

        cavity_settings = CavitySettings(k_e,
                                         phi_0,
                                         reference,
                                         status,
                                         self.freq_bunch_mhz,
                                         )
        return cavity_settings

    def from_optimisation_algorithm(
            self,
            var: np.ndarray,
            reference: str,
            freq_cavities_mhz: Sequence[float],
            status: str,
            transf_mat_func_wrappers: Sequence[dict[str, Callable]],
    ) -> list[CavitySettings]:
        """Create the cavity settings to try during an optimisation."""
        amplitudes = list(var[var.shape[0] // 2:])
        phases = list(var[:var.shape[0] // 2])
        variables = zip(amplitudes, phases, freq_cavities_mhz,
                        transf_mat_func_wrappers, strict=True)

        several_cavity_settings = [
            CavitySettings(k_e, phi, reference, status, self.freq_bunch_mhz,
                           freq_cavity_mhz, wrapper)
            for k_e, phi, freq_cavity_mhz, wrapper in variables]
        return several_cavity_settings

    def _reference(self,
                   absolute_phase_flag: bool,
                   set_sync_phase: bool,
                   ) -> str:
        """Determine which phase will be the reference one."""
        if set_sync_phase:
            return 'phi_s'
        if absolute_phase_flag:
            return 'phi_0_abs'
        return 'phi_0_rel'
