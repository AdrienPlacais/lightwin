#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:44:00 2021.

@author: placais

In this module, two classes are defined:
    - ParticleInitialState, which is just here to save the position and energy
    of a particle at the entrance of the linac. Saved as an Accelerator
    attribute.
    - ParticleFullTrajectory, which saves the energy, phase, position of a
    particle along the linac. As a single ParticleInitialState can lead to
    several ParticleFullTrajectory (according to size of the mesh, the solver,
    etc), ParticleFullTrajectory are stored in SimulationOutput.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np

from util.helper import recursive_items, recursive_getter, range_vals
import util.converters as convert


@dataclass
class ParticleInitialState:
    """
    Hold the initial energy/phase of a particle, and if it is synchronous.

    It is stored in Accelerator, and is parent of ParticleFullTrajectory.

    """

    w_kin: float | np.ndarray | list
    phi_abs: float | np.ndarray | list
    synchronous: bool


@dataclass
class ParticleFullTrajectory(ParticleInitialState):
    """
    Hold the full energy, phase, etc of a particle.

    It is stored in a SimulationOutput. A single Accelerator can have several
    SimulationOutput, hence an Accelerator.ParticleInitialState can have
    several SimulationOutput.ParticleFullTrajectory.

    Phase is defined as:
        phi = omega_0_bunch * t
    while in electric_field it is:
        phi = omega_0_rf * t

    """

    def __post_init__(self):
        """Ensure that LightWin has everything it needs, with proper format."""
        if isinstance(self.phi_abs, list):
            self.phi_abs = np.array(self.phi_abs)

        if isinstance(self.w_kin, list):
            self.w_kin = np.array(self.w_kin)

        self.gamma = convert.energy(self.get('w_kin'), "kin to gamma")
        self.beta: np.ndarray

    def __str__(self) -> str:
        """Show amplitude of phase and energy."""
        out = "\tParticleFullTrajectory:\n"
        out += "\t\t" + range_vals("w_kin", self.w_kin)
        out += "\t\t" + range_vals("phi_abs", self.phi_abs)
        return out

    def compute_complementary_data(self):
        """Compute some data necessary to do the post-treatment."""
        self.beta = convert.energy(self.get('gamma'), "gamma to beta")

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: tuple[str], to_deg: bool = False, **kwargs: dict
            ) -> tuple[Any]:
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)

            if val[key] is not None and to_deg and 'phi' in key:
                val[key] = np.rad2deg(val[key])

        out = [val[key] for key in keys]

        if len(out) == 1:
            return out[0]
        return tuple(out)


# def create_rand_particles(e_0_mev):
#     """Create two random particles."""
#     delta_z = 1e-4
#     delta_E = 1e-4

#     rand_1 = Particle(-1.42801442802603928417e-04,
#                       1.66094219207764304258e+01,)
#     rand_2 = Particle(2.21221539793564048182e-03,
#                       1.65923664093018210508e+01,)

#     # rand_1 = Particle(
#     #     random.uniform(0., delta_z * .5),
#     #     random.uniform(e_0_mev,  e_0_mev + delta_E * .5),
#     #     omega0_bunch)

#     # rand_2 = Particle(
#     #     random.uniform(-delta_z * .5, 0.),
#     #     random.uniform(e_0_mev - delta_E * .5, e_0_mev),
#     #     omega0_bunch)

#     return rand_1, rand_2
