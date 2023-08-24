"""
This package holds all the modules that are related to computing the
propagation of the beam in the linac.

:py:class:`BeamCalculator` is an Abstract Base Class to hold such a solver.
:Class:`Envelope1D` and :class:`TraceWin` are the currently implemented solvers
inheriting from it. They can be created very easily using the :mod:`factory`
module.

mod:`output` module is used to uniformly store beam properties across all
solvers.

In order to work, solvers rely on :class:`SingleElementCalculatorParameters` to
hold the meshing, etc.

"""
