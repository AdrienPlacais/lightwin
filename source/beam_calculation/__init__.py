"""
This folder holds all the modules that are related to computing the propagation
of the beam in the linac.

`BeamCalculator` is an Abstract Base Class to hold such a solver. `Envelope1D`
and `TraceWin` are the currently implemented solvers inheriting from it. They
can be created very easily using the `factory` module.

`output` module is used to uniformly store beam properties across all solvers.

In order to work, solvers rely on `SingleElementCalculatorParameters` to hold
the meshing, etc.

"""
