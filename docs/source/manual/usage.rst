Usage
=====

.. toctree::
   :maxdepth: 4

.. todo::
   Ease re-usage of old simulations. Very easy, just use the ``recompute`` flag, override the ``run_with_this`` method, and user must provide a ``results`` dir.

.. todo::
   Create a real transfer matrix object.

General structure of the code
*****************************
The highest-level object is an :py:class:`.Accelerator`.
It is initialized thanks to a ``.dat`` file (the same format as TraceWin).
It's main purpose is to store a :class:`.ListOfElements`, which is a ``list`` containing all the :class:`.Element`\s of the ``.dat`` file.

The propagation of the beam through the accelerator is performed thanks to a :class:`.BeamCalculator`.
As for now, two different :class:`.BeamCalculator`\s are implemented:
   - :class:`.Envelope1D`, which computes the propagation of the beam in envelope and in 1D (longitudinal).
   - :class:`.Envelope3D`, which computes the propagation of the beam in envelope and in 3D.
   - :class:`.TraceWin`, which simply calls TraceWin from the command-line interface.

All :class:`.BeamCalculator`\s have a :meth:`.BeamCalculator.run` method, which perform the beam dynamics calculation along the linac; it takes in a :class:`.ListOfElements` and returns a :class:`.SimulationOutput`.
This last object contains all the useful information, such as kinetic energy along the linac.

Breaking and fixing a linac
***************************
The methods to break -- and then fix -- a linac are stored in the :class:`.Fault` objects, gathered in :class:`.FaultScenario`.
A :class:`.Fault` is composed of one or several failed cavities that are fixed together.
A :class:`.FaultScenario` is composed of one or several :class:`.Fault` happening at the same time.

The compensation is realized by an :class:`.OptimisationAlgorithm`.
It will try to find the *best* :class:`.Variable`\s that match the :class:`.Objective`\s while respecting the :class:`.Constraint`\s.
Under the hood, it converts at each iteration the list of :class:`.Variable`\s into a :class:`.SetOfCavitySettings`.
The latter is given as argument to the :meth:`.BeamCalculator.run_with_this` which gives a :class:`.SimulationOutput` from which the :class:`.Objective`\s are evaluated.

.. note::
   See example.

