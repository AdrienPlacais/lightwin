Usage
=====

.. toctree::
   :maxdepth: 4

.. todo::
   Ease re-usage of old simulations. Very easy to implement, just use the ``recompute`` flag, override the ``run_with_this`` method, and user must provide a ``results`` directory.

General structure of the code
-----------------------------
The highest-level object is an :class:`.Accelerator`.
It is initialized thanks to a ``.dat`` file (the same format as TraceWin).
Its main purpose is to store a :class:`.ListOfElements`, which is a ``list`` containing all the :class:`.Element`\s of the ``.dat`` file.

The propagation of the beam through the accelerator is performed thanks to a :class:`.BeamCalculator`.
As for now, three different :class:`.BeamCalculator`\s are implemented:
   - :class:`.Envelope1D`, which computes the propagation of the beam in envelope and in 1D (longitudinal).
   - :class:`.Envelope3D`, which computes the propagation of the beam in envelope and in 3D.
   - :class:`.TraceWin`, which simply calls TraceWin from the command-line interface.

All :class:`.BeamCalculator`\s have a :meth:`.BeamCalculator.run` method, which perform the beam dynamics calculation along the linac; it takes in a :class:`.ListOfElements` and returns a :class:`.SimulationOutput`.
This last object contains all the useful information, such as kinetic energy along the linac.

Breaking and fixing a linac
---------------------------
The methods to break -- and then fix -- a linac are stored in the :class:`.Fault` objects, gathered in :class:`.FaultScenario`.
A :class:`.Fault` is composed of one or several failed cavities that are fixed together.
A :class:`.FaultScenario` is composed of one or several :class:`.Fault` happening at the same time.

The compensation is realized by an :class:`.OptimisationAlgorithm`.
It will try to find the *best* :class:`.Variable`\s that match the :class:`.Objective`\s while respecting the :class:`.Constraint`\s.
Under the hood, it converts at each iteration the list of :class:`.Variable`\s into a :class:`.SetOfCavitySettings`.
The latter is given as argument to the :meth:`.BeamCalculator.run_with_this` which gives a :class:`.SimulationOutput` from which the :class:`.Objective`\s are evaluated.

.. _TraceWin compatibility note:

Compatibility with TraceWin `.dat` files
----------------------------------------
LightWin uses the same format as TraceWin for the linac structure.
As TraceWin developers implemented a significant number of elements and commands, those will be progressively implemented in LightWin too.

"Useless" commands and elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some instructions will raise a warning, even if they will not influence the results.
As an example, if you use :class:`.Envelope1D`, transverse dynamics are not considered and the fact that transverse field maps are not implemented should not be a problem.

"Useful" commands and elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You should clean the `.dat` to remove any command that influences the design of the linac.
In particular: `SET_ADV`, `SET_SYNC_PHASE`, `ADJUST` commands.
Warnings may not always appear, so be careful that :class:`.Envelope1D` or :class:`.Envelope3D` match with TraceWin.
If you choose :class:`.TraceWin` solver for the optimization, both LightWin and TraceWin could modify the design of the linac at the same time, so unexpected side effects may appear.

.. note::
   Since `0.6.21`, `SET_SYNC_PHASE` commands can be kept in the original `.dat`.
   The output `.dat` will contain relative or absolute phase, according to the corresponding :attr:`.BeamCalculator.reference_phase`.
   In the future, it will be possible to export `.dat` with `SET_SYNC_PHASE` for all cavities, or to keep the phase definitions of the original `.dat`.

   See also: :meth:`.ListOfElements.store_settings_in_dat` (the method which is actually called to create the `.dat`).

How to implement commands or elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can implement the desired elements and `git push` them, file an issue on GitHub or send me an `email`_ and I will try to add the desired element(s) as soon as possible.

.. note::
   See example.

.. warning::
   Field maps file formats must be ascii, binary files to handled yet

.. _email: mailto:placais@lpsc.in2p3.fr
