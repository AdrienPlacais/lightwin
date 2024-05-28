.. _units-label:

Units and conventions
=====================
.. toctree::
   :maxdepth: 4

.. todo::
   Consistency between font size of normal text and math text. Not aesthetic when I have text and units side to side.

.. todo::
   General units: MeV etc

.. _units-beam-parameters-label:

Beam parameters
***************
The beam parameters are defined in :mod:`core.beam_parameters` and stored in the :class:`.BeamParameters` object.
We use the same units and conventions as TraceWin.

RMS emittances
--------------

.. csv-table::
    :file: units/emittances.csv
    :widths: 30, 30, 30, 10
    :header-rows: 1

.. warning::
   In TraceWin's output files ``partran.out`` and ``tracewin.out``, ``eps_zdelta`` is expressed in :math:`\pi\mathrm{.mm.mrad}`, not in :math:`\pi\mathrm{mm.\%}`!
   The conversion is: :math:`1\pi\mathrm{.mm.mrad} = 10\pi\mathrm{.mm.\%}`

Twiss
-----

.. csv-table::
    :file: units/twiss.csv
    :widths: 33, 33, 33
    :header-rows: 1

Note that ``beta_kin``, ``gamma_kin`` are the Lorentz factors.

Envelopes
---------

.. csv-table::
    :file: units/envelopes.csv
    :widths: 33, 33, 33
    :header-rows: 1

.. note::
    Envelopes are at :math:`1\sigma` in LightWin, while they are plotted at
    :math:`6\sigma` by default in TraceWin.

.. note::
    Envelopes are calculated with un-normalized emittances in the
    :math:`[z-\delta]` and :math:`[z-z']` planes, but they are calculated with
    normalized emittance in the :math:`[\phi-W]` plane.

:math:`\sigma` beam matrix
--------------------------
We save this matrix in SI units, *i.e.* in :math:`\mathrm{m}` and :math:`\mathrm{rad}`.
When given in the ``.ini`` configuration file, it must also be in SI units.
