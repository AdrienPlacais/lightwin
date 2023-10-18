``beam_calculator`` section (mandatory)
***************************************
.. toctree::
   :maxdepth: 4


.. csv-table::
   :file: configuration_entries/beam_calculator.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

If the desired :class:`.BeamCalculator` is :class:`.Envelope1D`:

.. csv-table::
   :file: configuration_entries/beam_calculator_envelope1d.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

If the desired :class:`.BeamCalculator` is :class:`.TraceWin`:

.. csv-table::
   :file: configuration_entries/beam_calculator_tracewin.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

Check TraceWin's documentation for the list of command line arguments.

.. todo::
   List of allowed tracewin arguments in doc

.. todo::
   There are doublons between doc in :mod:`.config.beam_calculator` and here. Maybe keep in module, but make the format better.

The ``[beam_calculator_post]`` follows the same format.
It is used to store a second :class:`.BeamCalculator`.
This is mainly useful for defining a more precise -- but more time-consuming -- beam dynamics tool, for example to check your compensation settings.

