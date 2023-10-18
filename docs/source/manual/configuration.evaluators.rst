``evaluators`` section
**********************
.. toctree::
   :maxdepth: 4

This section is used to defined :class:`.FaultScenarioSimulationOutputEvaluator` objects.
They are used to evaluate the absolute or relative *quality* of compensation settings.

.. csv-table::
   :file: configuration_entries/evaluator.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

.. note::
   There is also a :class:`.SimulationOutputEvaluator` object that is used internally by LightWin to determine if a :class:`.Fault` was correctly fixed.
