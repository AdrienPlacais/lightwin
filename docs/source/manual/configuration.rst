Configuration
=============
Most of the configuration of LightWin is performed through a ``.ini`` configuration file, which should be given as argument to several objects initialisation.
The configuration file is treated with the help of the `configparser <https://docs.python.org/3/library/configparser.html>`_ module.
It is processed by the :func:`process_config` function, which checks it's validity and converts it to a dictionary.

.. note::
   In order to dynamically keep track of the options that are implemented, the *Allowed values* column of following tables contains a link to the variable storing the possible values, when relevant.

``[files]`` section (mandatory)
*******************************
This section is mandatory to initialize an :class:`.Accelerator` through the :func:`.accelerator_factory` function.
It must contain the key ``dat_file``, which is the path to the linac structure file (same format as TraceWin).

.. csv-table::
   :file: configuration_entries/files.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

See also: :mod:`config.files`

``[plots]`` section (mandatory)
*******************************

.. csv-table::
   :file: configuration_entries/plots.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

.. warning::
   Plot of transfer matrix currently not working.


``[beam_calculator]`` section (mandatory)
*****************************************

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

``[beam]`` section (mandatory)
******************************
Here we define the main properties of the beam at the entrance of the linac.
Note that with :class:`.TraceWin`, most of these properties are defined within it's own ``.ini`` file.

.. csv-table::
   :file: configuration_entries/beam.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

Format for the ``sigma_zdelta`` entry:

.. code-block:: ini

   sigma_zdelta =    ; Line skip is required
      1e-6, -2e-7
      -2e-7, 8e-7

``[wtf]`` section
*****************
``wtf`` stands for *what to fit*.
This section parametrizes the failed cavities, as well as how they are fixed.

.. csv-table::
   :file: configuration_entries/wtf.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

Each ``strategy`` entry requires specific additional arguments.
As an example, with the ``k out of n`` method, you need to give LightWin ``k``, the number of compensating cavities per failed cavity.
The specific documentation can be found in :mod:`failures.strategy`.

You can type the index of failed cavities on several lines if you want to study several fault scenarios at once.

.. rubric:: Example

.. code-block:: ini

   ; Indexes are cavity indexes
   idx = cavity
   failed =
      0, 1,       ; First simulation first cryomodule is down
      0,          ; Second simulation only first cavity is down
      1, 45       ; Third simulation second and 46th cavity are down


.. warning::
   ``phi_s fit`` key will be removed in future updates. For now, keep it consistent with what you ask in ``objective_preset`` and ``design_space_preset``.


``[evaluators]`` section
************************
This section is used to defined :class:`.Evaluator` objects.
They are used to evaluate the absolute or relative *quality* of compensation settings.

.. csv-table::
   :file: configuration_entries/evaluator.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

