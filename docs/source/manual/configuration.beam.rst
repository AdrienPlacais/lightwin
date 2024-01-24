``beam`` section (mandatory)
****************************
.. toctree::
   :maxdepth: 4

Here we define the main properties of the beam at the entrance of the linac.
Note that with :class:`.TraceWin`, most of these properties are defined within it's own ``.ini`` file.
The units must be consistent with LightWin's system of units, see also :ref:`units-label`.

.. csv-table::
   :file: configuration_entries/beam.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

Format for the ``sigma`` entry:

.. code-block:: toml

   sigma = [
      [ 1e-6, -2e-7,  0e+0, 0e+0,  0e+0, 0e+0],
      [-2e-7,  8e-7,  0e+0, 0e+0,  0e+0, 0e+0],
      [ 0e+0,  0e+0, -2e-7, 8e-7,  0e+0, 0e+0],
      [ 0e+0,  0e+0, -2e-7, 8e-7,  0e+0, 0e+0],
      [ 0e+0,  0e+0,  0e+0, 0e+0, -2e-7, 8e-7],
      [ 0e+0,  0e+0,  0e+0, 0e+0, -2e-7, 8e-7]
    ]

