``beam`` section (mandatory)
****************************
.. toctree::
   :maxdepth: 4

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

