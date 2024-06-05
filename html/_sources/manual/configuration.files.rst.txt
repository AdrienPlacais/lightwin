``files`` section (mandatory)
*****************************
.. toctree::
   :maxdepth: 4

This section is mandatory to initialize an :class:`.Accelerator` through the :func:`.accelerator_factory` function.
It must contain the key ``dat_file``, which is the path to the linac structure file (same format as TraceWin).

.. csv-table::
   :file: configuration_entries/files.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

See also: :mod:`config.files` and the `TraceWin compatibility note`_.

.. `TraceWin compatibility note`: https://adrienplacais.github.io/LightWin/html/manual/usage.html#compatibility-with-tracewin-dat-files

