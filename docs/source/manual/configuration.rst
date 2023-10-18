Configuration
=============
Most of the configuration of LightWin is performed through a ``.ini`` configuration file, which should be given as argument to several objects initialisation.
The configuration file is treated with the help of the `configparser <https://docs.python.org/3/library/configparser.html>`_ module.
It is processed by the :func:`process_config` function, which checks it's validity and converts it to a dictionary.

.. note::
   In order to dynamically keep track of the options that are implemented, the *Allowed values* column of following tables contains a link to the variable storing the possible values, when relevant.


.. toctree::
   :maxdepth: 4

   configuration.files
   configuration.plots
   configuration.beam_calculator
   configuration.beam
   configuration.wtf
   configuration.evaluators
