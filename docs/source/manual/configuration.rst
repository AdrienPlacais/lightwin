Configuration
=============
.. toctree::
   :maxdepth: 4
   :hidden:

   configuration.files
   configuration.beam
   configuration.beam_calculator
   configuration.plots
   configuration.wtf
   configuration.optimisation.design_space
   configuration.evaluators

Most of the configuration of LightWin is performed through a ``.toml`` configuration file, which should be given as argument to several objects initialisation.
The configuration file is treated with the help of the `tomllib <https://docs.python.org/3/library/tomllib.html>`_ module.
It is processed by the :func:`process_config` function, which checks it's validity and converts it to a dictionary.

.. note::
   Configuration was previously set up thanks to a ``.ini`` file. This is now deprecated.

The name of every section is not important, as long as every section is correctly passed to :func:`process_config`.
It is however recommended to use explicit names.

.. rubric:: Example for the ``beam`` section

.. code-block:: toml

   [beam_proton]     # this could be [bonjoure] or anything
   e_rest_mev = 938.27203
   q_adim = 1.0
   e_mev = 20.0
   f_bunch_mhz = 350.0
   i_milli_a = 25.
   sigma = [
      [ 1e-6, -2e-7, 0.0,   0.0,  0.0,   0.0],
      [-2e-7,  8e-7, 0.0,   0.0,  0.0,   0.0],
      [ 0.0,   0.0,  1e-6, -2e-7, 0.0,   0.0],
      [ 0.0,   0.0, -2e-7,  8e-7, 0.0,   0.0],
      [ 0.0,   0.0,  0.0,   0.0,  1e-6, -2e-7],
      [ 0.0,   0.0,  0.0,   0.0, -2e-7,  8e-7],
   ]

   [beam_proton_no_space_charge]   # or [sdaflghsh] but it would be less explicit in my humble opinion
   linac = my_ads
   e_rest_mev = 938.27203
   q_adim = 1.0
   e_mev = 20.0
   f_bunch_mhz = 350.0
   i_milli_a = 0.0
   sigma = [
      [ 1e-6, -2e-7, 0.0,   0.0,  0.0,   0.0],
      [-2e-7,  8e-7, 0.0,   0.0,  0.0,   0.0],
      [ 0.0,   0.0,  1e-6, -2e-7, 0.0,   0.0],
      [ 0.0,   0.0, -2e-7,  8e-7, 0.0,   0.0],
      [ 0.0,   0.0,  0.0,   0.0,  1e-6, -2e-7],
      [ 0.0,   0.0,  0.0,   0.0, -2e-7,  8e-7],
   ]


.. note::
   In order to dynamically keep track of the options that are implemented, the *Allowed values* column of following tables contains a link to the variable storing the possible values, when relevant.

.. include:: configuration.files.rst
.. include:: configuration.beam.rst
.. include:: configuration.beam_calculator.rst
.. include:: configuration.plots.rst
.. include:: configuration.wtf.rst
.. include:: configuration.optimisation.design_space.rst
.. include:: configuration.evaluators.rst

