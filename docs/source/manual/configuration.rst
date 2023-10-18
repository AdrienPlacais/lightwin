Configuration
=============
Most of the configuration of LightWin is performed through a ``.ini`` configuration file, which should be given as argument to several objects initialisation.
The configuration file is treated with the help of the `configparser <https://docs.python.org/3/library/configparser.html>`_ module.
It is processed by the :func:`process_config` function, which checks it's validity and converts it to a dictionary.

The name of every section is not important, as long as every section is correctly passed to :func:`process_config`.
It is however recommended to use explicit names.

.. rubric:: Example for the ``beam`` section

.. code-block:: ini

   [beam.proton]     ; this could be [bonjoure] or anything
   linac = my_ads
   e_rest_mev = 938.27203
   q_adim = 1.
   e_mev = 20.
   f_bunch_mhz = 350.
   i_milli_a = 25.
   sigma_zdelta =
      1e-6, -2e-7
      -2e-7, 8e-7

   [beam.proton_no_space_charge]   ; or [sdaflghsh] but it would be less explicit in my opinion
   linac = my_ads
   e_rest_mev = 938.27203
   q_adim = 1.
   e_mev = 20.
   f_bunch_mhz = 350.
   i_milli_a = 0.
   sigma_zdelta =
      1e-6, -2e-7
      -2e-7, 8e-7


.. note::
   In order to dynamically keep track of the options that are implemented, the *Allowed values* column of following tables contains a link to the variable storing the possible values, when relevant.


.. toctree::
   :maxdepth: 4

   configuration.files
   configuration.beam
   configuration.beam_calculator
   configuration.plots
   configuration.wtf
   configuration.evaluators
