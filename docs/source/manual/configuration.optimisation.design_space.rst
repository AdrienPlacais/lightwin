``optimisation.design_space`` section
*************************************
.. toctree::
   :maxdepth: 4

This section parametrizes how the design space will be set, i.e. what are the variables, their limits and initial values, and what are the constraints and their limits.
If you have any doubt, know that all these settings are passed down to :meth:`.DesignSpaceFactory.__init__` as ``design_space_kw``.

There are two ways to define the design space limits and initial values; the first is to let LightWin calculate it from the nominal settings of the linac.
This approach is easier to use for the first runs.

.. csv-table::
   :file: configuration_entries/design_space_no_file.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

When ``from_file`` is ``True``, you must provide a path to a ``.csv`` file containing, for every element, every variable, its initial value and limits.
If the problem is constrained, you must also provide a ``.csv`` with, for every element, the limits of every constraint.
This approach is more useful when you want to fine-tune the optimisation, as you can manually edit the ``.csv``, for example to take into account the specific multipacting barriers of a rogue cavity. 
To generate the ``.csv`` files with the proper format, look at ``examples/generate_design_space_files.py``.

.. csv-table::
   :file: configuration_entries/design_space_with_file.csv
   :widths: 30, 5, 50, 10, 5
   :header-rows: 1

Ultimately, these settings will be passed down to :class:`.DesignSpaceFactory`.
