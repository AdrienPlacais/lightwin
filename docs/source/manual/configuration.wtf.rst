``wtf`` section
***************
.. toctree::
   :maxdepth: 4

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

.. code-block:: toml

   # Indexes are cavity indexes
   idx = cavity
   failed = [
      [0, 1],       # First simulation first cryomodule is down
      [0],          # Second simulation only first cavity is down
      [1, 45]       # Third simulation second and 46th cavity are down
   ]
