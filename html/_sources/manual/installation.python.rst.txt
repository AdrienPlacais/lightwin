Python (mandatory)
------------------
Note that you will need Python 3.12 or higher.

Packages
^^^^^^^^

Mandatory
"""""""""

* `matplotlib`
* `numpy`
* `palettable`
* `pandas`
* `scipy`

.. todo::
   `tkinter` also necessary? Maybe dependency of another package? Check this out.

Mandatory third-party packages
""""""""""""""""""""""""""""""

* `pymoo` for genetic optimization algorithms. Remove related imports in `fault.py` and `fault_scenario.py` to run LightWin without it.

Optional
""""""""

* `cython` to speed up calculations. Check `cython integration documentation`_.
 * It can also be used to compile some `pymoo` functions to speed them up. Just install `cython` prior to `pymoo`, and the compilation should be done automatically when installing `pymoo`.
* `pytest` to ensure that everything is working as expected.
* `cloudpickle` to pickle/unpickle some objects (see `util.pickling` documentation).

.. _cython integration documentation: https://adrienplacais.github.io/LightWin/html/manual/cython.html

For developers
""""""""""""""

Those packages are necessary to compile the documentation.

* `sphinx_rtd_theme`
* `myst-parser`
* `nbsphinx`


Reminders
^^^^^^^^^

Installation of a package
"""""""""""""""""""""""""

If you manage your installation with `pip`: `pip install package`

If you manage it with `conda`, do not use `pip` as it may break your installation!
The generic procedure is: `conda install package`

As `pymoo` package is currently not on anaconda, create an conda environment and take your packages from `conda-forge`:

.. code-block:: bash

   conda create -n <env-name> -c conda-forge python=3.12
   conda activate <env_name>
   conda install cython matplotlib numpy palettable pandas scipy pymoo pytest -c conda-forge

Precise `-c conda-forge` each time you want to update or install packages.

On Windows, you may want to run these commands from the Anaconda Prompt.

.. warning::
   `pip` and `anaconda` are not compatible!
   Never mix them!
   Or create a dedicated environment.
   If you use Spyder, this `video`_ can provide you with more information.

.. _video: https://www.youtube.com/watch?v=Ul79ihg41Rs

Set up the Python path
""""""""""""""""""""""

You must add the `/path/to/lightwin/source` path to your `PYTHONPATH`.

.. todo::
   More detailed instructions? As for now, check "PYTHONPATH" or "ModuleNotFoundError" on your favorite search engine.
