# Installation
You will need Python 3.12 or higher.

## Mandatory packages
- `matplotlib`
- `numpy`
- `palettable`
- `pandas`
- `scipy`

## Mandatory third-party packages
`pymoo`:
for genetic optimisation algorithms.
Remove related imports in `fault.py` and `fault_scenario.py` to run LightWin without it.

## Optional packages
`cython` to speed up calculations; see instructions [here](https://adrienplacais.github.io/LightWin/html/manual/cython.html).
Check :ref:`cython`.
It can also be used to compile some `pymoo` functions to speed them up.
Just install `cython` prior to `pymoo`, and the compilation should be done automatically when installing `pymoo`.

`pytest` to ensure that everything is working as expected.

`cloudpickle` to pickle/unpickle some objects (see `util.pickling` documentation).

.. _cython: manual/cython.rst

## Packages for developers
- `sphinx_rtd_theme`
- `myst-parser`
- `nbsphinx`

## Installation of a package
If you manage your installation with `pip`:
`pip install package`

If you manage it with `conda`, do not use `pip` as it may break your installation!
The generic procedure is:
`conda install package`

As `pymoo` package is currently not on anaconda, create an conda environment and take your packages from `conda-forge`:
```
conda create -n <env-name> -c conda-forge python=3.12
conda activate <env_name>
conda install cython matplotlib numpy palettable pandas scipy pymoo pytest -c conda-forge
```
(may be necessary to install the different packages one at a time)
Precise `-c conda-forge` each time you want to update or install packages.

On Windows, you may want to run these commands from the Anaconda Prompt.

`pip` and `anaconda` are not compatible!
Never mix them!
Or create a dedicated environment.
If you use Spyder, check this out:
[https://www.youtube.com/watch?v=Ul79ihg41Rs](https://www.youtube.com/watch?v=Ul79ihg41Rs)

## TraceWin compatibility
To run TraceWin, add your computer/server paths in your `machine_config.toml`.
You can have a single `machine_config.toml` file for all your projects, or one per project.
The path to the `machine_config.toml` is defined in the `lightwin.toml` configuration, under the `tracewin` key, `machine_config_file` subkey.

An example is provided in the `data/example` folder.

TODO: Clearer explanations + example; maybe dedicated page in documentation, in the same section than Cython?


### Test
When everything is set up, navigate to the `lightwin` installation directory (where `data`, `source`, `tests` directories should be) and run `pytest`.
If TraceWin is not installed, run `pytest -m "not tracewin"`.

Note that as for now, `pytest` will raise errors if `pymoo` is not installed, and if the `cython` packages are not compiled.

TODO: Clearer explanations + example; maybe dedicated page in documentation, in the same section than Cython + TraceWin?


### FAQ
TODO
