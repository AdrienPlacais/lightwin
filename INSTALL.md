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
`cython` for `beam_calculation/envelope_1d/`:
To speed up calculations, the `transfer_matrices_p.py` file has a second version called `transfer_matrices_c.pyx`.
It can be compiled into C to speed up calculations.
You will need to compile it yourself at the installation, and every time you update your Python distribution.
Instructions in `util/setup.py`.
It can also be used to compile some `pymoo` functions to speed them up.
Just install `cython` prior to `pymoo`, and the compilation should be done automatically when installing `pymoo`.

`pytest` to ensure that everything is working as expected.

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
To run TraceWin, modify the paths in `source/config/beam_calculator.py` so that it matches with your installation.


### Test
When everything is set up, navigate to the `lightwin` dir and run `pytest`.
If TraceWin is not installed, run `pytest -m "not tracewin"`.

Note that as for now, `pytest` will raise errors if `pymoo` is not installed, and if the `cython` packages are not compiled.
