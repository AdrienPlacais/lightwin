# Installation
You will need Python 3.
Tested with Python 3.9, 3.10, 3.11.

## Mandatory packages
- `matplotlib`
- `numpy`
- `pandas`
- `scipy`

## Mandatory third-party packages
`palettable`:
for sweet colors.
Remove all lines related to `palettable`, `colorbrewer` and `cycler` in `visualisation/plot.py` to run LightWin without it.
`pymoo`:
for genetic optimisation algorithms.
Remove related imports in `fault.py` and `fault_scenario.py` to run LightWin without it.

## Optional packages
`cython`:
To speed up calculations, the `transfer_matrices_p.py` file has a second version called `transfer_matrices_c.pyx`.
It can be compiled into C to speed up calculations.
You will need to compile it yourself at the installation, and every time you update your Python distribution.
Instructions in `util/setup.py`.
It can also be used to compile some `pymoo` functions to speed them up.
Just install `cython` prior to `pymoo`, and the compilation should be done automatically when installing `pymoo`.

`spyder`:
A sweet IDE.

## Installation of a package
If you manage your installation with `pip`:
`pip install package`

If you manage it with `conda`, do not use `pip` as it may break your installation!
Instead:
`conda install package`

`pymoo` package is currently not on anaconda.
Create an conda environment and take your packages from `conda-forge`:
```
conda create -n <env_name> -c conda-forge python=3.10
conda activate <env_name>
conda install matplotlib numpy pandas scipy palettable pymoo -c conda_forge
```
(may be necessary to install the different packages one at a time)

`pip`, `anaconda` and `conda-forge` packages are not compatible!
Never mix them!
Or create a dedicated environment.
If you use Spyder, check this out:
[https://www.youtube.com/watch?v=Ul79ihg41Rs](https://www.youtube.com/watch?v=Ul79ihg41Rs)

