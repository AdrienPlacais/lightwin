# LightWin
LightWin is a tool to automatically find compensation settings for a various set of cavity failures.

## General behavior
The majority of the parameters are handled through a `lightwin_configuration.ini` file.
The `files` entry sets the `.dat` file, which is the same format as `TraceWin`.
It sets the `Accelerator` object, which holds a `ListOfElements`.
The latter is a `list` containing all the `_Element`s present in the `.dat.` file.

The `beam_calculator` entry sets the tool that will compute the propagation of the beam through the `ListOfElements`.
All the relatable data, such as the evolution of the energy with the position, are stored in a `SimulationOutput`.
Its `BeamParameters` and `ParticleInitialState` are set by the `beam` entry.

The `wtf` (what to fit) entry sets which cavities fail and how they are compensated for.
For each line of the `failed` entry of `wtf`, a `FaultScenario` is created.
A `FaultScenario` is a list of `Fault`s.
Each `Fault` has its own `ListOfElements`, which encompasses only the `_Element`s of the compensation zone.

Then, an `OptimisationAlgorithm` calls the `BeamCalculator` to try several `SetOfCavitySettings`.
Once the `SimulationOutput` are satisfactory, the `Fault` is compensated and `LightWin` moves on to the next one.

Once all `FaultScenario`s are dealt with, a new simulation (more precise) can be runned if a second `BeamCalculator` is provided.

The keys from `plots` determine which `plt.Figure`s are produced.
The `evaluators` entry sets a `ListOfSimulationOutputEvaluators` (which is a `list` of `SimulationOutputEvaluator`s), which will run tests to evaluate if the new settins are acceptable.

## How to run
TODO

## Example
TODO : file from TW?

## Documentation
To generate an interactive documentation, you must ensure that you have the packages `sphinx`, `sphinx_rtd_theme` and `myst-parser` (see `INSTALL.md` for packages installation).
Go to the `LightWin/docs` folder and run: `make html`.
On Windows, `make.bat html` should do the trick (untested).
