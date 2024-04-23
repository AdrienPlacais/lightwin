# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.18] 2024-??-??

### Added
- You can forbid a cavity from being retuned (ex: a rebuncher which is here to rebunch, not to try funny beamy things). Just set `my_cavity.can_be_retuned = False`.

### Changed
- New typing features impose the use of Python 3.12.

### Fixed
- Colors in Evaluator plots are now reset between executions

## [0.6.17] 2024-04-19

### Added
- Switch between different phases at `.dat` save.

### Fixed
- With the `"sync_phase_amplitude"` design space, the synchronous phases were saved in the `.dat` and labelled as relative phase (no `SET_SYNC_PHASE`).

## [0.6.16] 2024-04-17

### Added
- New design space `"rel_phase_amplitude_with_constrained_sync_phase"`
- Pytest for basic compensation with all `BeamCalculator`
- Pytest for every `Design Space`

### Deprecated
- Some design space names are not to be used.
 - `"unconstrained"` -> `"abs_phase_amplitude"`
 - `"unconstrained_rel"` -> `"rel_phase_amplitude"`
 - `"constrained_sync_phase"` -> `"abs_phase_amplitude_with_constrained_sync_phase"`
 - `"sync_phase_as_variable"` -> `"sync_phase_amplitude"`

### Removed
- Support for `.ini` configuration files.
- `"phi_s_fit"` entry in configuration (use the proper design space config entry instead)

### Fixed
- Lattices and their indexes correctly set.
- Synchronous phases correctly calculated and updated; can be used as a variable again.

<!-- ## [0.0.0] 1312-01-01 -->
<!---->
<!-- ### Added -->
<!---->
<!-- ### Changed -->
<!---->
<!-- ### Deprecated -->
<!---->
<!-- ### Removed -->
<!---->
<!-- ### Fixed -->
<!---->
<!-- ### Security -->
