# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.16] 2024-04-17

### Added
- New design space `"rel_phase_amplitude_with_constrained_sync_phase"`
- Pytest for basic compensation with all `BeamCalculator`
- Pytest for every `Design Space`

### Changed

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

### Security

## [0.0.0] 1312-01-01

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security
