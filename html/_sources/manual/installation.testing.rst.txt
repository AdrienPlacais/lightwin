Testing
-------

Pytest
^^^^^^

To test your installation, go to the base directory of LightWin (where the `pyproject.toml` file is located).
From here, run: `pytest`.
If TraceWin is not installed, run: `pytest -m not tracewin`.
You can combine the marks defined in `pyproject.toml`, for example `pytest -m (smoke and not slow)` for fast smoke tests.

Frequent errors
^^^^^^^^^^^^^^^

* `E   ModuleNotFoundError: No module named 'beam_calculation'`.
 * Your `PYTHONPATH` is not properly set.
* `xfailed` errors.
 * eXpected to fail errors are "normal" and should only worry developers.
