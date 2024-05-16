Cython
======

To speed things up, Envelope1D has a Cython version that must be compiled.
You can do so by going to `lightwin/source` and run:

```
python util/setup.py build_ext --inplace
```

A `.so` (Linux) or `.pyd` (Windows) file will be created.
If it is not done automatically, move the created file in ``beam_calculation/envelope_1d/``:
Unix: ``build/lib.linux-XXX-cpython=3XX/beam_calculation/envelope_1d/transfer_matrices_c.cpython-3XX-XXXX-linux-gnu.so``
Windows: ``build/lib.win-XXXX-cpython-3XX/beam_calculation/envelope_1d/transfer_matrices_c.cp3XX-win_XXXX.pyd``

Note that if you use an interpreter such as Spyder, you should restart the kernel to correctly load the file.
If you change your Python version or use another machine, you must recompile.

.. todo::
   Cython pytests 

Windows
-------
For Windows users, you may have to go  https://visualstudio.microsoft.com/visual-cpp-build-tools/ and download Build Tools:
   1. Download and execute `vs_BuildTools.exe`.
   2. Check "C++ Development Desktop".
   3. Install.
