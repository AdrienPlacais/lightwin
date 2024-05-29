Cython (mandatory)
------------------

.. todo::
   Revise integration so that a missing Cython does not lead to import errors.

To speed things up, Envelope1D has a Cython version that must be compiled.
You can do so by going to `lightwin/source` and run:

.. code-block:: bash

   python util/setup.py build_ext --inplace

A `.so` (Linux) or `.pyd` (Windows) file will be created.
If it is not done automatically, move the created file in ``beam_calculation/envelope_1d/``:

* Unix: ``build/lib.linux-XXX-cpython=3XX/beam_calculation/envelope_1d/transfer_matrices_c.cpython-3XX-XXXX-linux-gnu.so``
* Windows: ``build/lib.win-XXXX-cpython-3XX/beam_calculation/envelope_1d/transfer_matrices_c.cp3XX-win_XXXX.pyd``

Note that if you use an interpreter such as Spyder, you should restart the kernel to correctly load the file.
If you change your Python version or use another machine, you must recompile.

.. todo::
   Cython pytests 

On Windows
^^^^^^^^^^

If you have a `Microsoft Visual C++ 14.0 or greater is required` error,
go to `visual studio website`_ and download Build Tools:

#. Download and execute `vs_BuildTools.exe`.
#. Check "C++ Development Desktop".
#. Install.

.. _visual studio website: https://visualstudio.microsoft.com/visual-cpp-build-tools/
