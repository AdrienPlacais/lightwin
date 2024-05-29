TraceWin (facultative)
----------------------

Pre-requisite: TraceWin must be installed on your computer or server with a working license.

The `machine_configuration.toml` file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You must tell LightWin where TraceWin is installed.
This is done thanks to a `machine_configuration.toml` file, which looks like:

.. code-block:: toml

   [lpsc5057x]
   noX11_full = "/usr/local/bin/TraceWin/./TraceWin_noX11"
   noX11_minimal = "/home/placais/TraceWin/exe/./tracelx64"
   no_run = ""
   X11_full = "/usr/bin/local/bin/TraceWin/./TraceWin"

   [LPSC5011W]
   X11_full = "D:/tw/TraceWin.exe"
   noX11_full = "D:/tw/TraceWin.exe"

   [a_name_to_override_default_machine_name]
   X11_full = "D:/tw/TraceWin_old.exe"
   noX11_full = "D:/tw/TraceWin_old.exe"


Between the brackets, type the name of your machine.
Execute following lines of Python code if you do not know the name of your machine:

.. code-block:: python

   import socket
   machine_name = socket.gethostname()
   print(f"Entry in the machine_configuration.toml file should be:\n[{machine_name}]")

Link with the `lightwin.toml` main configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [my_tracewin_configuration]
   # Can be relative w.r.t `lightwin.toml`, or absolute:
   machine_config_file = "/path/to/machine_configuration.toml"
   # Corresponding path must be defined in the machne_configuration.toml
   simulation_type = "X11_full"                                
   # Facultative! Will override your actual machine name if provided:
   machine_name = "a_name_to_override_default_machine_name"
   # note that additional entries will be mandatory

See `dedicated documentation`_ for `lightwin.toml`.

.. _dedicated documentation: https://adrienplacais.github.io/LightWin/html/configuration/beam_calculator.html
