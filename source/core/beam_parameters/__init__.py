r"""Store and compute beam parameters in all the phase spaces.

In particular, this subpackage handles the calculation of the Twiss parameters,
the emittances, the mismatch factors, the :math:`\sigma` matrix.

.. todo::
    Creation of phase spaces directly in the __init__ methods.

.. todo::
    Remove tracewin command from BeamParameters

.. todo::
    Creation of various beam parameters too scattered. Some work is done in
    SinglePhaseSpaceBeamParameters (and Initial), some in BeamParameters (and
    Initial) and some in factory...
    Look at ``@classmethod``, ``@singledispatchmethod``
    https://realpython.com/python-multiple-constructors/

"""
