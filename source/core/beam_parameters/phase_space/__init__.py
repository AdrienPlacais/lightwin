r"""Define objects to store the beam parameters of a phase space.

The stored elements are the emittances, the Twiss parameters, the transfer and
:math:`\sigma` matrices, the mismatch factor.
:class:`.InitialPhaseSpaceBeamParameters` holds the beam parameters at a single
position. In particular, at the start of the portion of the linac under study.
:class:`.PhaseSpaceBeamParameters` holds the beam parameters at several
positions. In particular, in the full portion of the linac under study at the
end of a calculation.
:class:`.IPhaseSpaceBeamParameters` holds the attributes and methods that are
common between those two objects.

"""
