"""Define useless commands; raise warning as influence the linac design."""

import logging

from core.commands.dummy_command import DummyCommand


class Error(DummyCommand):
    """A class that does nothing but raise an error."""

    def __init__(self, *args, **kwargs) -> None:
        """Raise an error."""
        logging.error(
            "The ERROR commands are not implemented in LightWin. As this "
            "commands will influence the design of the linac, you should set "
            "the design and comment this commands out."
        )
        return super().__init__(*args, **kwargs)


class ErrorBeamDyn(Error):
    """A class that does nothing but raise an error."""


class ErrorBeamStat(Error):
    """A class that does nothing but raise an error."""


class ErrorBendCPLDyn(Error):
    """A class that does nothing but raise an error."""


class ErrorBendCPLStat(Error):
    """A class that does nothing but raise an error."""


class ErrorBendNCPLDyn(Error):
    """A class that does nothing but raise an error."""


class ErrorBendNCPLStat(Error):
    """A class that does nothing but raise an error."""


class ErrorCavCPLDyn(Error):
    """A class that does nothing but raise an error."""


class ErrorCavCPLStat(Error):
    """A class that does nothing but raise an error."""


class ErrorCavNCPLDyn(Error):
    """A class that does nothing but raise an error."""


class ErrorCavNCPLStat(Error):
    """A class that does nothing but raise an error."""


class ErrorCavNCPLStatFile(Error):
    """A class that does nothing but raise an error."""


class ErrorDynFile(Error):
    """A class that does nothing but raise an error."""


class ErrorGaussianCutOff(Error):
    """A class that does nothing but raise an error."""


class ErrorQuadNCPLDyn(Error):
    """A class that does nothing but raise an error."""


class ErrorQuadNCPLStat(Error):
    """A class that does nothing but raise an error."""


class ErrorRFQCelNCPLStat(Error):
    """A class that does nothing but raise an error."""


class ErrorRFQCelNCPLDyn(Error):
    """A class that does nothing but raise an error."""


class ErrorStatFile(Error):
    """A class that does nothing but raise an error."""


class ErrorSetRatio(Error):
    """A class that does nothing but raise an error."""
