"""Set a dummy class that raises an error message."""

import logging

from core.commands.dummy_command import DummyCommand


class SetSyncPhase(DummyCommand):
    """A class that does nothing but raise an error."""

    def __init__(self, *args, **kwargs) -> None:
        """Raise an error."""
        logging.error(
            "The SET_SYNC_PHASE command is not implemented yet in LightWin. "
            "You shall use the real phase instead (absolute or relative). "
            "Integration of this command is part of the development plan."
        )
        return super().__init__(*args, **kwargs)
