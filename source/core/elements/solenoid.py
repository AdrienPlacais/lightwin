"""Define :class:`Solenoid`."""

from core.elements.element import Element


class Solenoid(Element):
    """A partially defined solenoid."""

    base_name = "SOL"
    n_attributes = 3

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Check number of attributes."""
        super().__init__(line, dat_idx, name)
        self.reinsert_optional_commands_in_line()
