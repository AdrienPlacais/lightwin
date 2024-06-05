"""Define :class:`Drift`."""

from core.elements.element import Element


class Drift(Element):
    """A simple drift tube."""

    base_name = "DR"
    n_attributes = (2, 3, 5)

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Check that number of attributes is valid."""
        super().__init__(line, dat_idx, name)
        self.reinsert_optional_commands_in_line()
