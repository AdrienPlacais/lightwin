"""Define :class:`Aperture`. It does nothing."""

from core.elements.element import Element


class Aperture(Element):
    """A dummy object."""

    base_name = "AP"
    increment_lattice_idx = False
    is_implemented = False

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Force an element with null-length."""
        super().__init__(line, dat_idx, name)
        self.length_m = 0.0
        self.reinsert_optional_commands_in_line()
