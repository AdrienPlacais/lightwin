"""This module holds :class:`Quad`."""

from core.elements.element import Element


class Quad(Element):
    """A partially defined quadrupole."""

    base_name = "QP"
    n_attributes = range(3, 10)

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Check number of attributes, set gradient."""
        super().__init__(line, dat_idx, name)
        self.grad = float(line[2])
        self.reinsert_optional_commands_in_line()
