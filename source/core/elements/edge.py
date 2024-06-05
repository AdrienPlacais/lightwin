"""Define :class:`Edge`. It does nothing.

.. todo::
    Check behavior w.r.t. LATTICE.

"""

import logging

from core.elements.element import Element


class Edge(Element):
    """A dummy object."""

    base_name = "EDG"
    increment_lattice_idx = False
    is_implemented = False

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx, name)
        self.length_m = 0.0
        logging.warning(
            "Documentation does not mention that EDGE element should be "
            "ignored by LATTICE. So why did I set increment_lattice_idx to "
            "False?"
        )
        self.reinsert_optional_commands_in_line()
