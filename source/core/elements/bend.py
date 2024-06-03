"""Define an horizontal :class:`Bend`.

The transfer matrix parameters, and ``k_x`` in particular, are defined only for
a field gradient index inferior to unity in TraceWin documentation. For the
cases where ``n > 1``, see `this topic`_.

.. _this topic: https://dacm-codes.fr/forum/viewtopic.php?f=3&t=740&p=1633&hil\
it=BEND#p1633

"""

import math

from core.elements.element import Element


class Bend(Element):
    """Hold the strict minimum for BEND element.

    In TraceWin documentation, transfer matrix is in :math:`mm` and
    :math:`deg`. We use here :math:`m` and :math:`rad`.

    """

    base_name = "DIP"
    n_attributes = 5

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Precompute the parameters used to compute transfer matrix."""
        super().__init__(line, dat_idx, name)

        self.bend_angle = float(math.radians(float(line[1])))
        self.curvature_radius = float(line[2]) * 1e-3
        self.field_grad_index = float(line[3])
        self.length_m = self.curvature_radius * abs(self.bend_angle)

        self._h_squared: float
        self._k_x: float
        self.reinsert_optional_commands_in_line()

    @property
    def h_parameter(self) -> float:
        """Compute the parameter ``h``."""
        return math.copysign(1.0 / self.curvature_radius, self.bend_angle)

    @property
    def h_squared(self) -> float:
        """Compute the ``h**2`` parameter."""
        if not hasattr(self, "_h_squared"):
            self._h_squared = self.h_parameter**2
        return self._h_squared

    @property
    def k_x(self) -> float:
        """Compute the ``k_x`` parameter."""
        if not hasattr(self, "_k_x"):
            _tmp = 1.0 - self.field_grad_index
            if self.field_grad_index > 1.0:
                _tmp *= -1.0
            self._k_x = math.sqrt(_tmp * self.h_squared)
        return self._k_x
