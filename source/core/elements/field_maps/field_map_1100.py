"""Define a field map with 1D rf electro-magnetic field."""

from core.elements.field_maps.field_map import FieldMap


class FieldMap1100(FieldMap):
    """1D rf electro-magnetic field.

    Just inherit from the classic :class:`.FieldMap`; we override the
    ``to_line`` to also update ``k_b`` (keep ``k_e == k_b``).

    """

    def __init__(self, *args, **kwargs) -> None:
        """Init the same object as :class:`.FieldMap100`."""
        return super().__init__(*args, **kwargs)

    def to_line(
        self,
        which_phase: str = "phi_0_rel",
        *args,
        inplace: bool = False,
        **kwargs,
    ) -> list[str]:
        r"""Convert the object back into a line in the ``.dat`` file.

        Parameters
        ----------
        which_phase : {'phi_0_abs', 'phi_0_rel', 'phi_s', 'as_in_settings',
                \ 'as_in_original_dat'}
            Which phase should be putted in the output ``.dat``.
        inplace : bool, optional
            To modify or not the :attr:`.Element` inplace. If False, we return
            a modified copy. The default is False.

        Returns
        -------
        list[str]
            The line in the ``.dat``, with updated amplitude and phase from
            current object.

        """
        line = super().to_line(
            which_phase=which_phase, *args, inplace=inplace, **kwargs
        )
        shift = 0
        if self._personalized_name:
            shift += 1
        line[5 + shift] = str(self.cavity_settings.k_e)
        return line
