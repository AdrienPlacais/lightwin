"""Define a ``list`` of :class:`.Element`, with some additional methods.

Two objects can have a :class:`ListOfElements` as attribute:
    - :class:`.Accelerator`: holds all the :class:`.Element` of the linac.
    - :class:`.Fault`: it holds only a fraction of the linac
      :class:`.Element`. Beam will be propagated a huge number of times
      during optimisation process, so we recompute only the strict necessary.

.. todo::
    Delete ``dat_content``, which does the same thing as ``elts_n_cmds`` but
    less good

"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self, overload

import numpy as np

from core.beam_parameters.initial_beam_parameters import InitialBeamParameters
from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap
from core.list_of_elements.helper import (
    first,
    group_elements_by_lattice,
    group_elements_by_section,
    group_elements_by_section_and_lattice,
)
from core.particle import ParticleInitialState
from tracewin_utils.dat_files import export_dat_filecontent
from tracewin_utils.interface import list_of_elements_to_command
from util.helper import recursive_getter, recursive_items
from util.pickling import MyPickler

element_id = int | str
elements_id = Sequence[int] | Sequence[str]
nested_elements_id = Sequence[Sequence[int]] | Sequence[Sequence[str]]


class ListOfElements(list):
    """Class holding the elements of a fraction or of the whole linac."""

    def __init__(
        self,
        elts: list[Element],
        input_particle: ParticleInitialState,
        input_beam: InitialBeamParameters,
        tm_cumul_in: np.ndarray,
        files: dict[str, Path | str | list[list[str]]],
        first_init: bool = True,
    ) -> None:
        """Create the object, encompassing all the linac or only a fraction.

        The first case is when you initialize an Accelerator and compute the
        baseline energy, phase, etc values.
        The second case is when you only recompute a fraction of the linac,
        which is part of the optimisation process.

        Parameters
        ----------
        elts : list[Element]
            List containing the element objects.
        input_particle : ParticleInitialState
            An object to hold initial energy and phase of the particle at the
            entry of the first element/
        input_beam : InitialBeamParameters
            An object to hold emittances, Twiss, sigma beam matrix, etc at the
            entry of the first element.
        first_init : bool, optional
            To indicate if this a full linac or only a portion (fit process).
            The default is True.
        files : dict[str, str | list[list[str]] | Path]
            A dictionary to hold information on the source and output
            files/folders of the object. The keys are:
                - ``dat_file``: absolute path to the ``.dat`` file
                - ``elts_n_cmds``: list of objects representing dat content
                - ``accelerator_path``: where calculation results for each
                :class:`.BeamCalculator` will be stored.
                - ``dat_content``: list of list of str, holding content of the
                ``.dat``.

        """
        self.input_particle = input_particle
        self.input_beam = input_beam
        self.files = files
        assert tm_cumul_in.shape == (6, 6)
        self.tm_cumul_in = tm_cumul_in

        super().__init__(elts)
        self.by_section_and_lattice: list[list[list[Element]]] | None = None
        self.by_lattice: list[list[Element]]

        if first_init:
            self._first_init()

        self._l_cav: list[FieldMap] = list(
            filter(lambda cav: isinstance(cav, FieldMap), self)
        )
        logging.info(
            "Successfully created a ListOfElements with "
            f"{self.w_kin_in = } MeV and {self.phi_abs_in = } rad."
        )

    @property
    def w_kin_in(self):
        """Get kinetic energy at entry of first element of self."""
        return self.input_particle.w_kin

    @property
    def phi_abs_in(self):
        """Get absolute phase at entry of first element of self."""
        return self.input_particle.phi_abs

    @property
    def l_cav(self) -> list[FieldMap]:
        """Easy access to the list of cavities."""
        return self._l_cav

    @property
    def tunable_cavities(self) -> list[FieldMap]:
        """All the elements that can be used for compensation.

        For now, only :class:`.FieldMap`. But in the future... Who knows?

        """
        return [cavity for cavity in self.l_cav if cavity.can_be_retuned]

    @property
    def tracewin_command(self) -> list[str]:
        """Create the command to give proper initial parameters to TraceWin."""
        dat_file = self.files["dat_file"]
        assert isinstance(dat_file, Path)
        _tracewin_command = [
            command_bit
            for command in [
                list_of_elements_to_command(dat_file),
                self.input_particle.tracewin_command,
                self.input_beam.tracewin_command,
            ]
            for command_bit in command
        ]

        return _tracewin_command

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self)) or key in recursive_items(
            vars(self[0])
        )

    def get(
        self,
        *keys: str,
        to_numpy: bool = True,
        remove_first: bool = False,
        **kwargs: bool | str | Element | None,
    ) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        This method also looks into the first :class:`.Element` of self. If the
        desired ``key`` is in this :class:`.Element`, we recursively get ``key``
        from every :class:`.Element` and concatenate the output.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        remove_first : bool, optional
            If you want to remove the first item of every :class:`.Element`
            ``key``.
            It the element is the first of the list, we do not remove its first
            item.  It is useful when the last item of an element is the same as
            the first item of the next element. For example, ``z_abs``. The
            default is False.
        **kwargs : bool | str | Element | None
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val: dict[str, Any] = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            # Specific case: key is in Element
            if self[0].has(key):
                for elt in self:
                    data = elt.get(key, to_numpy=False, **kwargs)

                    if remove_first and elt is not self[0]:
                        data = data[1:]
                    if isinstance(data, list):
                        val[key] += data
                        continue
                    val[key].append(data)
            else:
                val[key] = recursive_getter(key, vars(self), **kwargs)

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) for val in out]

        if len(keys) == 1:
            return out[0]
        return tuple(out)

    def _first_init(self) -> None:
        """Set structure, elements name, some indexes."""
        by_section = group_elements_by_section(self)
        self.by_lattice = group_elements_by_lattice(self)
        self.by_section_and_lattice = group_elements_by_section_and_lattice(
            by_section
        )
        self._set_element_indexes()

    def _set_element_indexes(self) -> None:
        """Set the element index."""
        elts_with_a_number = list(
            filter(lambda elt: elt.increment_elt_idx, self)
        )

        for i, elt in enumerate(elts_with_a_number):
            elt.idx["elt_idx"] = i

    def force_reference_phases_to(self, new_reference_phase: str) -> None:
        """Change the reference phase of the cavities in ``self``.

        This method is called by the :class:`.BeamCalculator`. It is used after
        the first propagation of the beam in the full :class:`ListOfElements`,
        to force every :class:`.CavitySettings` to use the reference phase
        specified by the ``beam_calculator`` entry of the ``.toml``.

        """
        for cavity in self.l_cav:
            settings = cavity.cavity_settings
            if settings.reference == new_reference_phase:
                continue
            settings.reference = new_reference_phase

    def store_settings_in_dat(
        self,
        dat_file: Path,
        which_phase: Literal["phi_0_abs", "phi_0_rel", "phi_s"] = "phi_0_abs",
        save: bool = True,
    ) -> None:
        r"""Update the ``dat`` file, save it if asked.

        Parameters
        ----------
        dat_file : Path
            Where the output ``.dat`` should be saved.
        which_phase : {'phi_0_abs', 'phi_0_rel', 'phi_s', 'as_in_settings',
                \ 'as_in_original_dat'}
            Which phase should be putted in the output ``.dat``.
        save : bool, optional
            If the output file should be created. The default is True.

        Important notice
        ----------------
        The phases of the cavities are rephased if the first :class:`.Element`
        in self is not the first of the linac. This way, the beam enters each
        cavity with the intended phase in :class:`.TraceWin`.

        """
        if which_phase == "phi_s":
            raise IOError("Output with SET_SYNC_PHASE not implemented yet.")
        if which_phase in ("as_in_settings", "as_in_original_dat"):
            raise NotImplementedError
        self.files["dat_file"] = dat_file
        dat_content = [
            instruction.to_line(which_phase=which_phase, inplace=False)
            for instruction in self.files["elts_n_cmds"]
        ]
        if save:
            export_dat_filecontent(dat_content, dat_file)

    @overload
    def take(self, ids: int, id_nature: Literal["cavity"]) -> FieldMap: ...

    @overload
    def take(
        self, ids: Sequence[int], id_nature: Literal["cavity"]
    ) -> list[FieldMap]: ...

    @overload
    def take(self, ids: int, id_nature: Literal["element"]) -> Element: ...

    @overload
    def take(
        self, ids: Sequence[int], id_nature: Literal["element"]
    ) -> list[Element]: ...

    @overload
    def take(self, ids: str, id_nature: Literal["name"]) -> Element: ...

    @overload
    def take(
        self, ids: Sequence[str], id_nature: Literal["name"]
    ) -> list[Element]: ...

    @overload
    def take(
        self,
        ids: nested_elements_id,
        id_nature: Literal["cavity", "element", "name"],
    ) -> list[Sequence[Element]]: ...

    def take(
        self,
        ids: element_id | elements_id | nested_elements_id,
        id_nature: Literal["cavity", "element", "name"],
    ) -> (
        Element
        | list[Element]
        | list[Sequence[Element]]
        | FieldMap
        | list[FieldMap]
        | list[Sequence[FieldMap]]
    ):
        """Convert list of indexes or names to a list of :class:`.Element`."""
        if isinstance(ids, Sequence) and not isinstance(ids, str):
            return [self.take(idx, id_nature) for idx in ids]

        match id_nature:
            case "cavity":
                assert isinstance(ids, int)
                output = self.l_cav[ids]
            case "element":
                assert isinstance(ids, int)
                output = self[ids]
            case "name":
                name = ids
                assert isinstance(name, str)
                output = first(self, condition=lambda elt: elt.name == name)
            case _:
                raise IOError(f"{id_nature = } not understood.")
        return output

    def pickle(
        self, pickler: MyPickler, path: Path | str | None = None
    ) -> Path:
        """Pickle (save) the object.

        This is useful for debug and temporary saves; do not use it for long
        time saving.

        """
        if path is None:
            path = self.files["accelerator_path"] / "list_of_elements.pkl"
        assert isinstance(path, Path)
        pickler.pickle(self, path)

        if isinstance(path, str):
            path = Path(path)
        return path

    @classmethod
    def from_pickle(cls, pickler: MyPickler, path: Path | str) -> Self:
        """Instantiate object from previously pickled file."""
        list_of_elements = pickler.unpickle(path)
        return list_of_elements  # type: ignore
