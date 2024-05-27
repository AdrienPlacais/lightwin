"""Define a class to pickle objects.

"pickling" a file comes down to saving it in binary format. It can be loaded
and used again later, even with a different Python instance. This is useful
when you want to study a :class:`.Fault` that took a long time to be
compensated, or a :class:`.SimulationOutput` obtained by a time-consuming
TraceWin multiparticle simulation.

.. warning::
    This a very basic pickling. Do not use for long-term storage, but for debug
    only.

.. note::
    Some attributes such as lambda function in :class:`.FieldMap` or modules in
    :class:`.SimulationOutput` cannot be pickled by the built-in `pickle`
    module. I do not plan to refactor them, so for now we stick with
    `cloudpickle` module.

Some objects have built-in `pickle` and `unpickle` methods, namely:

    - :class:`.Accelerator`
    - :class:`.Fault`
    - :class:`.FaultScenario`
    - :class:`.ListOfElements`
    - :class:`.SimulationOutput`

"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType

try:
    import cloudpickle
except ModuleNotFoundError:
    logging.error(
        "cloudpickler module not found. This should not be a problem if you "
        "do not want to use MyCloudPickler."
    )


class MyPickler(ABC):
    """Define an object that can save/load arbitrary objects to files."""

    @abstractmethod
    def pickle(self, my_object: object, path: Path | str) -> None:
        """Pickle ("save") the object to a binary file."""
        pass

    @abstractmethod
    def unpickle(self, path: Path | str) -> object:
        """Unpickle ("load") the given path to recreate original object."""
        pass


class MyCloudPickler(MyPickler):
    """Define a :class:`.Pickler` that can handle modules and lambda functions.

    This pickler should not raise errors, but all attributes may not be
    properly re-created.

    """

    def __init__(self) -> None:
        """Import the necessary module."""
        assert isinstance(cloudpickle, ModuleType)

    def pickle(self, my_object: object, path: Path | str) -> None:
        """Pickle ("save") the object to a binary file."""
        with open(path, "wb") as f:
            cloudpickle.dump(my_object, f)
        logging.info(f"Pickled {my_object} to {path}.")

    def unpickle(self, path: Path | str) -> object:
        """Unpickle ("load") the given path to recreate original object."""
        with open(path, "rb") as f:
            my_object = cloudpickle.load(f)
        return my_object
