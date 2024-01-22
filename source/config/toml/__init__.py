"""This subpackage defines methods to validate the ``.toml`` config file.

It is a first filter, so that most obvious input mistakes are detected at the
beginning of the script rather than after hours of simulation.
Also, some complementary data is calculated.

.. note::
    I think that some day this subpackage will be removed. As most of the
    ``.toml`` entries are simply objects ``kwargs``, I could create all the
    necessary objects at the beginning of the script, with the tests at the
    object instantiation.
    It would be clearer I think...

"""
