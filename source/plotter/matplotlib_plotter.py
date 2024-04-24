"""Define a matplotlib-based plotter."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from plotter.i_plotter import IPlotter


class MatplotlibPlotter(IPlotter):

    def __init__(self) -> None:
        raise NotImplementedError
