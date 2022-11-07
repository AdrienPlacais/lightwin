#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:39:32 2022.

@author: placais

TODO : discriminate feasible from unfeasible solutions
"""

from matplotlib import animation
import numpy as np


class AnimatedScatterDesign(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, fig, hist, n_cav, interval=300, blit=True,
                 repeat=False):
        self.fig = fig
        self.axx = fig.get_axes()
        self.l_scat = []

        self.numpoints = hist[0].pop.size
        self.n_cav = n_cav
        frames = len(hist) - 1

        self.hist = hist
        self.stream = self.data_stream()

        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=interval, frames=frames,
            init_func=self.setup_plot, blit=blit, repeat=repeat)

        writer = animation.ImageMagickWriter(fps=2)
        self.anim.save("anim.gif", writer=writer)

    def setup_plot(self):
        """Initialize drawing of the scatter plot."""
        x_ini = next(self.stream)

        for j, axx in enumerate(self.axx):
            x_j = np.column_stack((x_ini[:, j],
                                   x_ini[:, j + self.n_cav]))
            self.l_scat.append(
                axx.scatter(x_j[0, :], x_j[1, :], c='r', s=5)
            )
        return self.l_scat

    def update(self, frame):
        """Update the figure at the new frame."""
        # List of X values for each population member
        x_frame = next(self.stream)

        for j, scat in enumerate(self.l_scat):
            x_j = np.column_stack((x_frame[:, j],
                                   x_frame[:, j + self.n_cav]))
            scat.set_offsets(x_j)
        return self.l_scat

    def data_stream(self):
        """Create the generator object."""
        generator_data = (algo.pop.get('X') for algo in self.hist)
        for data in generator_data:
            yield data
