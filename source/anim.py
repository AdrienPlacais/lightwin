#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:39:32 2022.

@author: placais
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from helper import create_fig_if_not_exist


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, hist, numpoints=100):
        # self.hist = hist
        self.numpoints = len(hist[0].pop.get('X'))
        self.stream = self.data_stream()
        self.n_cav = 6

        self.data = [algo.pop.get('X') for algo in hist]
        frames = len(self.data) - 1

        # Setup the figure and axes...
        self.fig, self.axx = create_fig_if_not_exist(63, range(231, 237))
        for ax in self.axx:
            x_min = 0. - .1 * 2. * np.pi
            x_max = 2.1 * np.pi
            y_min = .9 * 0.387678
            y_max = 1.1 * 0.93024272
            ax.axis([x_min, x_max, y_min, y_max])

        # Then setup FuncAnimation.
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=300, frames=frames,
            init_func=self.setup_plot, blit=True, repeat=False)

    def setup_plot(self):
        """
        Initialize drawing of the scatter plot.

        Must return an iterable_of_artists.
        """
        x_i = next(self.stream)
        self.l_scat = []
        for j, ax in enumerate(self.axx):
            self.l_scat.append(
                ax.scatter(x_i[:, j], x_i[:, j + self.n_cav],
                           c='r', s=10),
            )
        return self.l_scat

    def data_stream(self):
        """Give the norm and phase at proper step."""
        i = -1
        while True:
            i += 1
            yield self.data[i]

    def update(self, frame, *fargs):
        r"""
        Update the figure at the new frame.

        Parameters
        ----------
        frame : `~.collections.PathCollection`
            Next frame value.

        Return
        ------
        self.l_scat : iterable of artists
        """
        # List of X values for each population member
        x_frame = next(self.stream)
        for j, scat in enumerate(self.l_scat):
            x_j = np.column_stack((x_frame[:, j],
                                   x_frame[:, j + self.n_cav]))
            scat.set_offsets(x_j)
        return self.l_scat
