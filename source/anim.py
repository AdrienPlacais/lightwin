#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:39:32 2022.

@author: placais
"""

import matplotlib.animation as animation
import numpy as np
from helper import create_fig_if_not_exist


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, hist, interval=300, blit=True, repeat=False):
        self.numpoints = hist[0].pop.size
        self.n_cav = int(np.shape(hist[0].pop.get('X'))[1] / 2)

        self.hist = hist
        self.stream = self.data_stream()

        self.fig, self.axx = create_fig_if_not_exist(63, range(231, 237))
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=interval,
            init_func=self.setup_plot, blit=blit, repeat=repeat)

    def setup_plot(self):
        """Initialize drawing of the scatter plot."""
        x_i = next(self.stream)
        self.l_scat = []

        for j, ax in enumerate(self.axx):
            x_min = 0. - .01 * 2. * np.pi
            x_max = 2.01 * np.pi
            y_min = .99 * 0.387678
            y_max = 1.01 * 0.93024272
            ax.axis([x_min, x_max, y_min, y_max])
            self.l_scat.append(
                ax.scatter(x_i[:, j], x_i[:, j + self.n_cav], c='r', s=5)
            )
        return self.l_scat

    def data_stream(self):
        """Create the generator object."""
        generator_data = (algo.pop.get('X') for algo in self.hist)
        for data in generator_data:
            yield data

    def update(self, frame, *fargs):
        """Update the figure at the new frame."""
        # List of X values for each population member
        x_frame = next(self.stream)
        for j, scat in enumerate(self.l_scat):
            x_j = np.column_stack((x_frame[:, j],
                                   x_frame[:, j + self.n_cav]))
            scat.set_offsets(x_j)
        return self.l_scat
