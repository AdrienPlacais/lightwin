#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:44:53 2022.

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetric


class MyProblem(ElementwiseProblem):
    """Class holding PSO."""

    def __init__(self, wrapper, n_var, bounds, wrapper_args):
        self.wrapper = wrapper
        self.fault = wrapper_args[0]
        self.fun_residual = wrapper_args[1]
        self.d_idx = wrapper_args[2]
        super().__init__(n_var=n_var,
                         n_obj=len(self.d_idx['l_ref']),
                         n_constr=0,
                         xl=bounds[:, 0], xu=bounds[:, 1])

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.wrapper(x, self.fault, self.fun_residual, self.d_idx)
        out["G"] = None


def perform_pso(problem):
    """Perform the PSO."""
    algorithm = NSGA2(pop_size=100,
                      n_offsprings=20,
                      sampling=get_sampling("real_random"),
                      crossover=get_crossover("real_sbx", prob=.9, eta=15),
                      mutation=get_mutation("real_pm", eta=20),
                      # Ensure that offsprings are different from each
                      # other and from existing population:
                      eliminate_duplicates=True)
    termination = get_termination("n_gen", 100)
    res = minimize(problem, algorithm, termination, seed=1,
                   save_history=True,
                   verbose=True)
    return res


def mcdm(res, weights):
    """Perform Multi-Criteria Decision Making."""
    # Solutions
    F = res.F

    # Multi-Criteria Decision Making
    fl = F.min(axis=0)
    fu = F.max(axis=0)
    for _l, _u in zip(fl, fu):
        print(f"Scale f: [{_l}, {_u}]")

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

    fl = nF.min(axis=0)
    fu = nF.max(axis=0)
    for _l, _u in zip(fl, fu):
        print(f"Scale f: [{_l}, {_u}]")

    decomp = ASF()
    minASF = decomp.do(nF, 1. / weights)
    i = minASF.argmin()
    print('Best solution regarding ASF:\nPoint i = %s\tF = %s' % (i, F[i]))

    i = PseudoWeights(weights).do(nF)
    print('Best solution regarding Pseudo Weights:\nPoint i = %s\tF = %s'
          % (i, F[i]))

    return res.X[i], approx_ideal, approx_nadir


def convergence(hist, approx_ideal, approx_nadir):
    """Study the convergence of the algorithm."""
    flag_hypervolume = False
    flag_running = False

    # Convergence study
    n_evals = []      # Num of func evaluations
    hist_F = []       # Objective space values in each generation
    hist_cv = []      # Constraint violation in each generation
    hist_cv_avg = []  # Average contraint violation in the whole population

    for algo in hist:
        n_evals.append(algo.evaluator.n_eval)
        opt = algo.opt
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())
        # Filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])
    k = np.where(np.array(hist_cv) <= 0.)[0].min()
    print(f"At least one feasible solution in Generation {k} after",
          f"{n_evals[k]} evaluations.")
    vals = hist_cv_avg
    # Can be replaced by hist_cv to analyse the least feasible optimal solution
    # instead of the population

    k = np.where(np.array(vals) <= 0.)[0].min()
    print(f"Whole population feasible in Generation {k} after {n_evals[k]}",
          "evaluations.")

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(n_evals, hist_cv_avg, marker='o', c='k', lw=.7,
               label="Avg. CV of pop.")
    ax[0].axvline(n_evals[k], ls='--', label='All feasible', c='r')
    ax[0].set_title("Convergence")

    ax[1].plot(n_evals, hist_cv, marker='o', c='b', lw=.7,
               label="Least feasible opt. sol.")

    for i in range(2):
        ax[i].set_xlabel("Function Evaluations")
        ax[i].set_ylabel("Constraint Violation")
        ax[i].legend()
    plt.show()

    if flag_hypervolume:
        _convergence_hypervolume(n_evals, hist_F, approx_ideal, approx_nadir)

    if flag_running:
        _convergence_running_metrics(hist)


def _convergence_hypervolume(n_evals, hist_F, approx_ideal, approx_nadir):
    """Study convergence using hypervolume. Not adapted when too many dims."""
    metric = Hypervolume(
        ref_point=np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
        norm_ref_point=False,
        zero_to_one=True,
        ideal=approx_ideal,
        nadir=approx_nadir,
    )

    hv = [metric.do(_F) for _F in hist_F]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    ax.plot(n_evals, hv, lw=.7, marker='o', c='k')

    ax.set_title("Objective Space")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Hypervolume")
    plt.show()


def _convergence_running_metrics(hist):
    """Study convergence using running metrics."""
    running = RunningMetric(delta_gen=2,
                            n_plots=10,
                            only_if_n_plots=True,
                            key_press=True,
                            do_show=True)
    for algorithm in hist:
        running.notify(algorithm)
