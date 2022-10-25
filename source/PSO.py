#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:44:53 2022.

@author: placais

FIXME : removed pymoo.util.termination in import and _set_termination
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, \
    get_termination, get_reference_directions
# from pymoo.util.termination.default import \
#     MultiObjectiveSpaceToleranceTermination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetric
from pymoo.visualization.pcp import PCP
from pymoo.core.callback import Callback


str_algorithm = "NSGA-III"
flag_verbose = False
flag_hypervolume = False
flag_running = False
flag_convergence_history = True  # Heavier in terms of memory usage
flag_convergence_callback = True


class MyCallback(Callback):
    """Class to receive notification from algo at each iteration."""

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []

    def notify(self, algorithm):
        """Notify."""
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)


class MyProblem(ElementwiseProblem):
    """Class holding PSO."""

    def __init__(self, wrapper_fun, n_var, n_obj, n_constr, bounds,
                 wrapper_args, phi_s_limits):
        self.wrapper_pso = wrapper_fun
        self.fault = wrapper_args[0]
        self.fun_residual = wrapper_args[1]
        self.d_idx = wrapper_args[2]
        self.phi_s_limits = phi_s_limits
        self.n_obj = n_obj
        print('number of objectives:', n_obj)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=bounds[:, 0], xu=bounds[:, 1])
        if n_constr > 0:
            print(f"{n_constr} constraints on phi_s:\n{phi_s_limits}")

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Compute residues and constraints.

        Parameters
        ----------
        x : np.array
            Holds phases (first half) and norms (second half) of cavities.
        out : truc
            Mmmh
        """
        out["F"], results = self.wrapper_pso(
            x, self.fault, self.fun_residual, self.d_idx)

        out_G = []
        for i in range(len(results['phi_s_rad'])):
            out_G.append(self.phi_s_limits[i][0] - results['phi_s_rad'][i])
            out_G.append(results['phi_s_rad'][i] - self.phi_s_limits[i][1])
        out["G"] = np.array(out_G)

    def cheat(self):
        """Set a solution that works for comparison."""
        # Results found with least squares
        phi_0_cheat = np.deg2rad(np.array([22.091899,
                                           57.085305,
                                           224.586449,
                                           70.098014,
                                           187.441566]))
        k_e_cheat = np.array([3.523201, 3.893933, 3.948438,
                              3.402330, 3.527800])
        X_cheat = np.hstack((phi_0_cheat, k_e_cheat))
        print('X_cheat:', X_cheat)

        F_cheat, brok_res_cheat = self.wrapper_pso(
            X_cheat, self.fault, self.fun_residual, self.d_idx)
        print('F_cheat:', F_cheat)
        print('phi_s_cheat:', np.rad2deg(brok_res_cheat['phi_s_rad']))

        G_cheat = []
        for i in range(len(brok_res_cheat['phi_s_rad'])):
            G_cheat.append(self.phi_s_limits[i][0]
                           - brok_res_cheat['phi_s_rad'][i])
            G_cheat.append(brok_res_cheat['phi_s_rad'][i]
                           - self.phi_s_limits[i][1])
        G_cheat = np.array(G_cheat)
        print('G_cheat:', G_cheat)
        return X_cheat, F_cheat, G_cheat


def perform_pso(problem):
    """Perform the PSO."""
    if str_algorithm == 'NSGA-II':
        algorithm = NSGA2(pop_size=100,
                          n_offsprings=10,
                          sampling=get_sampling("real_random"),
                          crossover=get_crossover("real_sbx", prob=.9, eta=10),
                          mutation=get_mutation("real_pm", eta=5),
                          # Ensure that offsprings are different from each
                          # other and from existing population:
                          eliminate_duplicates=True)

    elif str_algorithm == 'NSGA-III':
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj,
                                            n_partitions=6)
        algorithm = NSGA3(pop_size=75, # 500
                          ref_dirs=ref_dirs,
                          )

    termination = _set_termination()
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=flag_convergence_history,
                   verbose=flag_verbose,
                   callback=MyCallback(),
                   )

    return res


def _set_termination():
    """Set a termination condition."""
    d_termination = {
        'NSGA-II': get_termination("n_gen", 1000),
        'NSGA-III': get_termination("n_gen", 200),# 200
    }
    termination = d_termination[str_algorithm]

    # termination = MultiObjectiveSpaceToleranceTermination(
    #     # What is the tolerance in the objective space on average. If the value
    #     # is below this bound, we terminate.
    #     tol=1e-5,
    #     # To make the criterion more robust, we consider the last n generations
    #     # and take the maximum. This considers the worst case in a window.
    #     n_last=10,
    #     # As a fallback, the generation number can be used. For some problems,
    #     # the termination criterion might not be reached; however, an upper
    #     # bound for generations can be defined to stop in that case.
    #     n_max_gen=1000,
    #     # Defines whenever the termination criterion is calculated by default,
    #     # every 10th generation.
    #     nth_gen=10,
    # )
    return termination


def mcdm(res, weights, fault_info):
    """Perform Multi-Criteria Decision Making."""
    print(f"Shapes: X={res.X.shape}, F={res.F.shape}, G={res.G.shape}")
    # Multi-Criteria Decision Making
    fl = res.F.min(axis=0)
    fu = res.F.max(axis=0)
    for _l, _u in zip(fl, fu):
        print(f"Pre-scale f: [{_l}, {_u}]")

    approx_ideal = res.F.min(axis=0)
    approx_nadir = res.F.max(axis=0)

    nF = (res.F - approx_ideal) / (approx_nadir - approx_ideal)
    fl = nF.min(axis=0)
    fu = nF.max(axis=0)

    pd_best_sol, i = _best_solutions(res, nF, weights, fault_info)
    fault_info['resume'] = pd_best_sol

    return res.X[i], approx_ideal, approx_nadir


def _best_solutions(res, nF, weights, fault_info):
    """Look for best solutions according to various criteria."""
    two_n_cav = res.X.shape[1]
    columns = ['Criteria', 'i'] + fault_info['l_prop_label'][:two_n_cav] \
        + fault_info['l_obj_label']
    pd_best_sol = pd.DataFrame(columns=(columns))

    decomp = ASF()
    minASF = decomp.do(nF, 1. / weights)
    i_asf = minASF.argmin()
    pd_best_sol.loc[0] = ['ASF', i_asf] + res.X[i_asf].tolist() \
        + res.F[i_asf].tolist()

    i_pw = PseudoWeights(weights).do(nF)
    pd_best_sol.loc[1] = ['PW', i_pw] + res.X[i_pw].tolist() \
        + res.F[i_pw].tolist()

    for col in pd_best_sol:
        if 'phi' in col:
            pd_best_sol[col] = np.rad2deg(pd_best_sol[col])
    print('\n\n', pd_best_sol[['Criteria', 'i'] + fault_info['l_obj_label']],
          '\n\n')

    # Viualize solutions
    kwargs_matplotlib = {'close_on_destroy': False}
    best_sol_plot = PCP(title=("Run", {'pad': 30}),
                        n_ticks=10,
                        legend=(True, {'loc': "upper left"}),
                        labels=fault_info['l_obj_label'],
                        **kwargs_matplotlib,
                        )
    best_sol_plot.set_axis_style(color="grey", alpha=0.5)
    best_sol_plot.add(res.F, color="grey", alpha=0.3)
    best_sol_plot.add(res.F[i_asf], linewidth=5, color="red", label='ASF')
    best_sol_plot.add(res.F[i_pw], linewidth=5, color="blue", label='PW')
    best_sol_plot.show()
    best_sol_plot.ax.grid(True)
    return pd_best_sol, i_asf


def convergence_callback(callback, l_obj_label):
    """Plot convergence info using the results of the callback."""
    fig = plt.figure(58)
    ax = fig.add_subplot(111)
    ax.set_title("Convergence")
    ax.plot(callback.n_evals, callback.opt, label=l_obj_label)
    ax.set_xlabel('Number of evaluations')
    ax.set_ylabel('res.F[0, :]')
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)


def convergence_history(hist, approx_ideal, approx_nadir):
    """Study the convergence of the algorithm."""
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

    fig = plt.figure(56)
    ax = [fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]

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
        ref_point=np.array([1.1, 1.1]),
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


def _convergence_running_metrics(hist):
    """Study convergence using running metrics."""
    running = RunningMetric(delta_gen=2,
                            n_plots=10,
                            only_if_n_plots=True,
                            key_press=True,
                            do_show=True)
    for algorithm in hist:
        running.notify(algorithm)


def set_weights(l_obj_str):
    """Set array of weights for the different objectives."""
    d_weights = {'energy': 1.,
                 'phase': 1.,
                 'eps': 1.,
                 'twiss_alpha': 1.,
                 'twiss_beta': 1.,
                 'twiss_gamma': 1.,
                 'M_11': 1.,
                 'M_12': 1.,
                 'M_21': 1.,
                 'M_22': 1.,
                 'mismatch_factor': 1,
                 }
    weights = [d_weights[obj] for obj in l_obj_str]
    return np.array(weights)
