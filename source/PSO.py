#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:44:53 2022.

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3

from pymoo.factory import get_termination, get_reference_directions
from pymoo.optimize import minimize

from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights

from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetricAnimation

from pymoo.visualization.pcp import PCP

from pymoo.core.callback import Callback

from helper import printc, create_fig_if_not_exist

STR_ALGORITHM = "NSGA-II"
FLAG_VERBOSE = False
FLAG_HYPERVOLUME = False
FLAG_RUNNING = False
FLAG_CONVERGENCE_HISTORY = True  # Heavier in terms of memory usage
FLAG_CONVERGENCE_CALLBACK = False
FLAG_CV = False
FLAG_DESIGN_SPACE = True


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
                 wrapper_args, phi_s_limits, **kwargs):
        self.wrapper_pso = wrapper_fun
        self.fault = wrapper_args[0]
        self.fun_residual = wrapper_args[1]
        self.d_idx = wrapper_args[2]
        self.phi_s_limits = phi_s_limits
        self.n_obj = n_obj

        # printc("Warning PSO.__init__: ", opt_message="Bounds manually " +
               # "modified.")
        # xl = np.array([
            # 0.2753397474805297, 1.64504348936957, 5.132411286781306,
            # 1.827816962622414, 3.3763299623648138, 4.9824476431763784,
            # # 5.694483445783, 5.712739409783303, 5.733318868185172,
            # # 5.75486203532526, 5.779637177905215, 5.805588466008629])
            # 0.861770057293058, 0.8750412568825341, 0.9060682969220509,
            # 0.9286910481509094, 0.9304272, 0.9304272])
        # xu = xl + 1e-5
        # xl -= 1e-5
        print(f"Number of objectives: {n_obj}")
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         # xl=xl, xu=xu)
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
        out : dict
            Holds function values in "F" key and constraints in "G".
        """
        out["F"], results = self.wrapper_pso(
            x, self.fault, self.fun_residual, self.d_idx)

        out_G = []
        for i in range(len(results['phi_s_rad'])):
            out_G.append(self.phi_s_limits[i][0] - results['phi_s_rad'][i])
            out_G.append(results['phi_s_rad'][i] - self.phi_s_limits[i][1])
        out["G"] = np.array(out_G)


def perform_pso(problem):
    """Perform the PSO."""
    if STR_ALGORITHM == 'NSGA-II':
        algorithm = NSGA2(pop_size=100,
                          eliminate_duplicates=True)

    elif STR_ALGORITHM == 'NSGA-III':
        # ref_dirs = get_reference_directions("das-dennis", problem.n_obj,
                                            # n_partitions=6)
        ref_dirs = get_reference_directions("energy", problem.n_obj, 90)
        algorithm = NSGA3(pop_size=75, # 500
                          ref_dirs=ref_dirs,
                          )

    termination = _set_termination()
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=FLAG_CONVERGENCE_HISTORY,
                   verbose=FLAG_VERBOSE,
                   callback=MyCallback(),
                   )
    return res


def _set_termination():
    """Set a termination condition."""
    d_termination = {
        'NSGA-II': get_termination("n_gen", 60),
        'NSGA-III': get_termination("n_gen", 200),# 200
    }
    termination = d_termination[STR_ALGORITHM]

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


def mcdm(res, weights, fault_info, compare=None):
    """Perform Multi-Criteria Decision Making."""
    print(f"Shapes: X={res.X.shape}, F={res.F.shape}, G={res.G.shape}")
    # Multi-Criteria Decision Making
    f_l = res.F.min(axis=0)
    f_u = res.F.max(axis=0)
    for _l, _u in zip(f_l, f_u):
        print(f"Pre-scale f: [{_l}, {_u}]")

    d_approx = {'ideal': res.F.min(axis=0),
                'nadir': res.F.max(axis=0),}

    n_f = (res.F - d_approx['ideal']) / (d_approx['nadir'] - d_approx['ideal'])
    f_l = n_f.min(axis=0)
    f_u = n_f.max(axis=0)

    pd_best_sol, d_asf, d_pw = _best_solutions(res, n_f, weights, fault_info,
                                               compare=compare)
    fault_info['resume'] = pd_best_sol
    d_opti = {'asf': d_asf, 'pw': d_pw}

    return d_opti, d_approx


def _best_solutions(res, n_f, weights, fault_info, compare=None):
    """Look for best solutions according to various criteria."""
    # Create a pandas dataframe for the final objective values
    n_var = res.X.shape[1]
    columns = ['Criteria', 'i'] + fault_info['l_prop_label'][:n_var] \
        + fault_info['l_obj_label']
    pd_best_sol = pd.DataFrame(columns=(columns))

    # Best solution according to ASF and Pseudo Weights
    decomp = ASF()
    min_asf = decomp.do(n_f, 1. / weights)

    idx = min_asf.argmin()
    d_asf = {'idx': idx, 'X': res.X[idx].tolist(), 'F': res.F[idx].tolist()}
    pd_best_sol.loc[0] = ['ASF', idx] + res.X[idx].tolist() \
            + res.F[idx].tolist()
    # Best solution according to Pseudo-Weights
    idx = PseudoWeights(weights).do(n_f)
    d_pw = {'idx':  idx, 'X': res.X[idx].tolist(), 'F': res.F[idx].tolist()}
    pd_best_sol.loc[1] = ['PW', idx] + res.X[idx].tolist() \
            + res.F[idx].tolist()

    for col in pd_best_sol:
        if 'phi' in col:
            pd_best_sol[col] = np.rad2deg(pd_best_sol[col])
    print(f"\n\n{pd_best_sol[['Criteria', 'i'] + fault_info['l_obj_label']]}"
          + '\n\n')

    # Viualize solutions
    if res.F.shape[1] != 3:
        kwargs_matplotlib = {'close_on_destroy': False}
        best_sol_plot = PCP(title=("Run", {'pad': 30}),
                            n_ticks=10,
                            legend=(True, {'loc': "upper left"}),
                            labels=fault_info['l_obj_label'],
                            **kwargs_matplotlib,
                            )
        best_sol_plot.set_axis_style(color="grey", alpha=0.5)
        best_sol_plot.add(res.F, color="grey", alpha=0.3)
        best_sol_plot.add(d_asf['F'], linewidth=5, color="red", label='ASF')
        best_sol_plot.add(d_pw['F'], linewidth=5, color="blue", label='PW')
        best_sol_plot.show()
        best_sol_plot.ax.grid(True)
    else:
        fig = plt.figure(2)
        axx = fig.add_subplot(projection='3d')
        kwargs = {'marker': '^', 'alpha': 1, 's': 30}
        tmp = np.log(res.F)
        axx.scatter(tmp[:, 0], tmp[:, 1], tmp[:, 2])
        axx.scatter(tmp[d_asf['idx'], 0], tmp[d_asf['idx'], 1],
                    tmp[d_asf['idx'], 2],
                    color='green', label='ASF', **kwargs)
        axx.scatter(tmp[d_pw['idx'], 0], tmp[d_pw['idx'], 1],
                    tmp[d_pw['idx'], 2],
                    color='blue', label='PW', **kwargs)
        if compare is not None:
            axx.scatter(compare[0], compare[1], compare[2], color='k',
                        label='Least-squares', **kwargs)
        axx.set_xlabel(r"$W_{kin}$")
        axx.set_ylabel(r"$\phi$")
        axx.set_zlabel(r"$M$")
        axx.legend()
        fig.show()

    return pd_best_sol, d_asf, d_pw


def convergence_callback(callback, l_obj_label):
    """Plot convergence info using the results of the callback."""
    fig = plt.figure(58)
    axx = fig.add_subplot(111)
    axx.set_title("Convergence")
    axx.plot(callback.n_evals, callback.opt, label=l_obj_label)
    axx.set_xlabel('Number of evaluations')
    axx.set_ylabel('res.F[0, :]')
    axx.set_yscale("log")
    axx.legend()
    axx.grid(True)


def convergence_history(hist, d_approx, str_obj):
    """Study the convergence of the algorithm."""
    # Convergence study
    n_evals = []      # Num of func evaluations
    hist_f = []       # Objective space values in each generation
    hist_cv = []      # Constraint violation in each generation
    hist_cv_avg = []  # Average contraint violation in the whole population

    for algo in hist:
        n_evals.append(algo.evaluator.n_eval)
        opt = algo.opt
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # Filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_f.append(opt.get("F")[feas])

    k = np.where(np.array(hist_cv) <= 0.)[0].min()
    print(f"At least one feasible solution in Generation {k} after",
          f"{n_evals[k]} evaluations.")
    vals = hist_cv_avg
    # Can be replaced by hist_cv to analyse the least feasible optimal solution
    # instead of the population

    k = np.where(np.array(vals) <= 0.)[0].min()
    print(f"Whole population feasible in Generation {k} after {n_evals[k]}",
          "evaluations.")

    if FLAG_CV:
        fig, axx = create_fig_if_not_exist(61, [211, 212])

        axx[0].plot(n_evals, hist_cv_avg, marker='o', c='k', lw=.7,
                    label="Avg. CV of pop.")
        axx[0].axvline(n_evals[k], ls='--', label='All feasible', c='r')
        axx[0].set_title("Convergence")

        axx[1].plot(n_evals, hist_cv, marker='o', c='b', lw=.7,
                   label="Least feasible opt. sol.")
        for i in range(2):
            axx[i].set_xlabel("Function evaluations")
            axx[i].set_ylabel("Constraint Violation")
            axx[i].legend()
            axx[i].grid(True)
        fig.show()

    if FLAG_HYPERVOLUME:
        _convergence_hypervolume(n_evals, hist_f, d_approx,
                                 str_obj)

    if FLAG_RUNNING:
        _convergence_running_metrics(hist)


def convergence_design_space(hist, d_opti, lsq_x=None):
    """Represent the variables that were tried during optimisation."""
    hist_xf = []      # Explored variables (Feasible)
    hist_xu = []      # Explored variables (Unfeasible)

    for algo in hist:
        pop = algo.pop
        feas = np.where(pop.get("feasible"))[0]
        hist_xf.append(pop.get("X")[feas])
        unfeas = np.where(~pop.get("feasible"))[0]
        hist_xu.append(pop.get("X")[unfeas])

    _plot_design(hist_xf, hist_xu, d_opti, lsq_x)


def _convergence_hypervolume(n_eval, hist_f, d_approx, str_obj):
    """Study convergence using hypervolume. Not adapted when too many dims."""
    # Dictionary for reference points
    # They must be typical large values for the objective
    d_ref = {
        'energy': 70.,
        'phase': 0.5,
        'mismatch_factor': 1.,
        'M_11': None,
        'M_12': None,
        'M_21': None,
        'M_22': None,
        'eps': None,
        'twiss_alpha': None,
        'twiss_beta': None,
        'twiss_gamma': None,
    }
    ref_point = [d_ref[obj] for obj in str_obj]
    metric = Hypervolume(
        ref_point=ref_point,
        ideal=d_approx['ideal'],
        nadir=d_approx['nadir'],
    )

    h_v = [metric.do(_F) for _F in hist_f]

    printc("Warning PSO._convergence_hypervolume: ", opt_message="manually"
           + "added the optimal point to the hypervolume plot.")
    ideal = np.abs([1075.34615847359 - 1075.346158310222,
                    77.17023331557031 - 77.17023332120309,
                    7.324542512066046e-06])
    h_v.append(metric.do(ideal))

    _, axx = create_fig_if_not_exist(60, [111])
    axx = axx[0]

    axx.plot(n_eval + [n_eval[-1] + n_eval[0]], h_v, lw=.7, marker='o', c='k')
    axx.set_title("Objective space")
    axx.set_xlabel("Function evaluations")
    axx.set_ylabel("Hypervolume")
    axx.grid(True)


def _convergence_running_metrics(hist):
    """Study convergence using running metrics."""
    running = RunningMetricAnimation(
        delta_gen=10,
        n_plots=10,
        # only_if_n_plots=True,
        key_press=True,
        do_show=True,)
    for algorithm in hist:
        running.update(algorithm)


def _plot_design(hist_xf, hist_xu, d_opti, lsq_x=None):
    """Plot for each cavity the norm and phase that were tried."""
    n_cav = np.shape(hist_xf[-1])[1] / 2
    n_cav = int(n_cav)
    assert n_cav == 6, "Not designed for number of cavities different from 6."

    fig, axx = create_fig_if_not_exist(63, range(231, 237))
    color = ['g', 'b']

    for i in range(n_cav):
        # Plot X corresponding to feasible and unfeasible F
        for xf, xu in zip(hist_xf, hist_xu):
            if np.shape(xf)[0] > 0:
                axx[i].scatter(np.mod(xf[:, i], 2. * np.pi), xf[:, i + n_cav],
                               marker='o', c='r', alpha=.5)
            if np.shape(xu)[0] > 0:
                axx[i].scatter(np.mod(xu[:, i], 2. * np.pi), xu[:, i + n_cav],
                               marker='x', c='r', alpha=.5)
        # Plot solution(s) X found in mcdm:
        for j, key in enumerate(d_opti.keys()):
            axx[i].scatter(d_opti[key]['X'][i], d_opti[key]['X'][i + n_cav],
                           marker='^', c=color[j], alpha=1, label=key)
        # Plot solution found by LSQ
        if lsq_x is not None:
            axx[i].scatter(np.mod(lsq_x[i], 2. * np.pi), lsq_x[i + n_cav],
                           marker='^', c='k', alpha=1, label='LSQ')
        axx[i].grid(True)
        axx[i].set_xlim([0., 2. * np.pi])
        axx[i].set_ylim([0.387678, 0.9304272])
    axx[0].set_ylabel(r'$k_e$')
    axx[0].legend()
    axx[3].set_ylabel(r'$k_e$')
    axx[4].set_xlabel(r'$\phi_0$')
    fig.show()
    printc("Warning PSO._space_design_exploration: ", opt_message="Limits "
           + "manually entered.")


def set_weights(l_obj_str):
    """Set array of weights for the different objectives."""
    d_weights = {'energy': 2.,
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
