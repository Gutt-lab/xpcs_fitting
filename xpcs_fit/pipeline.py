"""Top-level orchestration: run the joint-smooth BO/VarPro pipeline
against the independent-CF ground-truth check, on one simulated run.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .engines.bayesian import fit_joint_bo_varpro, select_common_lambda_bo
from .engines.curve_fit import fit_independent_curvefit
from .plotting import (
    plot_D_vs_q_three_way,
    plot_joint_vs_independent_curves,
    plot_joint_vs_independent_parameters,
    plot_lambda_scan,
    plot_residuals,
)
from .preprocessing import bootstrap_D_estimate
from .simulate import simulate_xpcs_data
from .uncertainty import joint_param_errors


def run_joint_smooth_analysis(cfg: Config, lambdas=(0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                               holdout_stride=5, n_iter_joint=25, n_iter_indep=None,
                               make_plots=True) -> dict:
    """Run joint-smooth BO/VarPro (with lag-holdout lambda selection)
    against independent per-channel CF, on synthetic data drawn from
    `cfg`'s ground-truth parameters.

    `make_plots` gates the core diagnostic plots; `cfg.plot_residuals`
    and `cfg.plot_lambda_scan` separately gate the two optional extra
    plots.

    Returns a dict with the fitted arrays for both engines, the
    selected lambda, and the lambda scan.
    """
    if n_iter_indep is None:
        n_iter_indep = cfg.n_iter_bo

    tau_list, g2_list, sig_list, qs = simulate_xpcs_data(cfg)

    n_q = len(qs)
    print(f"\n[joint-smooth-BO] SIMULATED  ({n_q} q-channels)")

    D_true_arr = np.full(n_q, cfg.sim_D_true) if np.isscalar(cfg.sim_D_true) else np.asarray(cfg.sim_D_true)
    beta_true_arr = np.full(n_q, cfg.sim_beta_true) if np.isscalar(cfg.sim_beta_true) else np.asarray(cfg.sim_beta_true)
    B_true_arr = np.full(n_q, cfg.sim_B_true) if np.isscalar(cfg.sim_B_true) else np.asarray(cfg.sim_B_true)
    alpha_true_arr = np.full(n_q, cfg.sim_alpha_true) if np.isscalar(cfg.sim_alpha_true) else np.asarray(cfg.sim_alpha_true)

    D_boot_per_q = [bootstrap_D_estimate(qs[k], tau_list[k], g2_list[k]) for k in range(n_q)]

    print("  Selecting lambda via lag-holdout …")
    best_lambda, scan_df, _ = select_common_lambda_bo(
        tau_list, g2_list, sig_list, qs, D_boot_per_q,
        lambdas=lambdas, holdout_stride=holdout_stride, n_iter=n_iter_joint)
    print(f"  Selected lambda = {best_lambda:.4g}")

    print("  Refitting joint-BO on the FULL dataset at selected lambda …")
    weights_full = [1.0 / s ** 2 for s in sig_list]
    D_joint, alpha_joint, beta_joint, B_joint, data_sse, penalty = fit_joint_bo_varpro(
        tau_list, qs, g2_list, weights_full, D_boot_per_q, np.full(n_q, 0.9),
        common_lambda=best_lambda, rng_seed=42, n_iter=n_iter_joint)

    print("\n  q_idx        D_joint_BO        alpha_joint")
    for k in range(n_q):
        print(f"  {k + 1:>5d}   {D_joint[k]:>18.4e}   {alpha_joint[k]:>10.4f}")

    err_joint = joint_param_errors(tau_list, qs, g2_list, weights_full,
                                    D_joint, alpha_joint, beta_joint, B_joint, best_lambda)

    print("  Running independent CF per channel (ground-truth check) …")
    D_cf, alpha_cf, beta_cf, B_cf = [], [], [], []
    err_cf_D, err_cf_alpha, err_cf_beta = [], [], []
    D_prior = 10
    for k in range(n_q):
        D_k, a_k, b_k, B_k, sD, sa, sb = fit_independent_curvefit(
            tau_list[k], qs[k], g2_list[k], sig_list[k], D_prior=D_prior)
        D_cf.append(D_k); alpha_cf.append(a_k); beta_cf.append(b_k); B_cf.append(B_k)
        err_cf_D.append(sD); err_cf_alpha.append(sa); err_cf_beta.append(sb)
    D_cf, err_cf_D = np.array(D_cf), np.array(err_cf_D)
    alpha_cf, beta_cf, B_cf = np.array(alpha_cf), np.array(beta_cf), np.array(B_cf)
    err_cf = np.column_stack([err_cf_D, np.array(err_cf_alpha), np.array(err_cf_beta)])

    if make_plots:
        plot_joint_vs_independent_curves(
            tau_list, g2_list, sig_list, qs,
            D_true_arr, beta_true_arr, B_true_arr, alpha_true_arr,
            D_joint, alpha_joint, beta_joint, B_joint,
            D_cf, alpha_cf, beta_cf, B_cf, "simulated")

        if cfg.plot_residuals:
            plot_residuals(tau_list, g2_list, qs, D_joint,
                            alpha_joint, beta_joint, B_joint, D_cf,
                            alpha_cf, beta_cf, B_cf, "simulated")

        plot_D_vs_q_three_way(qs, D_true_arr, D_joint, err_joint[:, 0], D_cf, err_cf_D, "simulated")

        plot_joint_vs_independent_parameters(
            qs, D_true_arr, beta_true_arr, B_true_arr, alpha_true_arr,
            D_joint, alpha_joint, beta_joint, B_joint,
            D_cf, alpha_cf, beta_cf, B_cf, "simulated",
            err_joint=err_joint, err_cf=err_cf)

        if cfg.plot_lambda_scan:
            plot_lambda_scan(scan_df, "simulated")

    return {
        "qs": qs,
        "D_joint": D_joint, "alpha_joint": alpha_joint,
        "beta_joint": beta_joint, "B_joint": B_joint,
        "err_joint": err_joint,
        "selected_lambda": best_lambda, "lambda_scan": scan_df,
        "D_boot_per_q": D_boot_per_q,
        "D_cf": D_cf, "alpha_cf": alpha_cf, "beta_cf": beta_cf, "B_cf": B_cf,
        "err_cf_D": err_cf_D, "err_cf_alpha": np.array(err_cf_alpha), "err_cf_beta": np.array(err_cf_beta),
    }
