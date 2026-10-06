"""Joint Bayesian-Optimization + VarPro (BO/VarPro) engine.

Kept architecturally independent from the CF engine
(`engines/curve_fit.py`): no shared fitting logic, only the common
`PhysicsBounds` from `xpcs_fit.physics`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from ..physics import PhysicsBounds


def curvature_operator_q(qs):
    """Nonuniform second-derivative operator on q, normalized to [0, 1].
    `L @ x ~ 0` for a smooth (low-curvature) x(q); has (n_q - 2) rows.
    """
    q = np.asarray(qs, float)
    s = (q - q.min()) / (q.max() - q.min())
    n = len(q)
    L = np.zeros((n - 2, n))
    for row, i in enumerate(range(1, n - 1)):
        h_minus = s[i] - s[i - 1]
        h_plus = s[i + 1] - s[i]
        L[row, i - 1] = 2.0 / (h_minus * (h_minus + h_plus))
        L[row, i] = -2.0 / (h_minus * h_plus)
        L[row, i + 1] = 2.0 / (h_plus * (h_minus + h_plus))
    return L


def normalize_unit(x, lo, hi):
    """Rescale `x` from [lo, hi] to [0, 1]. Public (not module-private)
    because `xpcs_fit.uncertainty` reuses it for the same joint
    posterior the BO engine optimizes.
    """
    return (np.asarray(x, float) - lo) / (hi - lo)


def joint_ridge_beta_B(D_row, alpha_row, taus, qs, g2_raws, weights_list, common_lambda):
    """Closed-form joint solve for beta(q), B(q) given fixed D(q), alpha(q).

    This is the piece that actually couples channels for the linear
    parameters: a single (n_data_pts + reg_rows) x 2*n_q least-squares
    system, not n_q independent VarPro solves.
    """
    n_q = len(qs)
    L = curvature_operator_q(qs)

    rows_data = sum(len(t) for t in taus)
    rows_reg = 2 * L.shape[0] if common_lambda > 0 else 0
    A = np.zeros((rows_data + rows_reg, 2 * n_q))
    b = np.zeros(rows_data + rows_reg)

    r0 = 0
    for k in range(n_q):
        tau_k, q_k, g2_k, W_k = taus[k], qs[k], g2_raws[k], weights_list[k]
        n_k = len(tau_k)
        if n_k == 0:
            continue
        X_k = np.exp(-2.0 * (D_row[k] * q_k ** 2 * tau_k) ** alpha_row[k])
        sw = np.sqrt(W_k)
        rows = slice(r0, r0 + n_k)
        A[rows, k] = sw * X_k            # coefficient on beta_k
        A[rows, n_q + k] = sw * 1.0      # coefficient on B_k
        b[rows] = sw * g2_k
        r0 += n_k

    if rows_reg:
        scale_beta = np.sqrt(common_lambda) / (PhysicsBounds.beta_max - PhysicsBounds.beta_min)
        scale_B = np.sqrt(common_lambda) / (PhysicsBounds.B_max - PhysicsBounds.B_min)
        rr = rows_data
        A[rr:rr + L.shape[0], 0:n_q] = scale_beta * L
        rr2 = rr + L.shape[0]
        A[rr2:rr2 + L.shape[0], n_q:2 * n_q] = scale_B * L
        # b stays 0 on regularization rows (penalizing curvature toward 0)

    y, *_ = np.linalg.lstsq(A, b, rcond=None)
    beta = np.clip(y[:n_q], PhysicsBounds.beta_min, PhysicsBounds.beta_max)
    B = np.clip(y[n_q:], PhysicsBounds.B_min, PhysicsBounds.B_max)
    return beta, B


def joint_objective(theta, taus, qs, g2_raws, weights_list, common_lambda):
    """theta = concat[logD_1..N, alpha_1..N].

    Returns (total, data_sse, penalty, beta, B).
    """
    n_q = len(qs)
    logD, alpha = theta[:n_q], theta[n_q:]
    D_row = 10.0 ** logD
    beta, B = joint_ridge_beta_B(D_row, alpha, taus, qs, g2_raws, weights_list, common_lambda)

    data_sse = 0.0
    for k in range(n_q):
        tau_k, q_k, g2_k, W_k = taus[k], qs[k], g2_raws[k], weights_list[k]
        if len(tau_k) == 0:
            continue
        pred = B[k] + beta[k] * np.exp(-2.0 * (D_row[k] * q_k ** 2 * tau_k) ** alpha[k])
        data_sse += float(np.sum(W_k * (g2_k - pred) ** 2))

    if common_lambda <= 0:
        return data_sse, data_sse, 0.0, beta, B

    L = curvature_operator_q(qs)
    log_D_min, log_D_max = np.log10(PhysicsBounds.D_min), np.log10(PhysicsBounds.D_max)
    u_logD = normalize_unit(logD, log_D_min, log_D_max)
    u_alpha = normalize_unit(alpha, PhysicsBounds.alpha_min, PhysicsBounds.alpha_max)
    u_beta = normalize_unit(beta, PhysicsBounds.beta_min, PhysicsBounds.beta_max)
    u_B = normalize_unit(B, PhysicsBounds.B_min, PhysicsBounds.B_max)

    penalty = common_lambda * (
        np.sum((L @ u_logD) ** 2) + np.sum((L @ u_alpha) ** 2)
        + np.sum((L @ u_beta) ** 2) + np.sum((L @ u_B) ** 2)
    )
    return data_sse + penalty, data_sse, penalty, beta, B


def fit_joint_bo_varpro(taus, qs, g2_raws, weights_list, D_prior_list, alpha_prior_list=None,
                         common_lambda=0.0, rng_seed=None, n_iter=40):
    """Joint Bayesian optimization over (logD, alpha) across all
    q-channels at once, with beta/B solved in closed form (VarPro) at
    every candidate point via `joint_ridge_beta_B`.

    Returns (D_row, alpha_row, beta, B, data_sse, penalty) at the best
    point found, refined with a short local L-BFGS-B polish.
    """
    n_q = len(qs)
    rng = np.random.default_rng(rng_seed)
    safe_seed_int = int(rng.integers(0, 1_000_000))

    log_D_min, log_D_max = np.log10(PhysicsBounds.D_min), np.log10(PhysicsBounds.D_max)
    a_min, a_max = PhysicsBounds.alpha_min, PhysicsBounds.alpha_max
    lb = np.concatenate([np.full(n_q, log_D_min), np.full(n_q, a_min)])
    ub = np.concatenate([np.full(n_q, log_D_max), np.full(n_q, a_max)])
    d = 2 * n_q

    def obj(theta):
        total, *_ = joint_objective(theta, taus, qs, g2_raws, weights_list, common_lambda)
        return np.log(max(total, 1e-20))

    logD_prior = np.log10(np.clip(D_prior_list, PhysicsBounds.D_min, PhysicsBounds.D_max))
    alpha_prior = (np.full(n_q, 0.9) if alpha_prior_list is None
                   else np.clip(alpha_prior_list, a_min, a_max))
    theta_prior = np.concatenate([logD_prior, alpha_prior])

    sampler = LatinHypercube(d=d, scramble=True, seed=safe_seed_int)
    n_init = max(3 * d, 20)
    X_init = lb + sampler.random(n=n_init) * (ub - lb)
    X_init = np.vstack([theta_prior, X_init])
    Y_sample = np.array([obj(x) for x in X_init])
    X_sample = X_init.copy()

    cand_sampler = LatinHypercube(d=d, scramble=True, seed=safe_seed_int + 1)
    n_cand = max(200 * d, 3000)
    X_cand_base = lb + cand_sampler.random(n=n_cand) * (ub - lb)

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 2.0), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                   normalize_y=True, random_state=safe_seed_int)
    best_so_far = float(np.min(Y_sample))

    for _ in range(n_iter):
        top_idx = np.argsort(Y_sample)[:min(15, len(Y_sample))]
        perturb = X_sample[top_idx] + rng.normal(0, 0.05 * (ub - lb), (len(top_idx), d))
        X_cand = np.vstack([X_cand_base, np.clip(perturb, lb, ub)])

        gp.fit(X_sample, Y_sample)
        mu, std = gp.predict(X_cand, return_std=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            improvement = best_so_far - mu
            Z = np.zeros_like(mu)
            vmask = std > 1e-9
            Z[vmask] = improvement[vmask] / std[vmask]
            ei = np.zeros_like(mu)
            ei[vmask] = improvement[vmask] * norm.cdf(Z[vmask]) + std[vmask] * norm.pdf(Z[vmask])

        next_X = X_cand[np.argmax(ei)]
        new_y = obj(next_X)
        X_sample = np.vstack([X_sample, next_X])
        Y_sample = np.append(Y_sample, new_y)
        if new_y < best_so_far:
            best_so_far = new_y

    order = np.argsort(Y_sample)
    best_res = None
    for k in order[:4]:
        r = minimize(obj, X_sample[k], bounds=list(zip(lb, ub)), method="L-BFGS-B",
                      options={"ftol": 1e-14, "gtol": 1e-10})
        if best_res is None or r.fun < best_res.fun:
            best_res = r

    theta_star = best_res.x
    total, data_sse, penalty, beta, B = joint_objective(
        theta_star, taus, qs, g2_raws, weights_list, common_lambda)
    D_row = 10.0 ** theta_star[:n_q]
    alpha_row = theta_star[n_q:]
    return D_row, alpha_row, beta, B, data_sse, penalty


def select_common_lambda_bo(tau_list, g2_list_corr, sig_list, qs, D_boot_per_q,
                             lambdas=(0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                             holdout_stride=5, holdout_offset=2, n_iter=30, rng_seed=42):
    """Truth-blind lambda selection for real single-acquisition data.

    Holds out every `holdout_stride`-th real tau point per channel,
    fits joint-BO on the remainder for each candidate lambda, and
    scores on the held-out points. Mirrors the lag-holdout fallback
    used when no independent repeat measurement is available.

    Returns (best_lambda, scan_dataframe, all_rows).
    """
    n_q = len(qs)
    fit_taus, fit_g2, fit_sig, fit_w = [], [], [], []
    val_taus, val_g2 = [], []

    for k in range(n_q):
        tau_k, g2_k, sig_k = tau_list[k], g2_list_corr[k], sig_list[k]
        if len(tau_k) < 3 * holdout_stride:
            raise ValueError(f"channel q-index {k} has too few points ({len(tau_k)}) "
                              f"for a stride-{holdout_stride} holdout")
        held = (np.arange(len(tau_k)) % holdout_stride) == holdout_offset
        fit_taus.append(tau_k[~held]); fit_g2.append(g2_k[~held]); fit_sig.append(sig_k[~held])
        fit_w.append(1.0 / fit_sig[-1] ** 2)
        val_taus.append(tau_k[held]); val_g2.append(g2_k[held])

    rows = []
    theta_start_D = np.array(D_boot_per_q, float)
    theta_start_alpha = np.full(n_q, 0.9)
    for lam in lambdas:
        D_row, alpha_row, beta, B, data_sse, penalty = fit_joint_bo_varpro(
            fit_taus, qs, fit_g2, fit_w, theta_start_D, theta_start_alpha,
            common_lambda=float(lam), rng_seed=rng_seed, n_iter=n_iter)
        theta_start_D, theta_start_alpha = D_row, alpha_row  # warm-start next lambda

        val_sse, val_n = 0.0, 0
        for k in range(n_q):
            if len(val_taus[k]) == 0:
                continue
            pred = B[k] + beta[k] * np.exp(-2.0 * (D_row[k] * qs[k] ** 2 * val_taus[k]) ** alpha_row[k])
            val_sse += float(np.sum((val_g2[k] - pred) ** 2))
            val_n += len(val_taus[k])
        val_rmse = np.sqrt(val_sse / val_n) if val_n else np.nan

        rows.append({
            "lambda": float(lam), "val_RMSE": val_rmse,
            "training_data_sse": data_sse, "penalty": penalty,
            "D_row": D_row.copy(), "alpha_row": alpha_row.copy(),
            "beta": beta.copy(), "B": B.copy(),
        })
        print(f"    lambda={lam:.3g}: val_RMSE={val_rmse:.5e}  "
              f"data_sse={data_sse:.4e}  penalty={penalty:.4e}")

    scan_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("D_row", "alpha_row", "beta", "B")}
        for r in rows
    ])
    best = min(rows, key=lambda r: r["val_RMSE"])
    return best["lambda"], scan_df, rows
