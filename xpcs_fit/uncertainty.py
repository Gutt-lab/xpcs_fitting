"""Hessian-based (Laplace-approximation) parameter uncertainty for the
joint BO/VarPro fit.

Cov(theta) ~= 2 * H^-1, where H is the Hessian of chi^2 at the fitted
optimum. This holds because chi^2 = -2 * log-likelihood, and the joint
curvature penalty is itself a Gaussian smoothness prior, so
(data_sse + penalty) is -2 * log-posterior up to a constant — the same
objective the BO search minimized.
"""

from __future__ import annotations

import numpy as np

from .engines.bayesian import curvature_operator_q, normalize_unit
from .physics import PhysicsBounds


def _numerical_hessian(f, x0, rel_step=1e-3, min_step=1e-5):
    """Central-difference Hessian of scalar function `f` at `x0`."""
    x0 = np.asarray(x0, float)
    n = len(x0)
    step = np.maximum(min_step, rel_step * np.abs(x0))
    H = np.zeros((n, n))
    f0 = f(x0)
    for i in range(n):
        xp = x0.copy(); xp[i] += step[i]
        xm = x0.copy(); xm[i] -= step[i]
        H[i, i] = (f(xp) - 2 * f0 + f(xm)) / step[i] ** 2
        for j in range(i + 1, n):
            xpp = x0.copy(); xpp[i] += step[i]; xpp[j] += step[j]
            xpm = x0.copy(); xpm[i] += step[i]; xpm[j] -= step[j]
            xmp = x0.copy(); xmp[i] -= step[i]; xmp[j] += step[j]
            xmm = x0.copy(); xmm[i] -= step[i]; xmm[j] -= step[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * step[i] * step[j])
    return H


def _cov_from_chi2(chi2_fn, theta0):
    """Diagonal of the parameter covariance implied by `chi2_fn`'s
    curvature at `theta0`."""
    H = _numerical_hessian(chi2_fn, theta0)
    try:
        Cov = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Cov = 2.0 * np.linalg.pinv(H)
    return np.clip(np.diag(Cov), 0, None)


def _joint_full_chi2(theta, taus, qs, g2_raws, weights_list, common_lambda):
    n_q = len(qs)
    logD, alpha, beta, B = (theta[0:n_q], theta[n_q:2 * n_q],
                             theta[2 * n_q:3 * n_q], theta[3 * n_q:4 * n_q])
    D = 10.0 ** logD
    data_sse = 0.0
    for k in range(n_q):
        tau_k, q_k, g2_k, W_k = taus[k], qs[k], g2_raws[k], weights_list[k]
        if len(tau_k) == 0:
            continue
        pred = B[k] + beta[k] * np.exp(-2.0 * (D[k] * q_k ** 2 * tau_k) ** alpha[k])
        data_sse += float(np.sum(W_k * (g2_k - pred) ** 2))
    if common_lambda <= 0:
        return data_sse
    L = curvature_operator_q(qs)
    log_D_min, log_D_max = np.log10(PhysicsBounds.D_min), np.log10(PhysicsBounds.D_max)
    u_logD = normalize_unit(logD, log_D_min, log_D_max)
    u_alpha = normalize_unit(alpha, PhysicsBounds.alpha_min, PhysicsBounds.alpha_max)
    u_beta = normalize_unit(beta, PhysicsBounds.beta_min, PhysicsBounds.beta_max)
    u_B = normalize_unit(B, PhysicsBounds.B_min, PhysicsBounds.B_max)
    penalty = common_lambda * (np.sum((L @ u_logD) ** 2) + np.sum((L @ u_alpha) ** 2) +
                                np.sum((L @ u_beta) ** 2) + np.sum((L @ u_B) ** 2))
    return data_sse + penalty


def joint_param_errors(tau_list, qs, g2_list, weights_list, D, alpha, beta, B, common_lambda):
    """(sigma_D, sigma_alpha, sigma_beta) per channel from the Hessian
    of the FULL joint posterior at the joint-BO optimum, capturing the
    cross-channel variance reduction from the regularizer.

    Dimensionality is 4*n_q, so this costs a few thousand chi^2 evals
    for n_q ~ 10-15 (a few seconds).
    """
    n_q = len(qs)
    theta0 = np.concatenate([np.log10(D), alpha, beta, B])
    f = lambda th: _joint_full_chi2(th, tau_list, qs, g2_list, weights_list, common_lambda)
    var = _cov_from_chi2(f, theta0)
    sig_D = D * np.log(10) * np.sqrt(var[0:n_q])
    sig_alpha = np.sqrt(var[n_q:2 * n_q])
    sig_beta = np.sqrt(var[2 * n_q:3 * n_q])
    return np.column_stack([sig_D, sig_alpha, sig_beta])


# NOTE: the original script also defined a `_channel_chi2(theta, tau, q,
# g2, W)` helper (single-channel chi^2 under the KWW model) that wasn't
# called anywhere in the code you shared — dropped as dead code. If it
# was meant to drive a per-channel (rather than full-joint) uncertainty
# estimate somewhere downstream, let me know and I'll add it back with
# a caller.
