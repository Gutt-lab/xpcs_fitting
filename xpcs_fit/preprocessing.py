"""Smoothing, dense-grid construction, and bootstrap D estimation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from .physics import PhysicsBounds


def _light_smooth(g2_sorted, n_real):
    """Light Savitzky-Golay smoothing of real g2, skipped for short channels."""
    if n_real < 7:
        return g2_sorted
    win = 5 if n_real < 25 else 7
    if win >= n_real:
        win = n_real - 2 if (n_real - 2) % 2 == 1 else n_real - 3
    if win < 5:
        return g2_sorted
    return savgol_filter(g2_sorted, window_length=win, polyorder=2)


def build_dense_channel(tau_real, g2_real, sig_real, n_dense=3000, smooth=True):
    """Build a dense "imaginary" tau grid over [tau_min, tau_max] via
    PCHIP interpolation of the (optionally smoothed) real g2/sigma, for
    internal fitting use only.
    """
    log_tau_real = np.log10(tau_real)
    order = np.argsort(log_tau_real)
    log_tau_s, g2_s, sig_s = log_tau_real[order], g2_real[order], sig_real[order]

    n_real = len(tau_real)
    g2_for_interp = _light_smooth(g2_s, n_real) if smooth else g2_s

    log_tau_dense = np.linspace(log_tau_s[0], log_tau_s[-1], n_dense)
    tau_dense = 10 ** log_tau_dense

    g2_interp = PchipInterpolator(log_tau_s, g2_for_interp)
    sig_interp = PchipInterpolator(log_tau_s, sig_s)

    g2_dense = g2_interp(log_tau_dense)
    sig_dense = np.clip(sig_interp(log_tau_dense), 1e-6, None)

    w_dense = 1.0 / (sig_dense ** 2)

    return tau_dense, g2_dense, w_dense


def bootstrap_D_estimate(q, tau, g2):
    """Coarse-to-fine grid search for a diffusion-coefficient prior,
    using real (tau, g2) only. Solves beta/B analytically (closed-form
    WLS) at each candidate D and scores by median absolute residual.
    """
    def estimate_B_beta(D_cand):
        X = np.exp(-2.0 * D_cand * q ** 2 * tau)
        X_bar = np.mean(X)
        Y_bar = np.mean(g2)
        num = np.sum((X - X_bar) * (g2 - Y_bar))
        den = np.sum((X - X_bar) ** 2)
        beta = np.clip(num / den if den > 1e-12 else 0.5,
                        PhysicsBounds.beta_min, PhysicsBounds.beta_max)
        B = np.clip(Y_bar - beta * X_bar, PhysicsBounds.B_min, PhysicsBounds.B_max)
        return B, beta

    def mar_at_D(D_cand):
        B_c, beta_c = estimate_B_beta(D_cand)
        pred = B_c + beta_c * np.exp(-2.0 * D_cand * q ** 2 * tau)
        return np.median(np.abs(g2 - pred))

    log_D_grid = np.linspace(np.log10(PhysicsBounds.D_min), np.log10(PhysicsBounds.D_max), 120)
    best_mar, best_D = np.inf, 10 ** np.mean(log_D_grid)
    for ld in log_D_grid:
        D_cand = 10 ** ld
        mar = mar_at_D(D_cand)
        if mar < best_mar:
            best_mar, best_D = mar, D_cand

    log_fine = np.linspace(
        np.clip(np.log10(best_D) - 1.5, np.log10(PhysicsBounds.D_min), np.log10(PhysicsBounds.D_max)),
        np.clip(np.log10(best_D) + 1.5, np.log10(PhysicsBounds.D_min), np.log10(PhysicsBounds.D_max)),
        200,
    )
    for ld in log_fine:
        D_cand = 10 ** ld
        mar = mar_at_D(D_cand)
        if mar < best_mar:
            best_mar, best_D = mar, D_cand

    return float(np.clip(best_D, PhysicsBounds.D_min, PhysicsBounds.D_max))
