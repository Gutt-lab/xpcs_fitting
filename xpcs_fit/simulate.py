"""Synthetic XPCS g2 data generator (real-data CSV/NPY loading removed —
this package is simulated-data only)."""

from __future__ import annotations

import numpy as np

from .config import Config


def simulate_xpcs_data(cfg: Config):
    """Generate synthetic g2 data with known ground-truth per-channel
    parameters.

    Noise follows the Schatzel model, scaled per q-channel by an
    absorber-like exponential law (`cfg.sim_noise_level` sets the base
    level, i.e. ~ 1 / mean_intensity).

    Returns
    -------
    tau_list, g2_list, sig_list : list[np.ndarray]
    qs : np.ndarray
    """
    qs = np.asarray(cfg.sim_q_values, dtype=float)
    n_q = len(qs)

    def _broadcast(val, default):
        if val is None:
            return np.full(n_q, default)
        val = np.asarray(val, dtype=float)
        return np.full(n_q, float(val)) if val.ndim == 0 else val

    D_true = _broadcast(cfg.sim_D_true, 10.0)
    beta_true = _broadcast(cfg.sim_beta_true, 0.045)
    B_true = _broadcast(cfg.sim_B_true, 1.0)
    alpha_true = _broadcast(cfg.sim_alpha_true, 1.0)

    print("\n  [simulate_xpcs_data] Ground truth per channel:")
    print(f"  {'q_idx':>5} {'q':>9} {'D_true':>10} {'beta_true':>10} {'B_true':>8} {'alpha_true':>10}")
    for k in range(n_q):
        print(f"  {k + 1:>5} {qs[k]:>9.4f} {D_true[k]:>10.4f} {beta_true[k]:>10.4f} "
              f"{B_true[k]:>8.4f} {alpha_true[k]:>10.4f}")

    tau_list, g2_list, sig_list = [], [], []
    base_noise = cfg.sim_noise_level
    for k in range(n_q):
        noise = base_noise * (1 + k * 0.005)

        tau = np.logspace(-3, 4, cfg.sim_tau_n_points)
        g2_true_curve = B_true[k] + beta_true[k] * np.exp(
            -2.0 * (D_true[k] * qs[k] ** 2 * tau) ** alpha_true[k])

        mean_intensity = 1.0 / noise
        frame_time = tau[0]
        total_exp_time = tau[-1] * 50.0
        total_frames = total_exp_time / frame_time
        delta_tau = np.gradient(tau)
        tier_reduction = frame_time / np.maximum(delta_tau, frame_time)
        n_pairs_tau = np.clip(total_frames * tier_reduction, 1.0, 1e4)

        term1 = 1.0 / (mean_intensity ** 2)
        term2 = 2.0 * (g2_true_curve - 1.0) / mean_intensity
        term3 = (g2_true_curve - 1.0) ** 2
        variance = (term1 + term2 + term3) / (cfg.sim_n_speckles * n_pairs_tau)
        variance = np.clip(variance, a_min=(noise * 1e-6) ** 2, a_max=None)
        sigma = np.sqrt(variance)

        rng = np.random.default_rng(np.random.SeedSequence((cfg.sim_seed, k)))
        g2_noisy = g2_true_curve + rng.normal(0.0, sigma)

        tau_list.append(tau)
        g2_list.append(g2_noisy)
        sig_list.append(sigma)

    return tau_list, g2_list, sig_list, qs
