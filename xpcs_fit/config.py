"""Configuration for the XPCS BO-vs-CF benchmarking pipeline.

Simulated-data only — see xpcs_fit.simulate.simulate_xpcs_data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Config:
    """Ground-truth parameters and run settings for one simulated run.

    Parameters
    ----------
    n_iter_bo
        Default number of Bayesian-optimization iterations for the
        independent-CF ground-truth check.
    n_dense
        Size of the dense internal tau grid used only for fitting.
    smooth_before_dense
        Apply light Savitzky-Golay smoothing to real g2 before PCHIP
        interpolation onto the dense grid.
    sim_q_values, sim_D_true, sim_beta_true, sim_B_true, sim_alpha_true
        Ground-truth per-channel parameters `simulate_xpcs_data` draws
        synthetic g2 curves from. Scalars broadcast to every channel.
    sim_noise_level
        Base Schatzel-model noise level (~ 1 / mean_intensity).
    sim_tau_n_points
        Number of tau points per simulated channel.
    sim_n_speckles
        Number of speckles averaged per simulated point.
    sim_seed
        RNG seed for reproducible synthetic noise.
    make_diagnostic_plots
        Master switch for the joint-vs-independent curve/parameter
        plots produced by `run_joint_smooth_analysis`.
    plot_residuals, plot_lambda_scan
        Optional extra diagnostic plots, off by default.
    """

    n_iter_bo: int = 30

    n_dense: int = 3000
    smooth_before_dense: bool = True

    sim_q_values: Optional[np.ndarray] = None    # nm^-1
    sim_D_true: Optional[np.ndarray] = None       # nm^2/us
    sim_beta_true: Optional[np.ndarray] = None
    sim_B_true: Optional[np.ndarray] = None
    sim_alpha_true: Optional[np.ndarray] = None   # KWW stretching exponent
    sim_noise_level: float = 0.20                 # ~ 1 / mean_intensity
    sim_tau_n_points: int = 120
    sim_n_speckles: int = 1
    sim_seed: int = 0

    # Diagnostic-plot toggles
    make_diagnostic_plots: bool = True
    plot_residuals: bool = False
    plot_lambda_scan: bool = False
