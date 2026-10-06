"""Smoke test: the joint-smooth pipeline runs end-to-end on a small
synthetic dataset and recovers the ground-truth D to within a loose
tolerance. Not a precision test — iteration counts are kept small for
speed, so this only checks that the pipeline runs and is in the right
ballpark.
"""

import numpy as np

from xpcs_fit import Config, run_joint_smooth_analysis


def test_joint_smooth_pipeline_recovers_synthetic_D():
    cfg = Config(
        sim_q_values=np.linspace(0.05, 0.2, 4),
        sim_D_true=20.0,
        sim_beta_true=0.045,
        sim_B_true=1.0,
        sim_alpha_true=1.0,
        sim_noise_level=0.02,
        sim_tau_n_points=40,
        sim_seed=1,
    )

    result = run_joint_smooth_analysis(
        cfg, lambdas=(0.0, 1e-2), holdout_stride=5,
        n_iter_joint=5, n_iter_indep=5, make_plots=False,
    )

    D_joint = result["D_joint"]
    assert D_joint.shape == (4,)
    assert np.all(np.isfinite(D_joint))
    assert np.median(np.abs(D_joint - 20.0) / 20.0) < 0.5
