"""Example: run the joint-smooth BO/VarPro pipeline on synthetic data
with known ground truth.

Usage
-----
    python scripts/run_simulation.py
"""

import numpy as np

from xpcs_fit import Config, run_joint_smooth_analysis


def main():
    cfg = Config(
        sim_q_values=np.logspace(np.log10(0.02), np.log10(0.45), 10),  # nm^-1
        sim_D_true=np.logspace(np.log10(10), np.log10(50), 10),        # nm^2/us
        sim_beta_true=0.045,
        sim_B_true=1.0,
        sim_alpha_true=1.0,   # 1.0 = pure exponential; <1 = KWW stretch
        sim_noise_level=0.05,
        sim_tau_n_points=30,
        sim_seed=0,
    )

    result = run_joint_smooth_analysis(cfg, n_iter_joint=25)
    print(result)


if __name__ == "__main__":
    main()
