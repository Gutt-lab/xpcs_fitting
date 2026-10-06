"""Independent per-channel curve_fit (CF) engine.

Kept architecturally independent from the BO/VarPro engine
(`engines/bayesian.py`): no shared fitting logic, only the common
`PhysicsBounds` and model functions from `xpcs_fit.physics`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ..physics import PhysicsBounds, g2_kww_model


def resolvable_D_max(q_k, tau_min, safety=0.1):
    """Largest D for which the correlation hasn't already fully decayed
    by the first measured lag (tau_min). Above this, exp(-2*D*q^2*tau)
    is ~0 at every data point, so D is unidentifiable and curve_fit's
    result is just wherever the (near-zero) gradient dropped it.

    `safety` keeps the true resolvable edge (tau_c == tau_min) at a
    margin: 0.1 means tau_c must be >= tau_min / 0.1, i.e. 10x inside
    the window rather than barely inside it.
    """
    return 1.0 / (2.0 * q_k ** 2 * tau_min * safety)


def fit_independent_curvefit(tau_k, q_k, g2_k, sig_k, D_prior=10):
    """Fit one q-channel's (tau, g2, sigma) independently with a
    bounded, weighted KWW curve_fit.

    Returns
    -------
    D_k, alpha_k, beta_k, B_k, sig_D_k, sig_alpha_k, sig_beta_k : float
    """
    def model(t, logD, alpha, beta, B):
        return g2_kww_model(t, q_k, 10 ** logD, alpha, beta, B)

    tau_min = float(np.min(tau_k))
    D_max_k = min(PhysicsBounds.D_max, resolvable_D_max(q_k, tau_min))

    lb = [np.log10(PhysicsBounds.D_min), 0.8, PhysicsBounds.beta_min, PhysicsBounds.B_min]
    ub = [np.log10(D_max_k), 1.0, PhysicsBounds.beta_max, PhysicsBounds.B_max]
    p0 = np.clip(
        [np.log10(D_prior), 0.8, 0.5 * (PhysicsBounds.beta_min + PhysicsBounds.beta_max), 1.0],
        lb, ub,
    )

    try:
        popt, pcov = curve_fit(
            model, tau_k, g2_k, p0=p0, bounds=(lb, ub),
            sigma=np.clip(sig_k, 1e-6, None), absolute_sigma=True,
            method="trf", maxfev=20000,
        )
        D_k, alpha_k, beta_k, B_k = 10 ** popt[0], popt[1], popt[2], popt[3]

        perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
        sig_D_k = D_k * np.log(10) * perr[0]      # delta method, fit was in logD
        sig_alpha_k, sig_beta_k = perr[1], perr[2]
    except Exception as e:
        print(f"  [Indep CF Warning] channel q={q_k:.3e} failed: {e}")
        D_k, alpha_k, beta_k, B_k = D_prior, 1.0, np.nan, np.nan
        sig_D_k, sig_alpha_k, sig_beta_k = np.nan, np.nan, np.nan

    return D_k, alpha_k, beta_k, B_k, sig_D_k, sig_alpha_k, sig_beta_k


# NOTE: the original script had a second, commented-out variant of this
# function with alpha fixed at a constant (`alpha_fixed=0.9`) instead of
# fitted. That variant was unreachable dead code and has been dropped
# here rather than carried along as a comment. If you still want an
# alpha-fixed fitting mode, it's a clean candidate for a second function
# in this module (e.g. `fit_independent_curvefit_fixed_alpha`) rather
# than a commented-out duplicate of this one.
#
# Also dropped: an unused `resolved` flag that was computed in the try
# block (set False when the fit pinned against D_max_k) but never
# consumed anywhere in the code you shared. If a caller downstream was
# relying on that "unresolved" signal, it isn't currently returned —
# happy to wire it back in as an explicit return value or a warning.
