"""Physical models and parameter bounds for XPCS g2/g1 fitting.

Shared by both fitting engines (`engines/curve_fit.py`,
`engines/bayesian.py`) — this is the one piece of logic they're
allowed to have in common; the fitting strategies themselves stay
independent.
"""

from __future__ import annotations

import numpy as np


class PhysicsBounds:
    """Hard physical bounds enforced on every fitted parameter.

    Units: D in nm^2/us, beta and B dimensionless, alpha is the KWW
    stretching exponent (1.0 = pure exponential).
    """

    D_min, D_max = 1e-2, 1e4
    beta_min, beta_max = 0.001, 0.08
    B_min, B_max = 0.0, 1.20
    alpha_min, alpha_max = 0.8, 1.0

    @classmethod
    def validate(cls, param_name: str, value: float) -> None:
        """Raise ValueError if `value` falls outside the bound for `param_name`."""
        lo, hi = {
            "D": (cls.D_min, cls.D_max),
            "beta": (cls.beta_min, cls.beta_max),
            "B": (cls.B_min, cls.B_max),
        }[param_name]
        if not (lo <= value <= hi):
            raise ValueError(f"{param_name}={value:.2e} out of [{lo:.2e}, {hi:.2e}]")


def g2_model(tau, q, D, beta, B):
    """Standard single-exponential g2(tau) model."""
    return B + beta * np.exp(-2.0 * D * q ** 2 * tau)


def g2_kww_model(tau, q, D, alpha, beta, B):
    """Kohlrausch-Williams-Watts (stretched exponential) g2(tau) model."""
    return B + beta * np.exp(-2.0 * (D * q ** 2 * tau) ** alpha)


def g1_model(tau, q, D):
    """Standard single-exponential field autocorrelation g1(tau)."""
    return np.exp(-D * q ** 2 * tau)


def g1_kww_model(tau, q, D, alpha):
    """KWW field autocorrelation g1(tau)."""
    return np.exp(-(D * q ** 2 * tau) ** alpha)


def g2_to_g1(g2, sig, beta, B):
    """Siegert-style conversion of a measured g2 (+ sigma) to g1.

        g1       = sqrt((g2 - B) / beta)
        sigma_g1 = sigma_g2 / (2 * beta * g1)      [linear error propagation]

    (g2 - B) is clipped at 0 before the sqrt to guard against noise
    pushing baseline-subtracted points slightly negative. g1 is
    clipped away from 0 before dividing, to avoid a divide-by-zero
    blowup in the propagated sigma near the tail where g1 -> 0; the
    resulting sigma_g1 is additionally capped at 1.0 (a physically
    meaningless propagated error beyond that, given g1 itself is
    bounded in [0, 1]) purely so a handful of near-zero-g1 tail points
    don't blow out a plot's autoscaled y-axis.
    """
    ratio = np.clip((g2 - B) / beta, 0.0, None)
    g1 = np.sqrt(ratio)
    g1_safe = np.clip(g1, 1e-6, None)
    sig_g1 = np.clip(sig / (2.0 * beta * g1_safe), 0.0, 1.0)
    return g1, sig_g1


def compute_channel_residuals(tau_k, g2_k, q_k, D, alpha_k, beta_k, B_k):
    """Data-minus-model residuals for one q-channel under the KWW model."""
    pred = g2_kww_model(tau_k, q_k, D, alpha_k, beta_k, B_k)
    return g2_k - pred
