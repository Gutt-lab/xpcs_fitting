"""Independent fitting engines.

`curve_fit` (classical, per-channel curve_fit) and `bayesian` (joint
Bayesian Optimization + VarPro) are kept free of shared fitting logic
between them, per the project's independence requirement — they only
both depend on `xpcs_fit.physics` for models and bounds.
"""
