# xpcs-fit

Benchmarks joint Bayesian-Optimization + VarPro fitting against
classical, independent per-channel `curve_fit`, for extracting
diffusion coefficients D(q) from XPCS g2(q, τ) correlation functions
(KWW/stretched-exponential model). Simulated data only — real-data
loading was removed.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from xpcs_fit import Config, run_joint_smooth_analysis

cfg = Config(
    sim_q_values=np.logspace(np.log10(0.02), np.log10(0.45), 10),
    sim_D_true=np.logspace(np.log10(10), np.log10(50), 10),
    sim_beta_true=0.045,
    sim_B_true=1.0,
    sim_alpha_true=1.0,
)
result = run_joint_smooth_analysis(cfg)
```

See `scripts/run_simulation.py` for a runnable example.

## Layout

```
xpcs_fit/
    config.py         Config dataclass — ground-truth params + run settings
    physics.py         g2/g1 models, PhysicsBounds — shared by both engines
    simulate.py         Synthetic g2 data generator
    preprocessing.py   Smoothing, dense-grid construction, bootstrap D prior
    engines/
        curve_fit.py    Independent per-channel curve_fit (CF) engine
        bayesian.py     Joint Bayesian-Optimization + VarPro (BO) engine
    uncertainty.py      Hessian-based parameter uncertainty for the joint fit
    plotting.py         Diagnostic plots
    pipeline.py         run_joint_smooth_analysis — orchestration
scripts/
    run_simulation.py   Runnable example
tests/
    test_pipeline.py    End-to-end smoke test
```

The CF and BO engines are kept deliberately independent: they only
share `physics.py` (models and bounds), never fitting logic. This
mirrors the project's own requirement that the two be directly,
fairly comparable rather than accidentally coupled.

## License

MIT (adjust as needed before publishing).
