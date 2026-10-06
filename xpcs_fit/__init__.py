"""XPCS BO/VarPro-vs-curve_fit diffusion-coefficient benchmarking
pipeline (simulated data only).

Public entry points:
    Config                     — ground-truth params + run settings for one simulated run
    run_joint_smooth_analysis  — run the pipeline on one simulated run
"""

from .config import Config
from .pipeline import run_joint_smooth_analysis

__version__ = "0.1.0"

__all__ = [
    "Config",
    "run_joint_smooth_analysis",
]
