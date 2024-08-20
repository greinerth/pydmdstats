from __future__ import annotations

from ._version import version as __version__
from .util import experiment_utils, stats, vorticity
from .util.experiment_utils import (
    comp_checker,
    dmd_stats,
    dmd_stats_global_temp,
    signal,
    signal2d,
    ssim_multi_images,
    std_checker,
)
from .util.stats import runtime_stats
from .util.vorticity import (
    compute_spectral_vorticity_jnp,
    compute_spectral_vorticity_np,
)

__all__ = [
    "__version__",
    "experiment_utils",
    "stats",
    "vorticity",
    "comp_checker",
    "dmd_stats",
    "dmd_stats_global_temp",
    "signal",
    "signal2d",
    "ssim_multi_images",
    "std_checker",
    "runtime_stats",
    "compute_spectral_vorticity_np",
    "compute_spectral_vorticity_jnp",
]
