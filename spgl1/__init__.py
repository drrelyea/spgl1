"""SPGL1: Spectral Projected Gradient for L1 minimization."""

from .lsqr import lsqr
from .spgl1 import (
    oneprojector,
    norm_l1nn_primal,
    norm_l1nn_dual,
    norm_l1nn_project,
    norm_l12nn_primal,
    norm_l12nn_dual,
    norm_l12nn_project,
    spgl1,
    spg_bp,
    spg_bpdn,
    spg_lasso,
    spg_mmv,
    spg_group,
)

__all__: list[str] = [
    "oneprojector",
    "norm_l1nn_primal",
    "norm_l1nn_dual",
    "norm_l1nn_project",
    "norm_l12nn_primal",
    "norm_l12nn_dual",
    "norm_l12nn_project",
    "spgl1",
    "spg_bp",
    "spg_bpdn",
    "spg_lasso",
    "spg_mmv",
    "spg_group",
]


try:
    from .version import version as __version__
except ImportError:
    __version__: str = "0.0.0"
