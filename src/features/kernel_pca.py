from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from sklearn.decomposition import KernelPCA


@dataclass
class KPcaModel:
    kpca: KernelPCA
    meta: Dict[str, Any]  # stash for notes / chosen hyperparams


# =============================================================================
# EDIT HERE (Luke Wittemann): This is the ONLY place you should change code in this file.
#
# Goal:
#   Let the training script call fit_kpca(...) with (kernel, gamma) and get a
#   working KernelPCA model. If you want to experiment with different kernels,
#   you ONLY need to edit the return values below.
#
# How to use:
#   - Start with kernel="rbf" (recommended).
#   - If you try kernel="poly" or "sigmoid", you can set degree/coef0 here.
#   - The training script will still pass kernel and gamma into fit_kpca(...).
#     This config just supplies extra kernel-specific params.
#
# Safe workflow:
#   1) Keep defaults (rbf, no extra params).
#   2) If experimenting, change ONE thing at a time (e.g., try poly degree=3).
# =============================================================================
def kpca_extra_params(kernel: str) -> Dict[str, Any]:
    """
    Return extra KernelPCA kwargs based on the kernel.

    IMPORTANT:
      - Do NOT remove keys from the dict in fit_kpca; only add extras here.
      - If you don't know what to do, return {} and leave kernel="rbf".
    """
    k = str(kernel).lower()

    # ---- baseline: RBF needs no extras beyond gamma ----
    if k == "rbf":
        return {}

    # ---- optional: polynomial kernel ----
    # Common defaults: degree=3, coef0=1.0
    if k == "poly":
        return {
            "degree": 3,   # TODO: try 2, 3, 4
            "coef0": 1.0,  # TODO: try 0.0, 1.0
        }

    # ---- optional: sigmoid kernel ----
    if k == "sigmoid":
        return {
            "coef0": 0.0,  # TODO: try 0.0, 1.0
        }

    # ---- cosine kernel usually needs nothing extra ----
    if k == "cosine":
        return {}

    # Unknown kernel: do nothing extra
    return {}


def fit_kpca(
    X: np.ndarray,
    n_components: int,
    kernel: str,
    gamma: float | None = None,
    random_state: int | None = 271,
) -> KPcaModel:
    """
    Fit Kernel PCA on X.

    Notes for partner:
      - Scaling/standardization should happen in the TRAINING SCRIPT (recommended).
      - This function should remain a thin wrapper around sklearn's KernelPCA.
      - If you want to support poly/sigmoid knobs, edit kpca_extra_params().
    """
    assert X.ndim == 2, "X must be (N,D)"
    assert n_components >= 1, "n_components must be >= 1"

    extra = kpca_extra_params(kernel)

    kpca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=False,
        random_state=random_state,
        **extra,
    )
    kpca.fit(X)

    meta = {
        "n_components": int(n_components),
        "kernel": str(kernel),
        "gamma": None if gamma is None else float(gamma),
        "extra_params": dict(extra),
    }
    return KPcaModel(kpca=kpca, meta=meta)


def transform_kpca(model: KPcaModel, X: np.ndarray) -> np.ndarray:
    """
    Transform X using a fitted KPCA model.

    IMPORTANT:
      If you applied preprocessing before fit (e.g., StandardScaler),
      apply the SAME preprocessing before calling transform_kpca in the script.
    """
    assert X.ndim == 2, "X must be (N,D)"
    return model.kpca.transform(X)


def suggest_gamma_rbf(X: np.ndarray) -> float:
    """
    Simple heuristic gamma suggestion for RBF kernels.

    This one uses: gamma = 1 / D
    (basic, safe baseline; tuning around it in the training script is recommended)
    """
    assert X.ndim == 2
    _, D = X.shape
    return 1.0 / max(D, 1)