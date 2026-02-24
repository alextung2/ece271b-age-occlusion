from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class PcaModel:
    pca: PCA


@dataclass
class LdaModel:
    lda: LinearDiscriminantAnalysis
    classes_: np.ndarray  # store class labels seen during fit (for sanity/debug)


# ---------------- PCA ---------------- #

def fit_pca(X: np.ndarray, n_components: int, *, whiten: bool = True) -> PcaModel:
    """
    Fit PCA on X.

    X: (N, D)
    n_components: desired number of components
    whiten: whether to whiten (recommended for Gaussian classifier)
    """
    assert isinstance(X, np.ndarray) and X.ndim == 2, "X must be (N,D)"
    N, D = X.shape
    assert N >= 2, "Need N >= 2 samples to fit PCA"
    assert n_components >= 1, "n_components must be >= 1"

    # Safety cap (important when D or N is small)
    n_eff = min(n_components, N - 1, D)
    assert n_eff >= 1, f"Effective n_components became {n_eff}; check N, D, n_components"

    pca = PCA(
        n_components=n_eff,
        svd_solver="randomized",
        whiten=whiten,
        random_state=0,
    )
    pca.fit(X)
    return PcaModel(pca=pca)


def transform_pca(model: PcaModel, X: np.ndarray) -> np.ndarray:
    assert isinstance(X, np.ndarray) and X.ndim == 2
    return model.pca.transform(X)


# ---------------- LDA ---------------- #

def fit_lda(X: np.ndarray, y: np.ndarray) -> LdaModel:
    """
    Fit LDA projection.

    sklearn chooses output dimensionality automatically, up to <= (C-1),
    where C is the number of unique classes in y.
    """
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 1
    assert X.shape[0] == y.shape[0]

    # Basic hygiene
    assert np.isfinite(X).all(), "X contains NaN/inf"
    assert np.isfinite(y).all(), "y contains NaN/inf"

    classes = np.unique(y)
    assert classes.size >= 2, f"Need at least 2 classes for LDA, got {classes.size}: {classes.tolist()}"

    # Optional: enforce integer labels (sklearn can handle strings too, but your pipeline assumes ints)
    if not np.issubdtype(y.dtype, np.integer):
        # allow float labels that are actually integers
        y_int = y.astype(np.int64)
        assert np.all(y_int == y), "y must be integer-valued labels"
        y = y_int

    lda = LinearDiscriminantAnalysis(
        solver="svd",
    )
    lda.fit(X, y)

    return LdaModel(lda=lda, classes_=classes)


def transform_lda(model: LdaModel, X: np.ndarray) -> np.ndarray:
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert np.isfinite(X).all(), "X contains NaN/inf"
    return model.lda.transform(X)