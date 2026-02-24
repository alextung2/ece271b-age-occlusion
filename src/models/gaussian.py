from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GaussianClassifier:
    """
    Class-conditional Gaussian classifier.

    means:   (K, d)
    cov_inv: (K, d, d)  (if shared_cov=True, all K entries are identical)
    log_det: (K,)       (if shared_cov=True, all K entries are identical)
    priors:  (K,)
    """
    means: np.ndarray
    cov_inv: np.ndarray
    log_det: np.ndarray
    priors: np.ndarray


def _stable_cov(Zk: np.ndarray, reg: float) -> np.ndarray:
    """
    Return a full covariance estimate for Zk with diagonal regularization.
    Uses a stable formula; handles tiny sample sizes gracefully.
    """
    n, d = Zk.shape
    if n <= 1:
        # Too few samples: fall back to pure regularization
        return reg * np.eye(d, dtype=np.float64)

    # Centered covariance (bias=True => divide by n, not n-1)
    mu = Zk.mean(axis=0, keepdims=True)
    X = (Zk - mu).astype(np.float64)
    C = (X.T @ X) / float(n)  # (d,d)
    C = C + reg * np.eye(d, dtype=np.float64)
    return C


def _cov_to_diag(C: np.ndarray, reg: float) -> np.ndarray:
    """
    Convert full covariance C to diagonal covariance with extra reg.
    """
    d = C.shape[0]
    diag = np.clip(np.diag(C), 0.0, None) + reg
    return np.diag(diag.astype(np.float64))


def _assert_contiguous_labels(y: np.ndarray) -> int:
    """
    Enforce the common contract used by the rest of the codebase:
      y must contain integer class ids in {0,1,...,K-1} with no gaps.

    Returns:
      K
    """
    classes = np.unique(y)
    if classes.size == 0:
        raise ValueError("Empty label array y.")
    if classes.min() != 0 or classes.max() != classes.size - 1:
        raise ValueError(
            f"Labels must be contiguous 0..K-1. Got classes={classes.tolist()}."
        )
    return int(classes.size)


def fit_gaussian_classifier(
    Z: np.ndarray,
    y: np.ndarray,
    reg: float = 1e-3,
    *,
    shared_cov: bool = False,
    diagonal: bool = False,
) -> GaussianClassifier:
    """
    Fit a class-conditional Gaussian classifier.

    Z: (N, d) features
    y: (N,) integer labels in [0..K-1] (contiguous)

    reg:        diagonal regularization strength
    shared_cov: if True, uses a single covariance shared across classes (LDA-style)
    diagonal:   if True, uses diagonal covariance (more stable in high-d)
    """
    assert isinstance(Z, np.ndarray) and Z.ndim == 2, "Z must be (N,d) ndarray"
    assert isinstance(y, np.ndarray) and y.ndim == 1, "y must be (N,) ndarray"
    assert Z.shape[0] == y.shape[0], "Z and y must have same N"
    assert reg > 0, "reg must be > 0"
    assert np.isfinite(Z).all(), "Z contains NaN/inf"
    assert np.isfinite(y).all(), "y contains NaN/inf"

    y = y.astype(np.int64, copy=False)
    K = _assert_contiguous_labels(y)

    N, d = Z.shape

    means = np.zeros((K, d), dtype=np.float64)
    priors = np.zeros((K,), dtype=np.float64)

    # First pass: compute means and priors
    for k in range(K):
        idx = (y == k)
        nk = int(idx.sum())
        if nk <= 0:
            raise ValueError(f"class {k} has 0 samples (unexpected with contiguous labels)")
        Zk = Z[idx]
        means[k] = Zk.mean(axis=0)
        priors[k] = nk / float(N)

    # Covariance estimation
    if shared_cov:
        # Pooled covariance (weighted by class counts)
        C_pool = np.zeros((d, d), dtype=np.float64)
        for k in range(K):
            idx = (y == k)
            Zk = Z[idx]
            Ck = _stable_cov(Zk, reg=0.0)  # add reg later once pooled
            C_pool += (Zk.shape[0] / float(N)) * Ck

        # Add regularization after pooling
        C_pool = C_pool + reg * np.eye(d, dtype=np.float64)
        if diagonal:
            C_pool = _cov_to_diag(C_pool, reg=0.0)

        # Invert + logdet once, then replicate K times
        Ci = np.linalg.inv(C_pool)
        sign, ld = np.linalg.slogdet(C_pool)
        if sign <= 0:
            # Should be PD after reg, but just in case:
            C_pool = C_pool + (10.0 * reg) * np.eye(d, dtype=np.float64)
            Ci = np.linalg.inv(C_pool)
            sign, ld = np.linalg.slogdet(C_pool)
            if sign <= 0:
                raise np.linalg.LinAlgError("Shared covariance still not positive-definite after extra regularization.")

        cov_inv = np.repeat(Ci[None, :, :], K, axis=0)
        log_det = np.repeat(np.array([ld], dtype=np.float64), K, axis=0)

    else:
        cov_inv = np.zeros((K, d, d), dtype=np.float64)
        log_det = np.zeros((K,), dtype=np.float64)

        for k in range(K):
            idx = (y == k)
            Zk = Z[idx]

            C = _stable_cov(Zk, reg=reg)
            if diagonal:
                C = _cov_to_diag(C, reg=0.0)

            # Invert + logdet (robustify if needed)
            try:
                Ci = np.linalg.inv(C)
                sign, ld = np.linalg.slogdet(C)
                if sign <= 0:
                    raise np.linalg.LinAlgError("non-positive definite cov")
            except np.linalg.LinAlgError:
                # Strengthen reg and retry once
                C = C + (10.0 * reg) * np.eye(d, dtype=np.float64)
                Ci = np.linalg.inv(C)
                sign, ld = np.linalg.slogdet(C)
                if sign <= 0:
                    raise np.linalg.LinAlgError("Covariance not PD even after extra regularization.")

            cov_inv[k] = Ci
            log_det[k] = ld

    return GaussianClassifier(means=means, cov_inv=cov_inv, log_det=log_det, priors=priors)


def predict_gaussian(clf: GaussianClassifier, Z: np.ndarray) -> np.ndarray:
    """
    Predict class indices for Z using log p(z|k) + log p(k).
    Z: (N,d)
    """
    assert isinstance(Z, np.ndarray) and Z.ndim == 2, "Z must be (N,d)"
    assert np.isfinite(Z).all(), "Z contains NaN/inf"

    N, d = Z.shape
    K = clf.means.shape[0]
    assert clf.means.shape[1] == d, "feature dim mismatch"

    scores = np.zeros((N, K), dtype=np.float64)

    # log p(z|k) + log p(k) up to constant
    for k in range(K):
        diff = (Z - clf.means[k]).astype(np.float64)
        q = np.einsum("nd,dd,nd->n", diff, clf.cov_inv[k], diff)
        scores[:, k] = -0.5 * (q + clf.log_det[k]) + np.log(clf.priors[k] + 1e-12)

    return scores.argmax(axis=1)