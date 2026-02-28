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


def _assert_contiguous_labels(y: np.ndarray) -> int:
    classes = np.unique(y)
    if classes.size == 0:
        raise ValueError("Empty label array y.")
    if classes.min() != 0 or classes.max() != classes.size - 1:
        raise ValueError(f"Labels must be contiguous 0..K-1. Got classes={classes.tolist()}.")
    return int(classes.size)


def _within_cov(Zk: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Within-class covariance with 1/n scaling (ML estimate).
    """
    n, d = Zk.shape
    if n <= 1:
        return np.zeros((d, d), dtype=np.float64)
    X = (Zk - mu).astype(np.float64, copy=False)
    return (X.T @ X) / float(n)


def _make_pd_full(C: np.ndarray, reg: float) -> np.ndarray:
    """
    Add diagonal reg to make PD.
    """
    d = C.shape[0]
    return C + float(reg) * np.eye(d, dtype=np.float64)


def _make_pd_diag(var: np.ndarray, reg: float) -> np.ndarray:
    """
    Ensure diagonal variances are positive with reg.
    """
    var = var.astype(np.float64, copy=False)
    var = np.clip(var, 0.0, None)
    var = var + float(reg)
    return var


def _chol_inv_and_logdet(C: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute inverse and logdet via Cholesky.
    Assumes C is PD.
    """
    L = np.linalg.cholesky(C)  # C = L L^T
    # logdet(C) = 2 * sum(log(diag(L)))
    logdet = 2.0 * float(np.log(np.diag(L)).sum())

    # inv(C) via chol solve: inv = (L^-T)(L^-1)
    # Solve L * Y = I, then solve L^T * X = Y
    I = np.eye(C.shape[0], dtype=np.float64)
    Y = np.linalg.solve(L, I)
    Ci = np.linalg.solve(L.T, Y)
    return Ci, logdet


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
    counts = np.zeros((K,), dtype=np.int64)

    # compute means/priors
    for k in range(K):
        idx = (y == k)
        nk = int(idx.sum())
        if nk <= 0:
            raise ValueError(f"class {k} has 0 samples (unexpected with contiguous labels)")
        Zk = Z[idx]
        mu = Zk.mean(axis=0).astype(np.float64, copy=False)
        means[k] = mu
        counts[k] = nk
        priors[k] = nk / float(N)

    if shared_cov:
        # pooled within-class covariance
        if diagonal:
            pooled_var = np.zeros((d,), dtype=np.float64)
            for k in range(K):
                idx = (y == k)
                Zk = Z[idx]
                mu = means[k][None, :]
                X = (Zk.astype(np.float64, copy=False) - mu)
                # var with 1/n scaling
                var_k = (X * X).mean(axis=0)
                pooled_var += (counts[k] / float(N)) * var_k

            pooled_var = _make_pd_diag(pooled_var, reg=reg)
            Ci = np.diag(1.0 / pooled_var)
            logdet = float(np.log(pooled_var).sum())

            cov_inv = np.repeat(Ci[None, :, :], K, axis=0)
            log_det = np.repeat(np.array([logdet], dtype=np.float64), K, axis=0)

        else:
            C_pool = np.zeros((d, d), dtype=np.float64)
            for k in range(K):
                idx = (y == k)
                Zk = Z[idx]
                mu = means[k][None, :]
                Ck = _within_cov(Zk, mu)
                C_pool += (counts[k] / float(N)) * Ck

            C_pool = _make_pd_full(C_pool, reg=reg)

            # robustify if needed
            try:
                Ci, logdet = _chol_inv_and_logdet(C_pool)
            except np.linalg.LinAlgError:
                C_pool = _make_pd_full(C_pool, reg=10.0 * reg)
                Ci, logdet = _chol_inv_and_logdet(C_pool)

            cov_inv = np.repeat(Ci[None, :, :], K, axis=0)
            log_det = np.repeat(np.array([logdet], dtype=np.float64), K, axis=0)

    else:
        cov_inv = np.zeros((K, d, d), dtype=np.float64)
        log_det = np.zeros((K,), dtype=np.float64)

        for k in range(K):
            idx = (y == k)
            Zk = Z[idx].astype(np.float64, copy=False)
            mu = means[k][None, :]

            if diagonal:
                X = Zk - mu
                var = (X * X).mean(axis=0)  # 1/n
                var = _make_pd_diag(var, reg=reg)
                cov_inv[k] = np.diag(1.0 / var)
                log_det[k] = float(np.log(var).sum())
            else:
                Ck = _within_cov(Zk, mu)
                Ck = _make_pd_full(Ck, reg=reg)
                try:
                    Ci, logdet = _chol_inv_and_logdet(Ck)
                except np.linalg.LinAlgError:
                    Ck = _make_pd_full(Ck, reg=10.0 * reg)
                    Ci, logdet = _chol_inv_and_logdet(Ck)

                cov_inv[k] = Ci
                log_det[k] = logdet

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

    for k in range(K):
        diff = (Z - clf.means[k]).astype(np.float64, copy=False)
        q = np.einsum("nd,dd,nd->n", diff, clf.cov_inv[k], diff)
        scores[:, k] = -0.5 * (q + clf.log_det[k]) + np.log(clf.priors[k] + 1e-12)

    return scores.argmax(axis=1)