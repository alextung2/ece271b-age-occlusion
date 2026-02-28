# scripts/train_eval_pca_gaussian.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from src.config import Config
from src.utils import set_seed
from src.data.utkface import discover_utkface, load_image_gray
from src.data.splits import load_split
from src.features.pca_lda import fit_pca, transform_pca
from src.models.gaussian import fit_gaussian_classifier, predict_gaussian
from src.eval.metrics import overall_accuracy, macro_f1, confmat
from src.data.occlusion import occlude_region


def images_to_matrix(
    samples,
    indices: List[int],
    image_size: int,
    occlusion_type: str,
    fill: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(samples)
    if len(indices) == 0:
        raise ValueError("Got empty indices list (no samples to load).")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for i in indices:
        if not (0 <= i < n):
            raise IndexError(f"Index {i} out of bounds for samples of length {n}.")

        s = samples[i]
        img = load_image_gray(s.path, image_size=image_size)

        if occlusion_type is not None and str(occlusion_type).lower() != "none":
            img = occlude_region(img, occlusion_type, fill=fill)

        X_list.append(img.reshape(-1))
        y_list.append(int(s.y))

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def standardize_train_test(Ztr: np.ndarray, Zte: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training mean/std.
    Returns (Ztr_std, Zte_std, mean, std).
    """
    mu = Ztr.mean(axis=0, keepdims=True)
    sd = Ztr.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (Ztr - mu) / sd, (Zte - mu) / sd, mu, sd


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    cfg = Config.load("configs/default.yaml")
    seed = int(cfg.get("seed", 271))
    set_seed(seed)

    root = cfg.get("data.utkface_root")
    if root is None:
        raise ValueError("Config missing 'data.utkface_root'.")
    if not Path(root).exists():
        raise FileNotFoundError(f"UTKFace root does not exist: {root}")

    bins = cfg.get("labels.bins")
    if bins is None:
        raise ValueError("Config missing 'labels.bins'.")

    image_size = int(cfg.get("data.image_size", 128))

    split_path = Path("outputs/splits/utkface_split.json")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split = load_split(str(split_path))

    # NOTE: keep debug=False for normal runs; debug=True can be noisy/slow depending on your implementation
    samples = discover_utkface(root, bins)
    if len(samples) == 0:
        raise RuntimeError("No UTKFace samples discovered. Check root path and filtering.")

    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")

    tr_idx = list(map(int, split.train))
    te_idx = list(map(int, split.test))

    # ---- Train (clean) ----
    Xtr, ytr = images_to_matrix(samples, tr_idx, image_size, "none", fill)

    # PCA dims: read from YAML pca.components (matches your current yaml)
    n_pca = int(cfg.get("pca.components", 200))

    # IMPORTANT: whitening often hurts Gaussian classification; default to False
    whiten = bool(cfg.get("pca.whiten", False))
    pca_model = fit_pca(Xtr, n_components=n_pca, whiten=whiten)

    Ztr = transform_pca(pca_model, Xtr)

    # Standardize PCA features for stability
    # (This tends to make Gaussian fitting behave better.)
    Ztr_std, _, zmu, zsd = standardize_train_test(Ztr, Ztr)

    # Gaussian hyperparams: read from gaussian.* (matches yaml)
    reg = float(cfg.get("gaussian.reg", 1e-3))
    diagonal = bool(cfg.get("gaussian.diagonal", True))
    shared_cov = bool(cfg.get("gaussian.shared_cov", True))  # often more stable than class-specific

    clf = fit_gaussian_classifier(Ztr_std, ytr, reg=reg, shared_cov=shared_cov, diagonal=diagonal)

    # Determine K from union of train+test
    _, yte_clean = images_to_matrix(samples, te_idx, image_size, "none", fill)
    classes_all = np.unique(np.concatenate([ytr, yte_clean]))
    K = int(classes_all.max()) + 1

    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    for occ in occ_types:
        Xte, yte = images_to_matrix(samples, te_idx, image_size, str(occ), fill)
        Zte = transform_pca(pca_model, Xte)
        _, Zte_std, _, _ = standardize_train_test(Ztr, Zte)  # use train stats internally
        # NOTE: standardize_train_test recomputes mu/std if called this way.
        # To avoid recompute, do it explicitly:
        Zte_std = (Zte - zmu) / zsd

        yhat = predict_gaussian(clf, Zte_std)

        acc = overall_accuracy(yte, yhat)
        mf1 = macro_f1(yte, yhat)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}

        cm = confmat(yte, yhat, num_classes=K)
        confmats[str(occ)] = cm.tolist()

        print(f"[pca_gaussian] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pca": pca_model,
            "gaussian": clf,
            "pca_feature_mean": zmu,
            "pca_feature_std": zsd,
            "config": cfg.raw,
        },
        model_dir / "pca_gaussian.joblib",
    )

    result_obj = {
        "method": "pca_gaussian",
        "split": str(split_path.as_posix()),
        "bins": bins,
        "image_size": image_size,
        "occlusion": {"types": occ_types, "fill": fill},
        "hyperparams": {
            "pca_components": n_pca,
            "pca_whiten": whiten,
            "feature_standardize": True,
            "gaussian_reg": reg,
            "gaussian_diagonal": diagonal,
            "gaussian_shared_cov": shared_cov,
        },
        "metrics": metrics,
        "confusion_matrices": confmats,
        "train_classes": np.unique(ytr).tolist(),
        "eval_classes_union_train_test": classes_all.tolist(),
    }
    save_json(result_obj, res_dir / "pca_gaussian.json")

    print(f"Saved model -> {model_dir / 'pca_gaussian.joblib'}")
    print(f"Saved results -> {res_dir / 'pca_gaussian.json'}")


if __name__ == "__main__":
    main()