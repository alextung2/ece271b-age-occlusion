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
    """
    Load (and optionally occlude) images for given sample indices.

    Returns:
      X: (N, D) float32
      y: (N,) int64
    """
    n = len(samples)
    if len(indices) == 0:
        raise ValueError("Got empty indices list (no samples to load).")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for i in indices:
        if not (0 <= i < n):
            raise IndexError(f"Index {i} out of bounds for samples of length {n}.")

        s = samples[i]
        img = load_image_gray(s.path, image_size=image_size)  # (H,W)

        # Ensure 'none' is always a no-op
        if occlusion_type is not None and str(occlusion_type).lower() != "none":
            # Match your LDA script style: occlude_region(img, occlusion_type, fill=fill)
            img = occlude_region(img, occlusion_type, fill=fill)

        X_list.append(img.reshape(-1))
        y_list.append(int(s.y))

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


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

    samples = discover_utkface(root, bins, debug=True)
    if len(samples) == 0:
        raise RuntimeError("No UTKFace samples discovered. Check root path and filtering.")

    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")

    # ---- Train data (always clean) ----
    Xtr, ytr = images_to_matrix(samples, list(map(int, split.train)), image_size, "none", fill)

    # ---- Fit PCA on train only ----
    n_pca = int(cfg.get("classical.pca_components", 100))
    pca_model = fit_pca(Xtr, n_components=n_pca)

    Ztr = transform_pca(pca_model, Xtr)

    # ---- Fit Gaussian on PCA features ----
    reg = float(cfg.get("classical.gaussian_reg", 1e-3))
    clf = fit_gaussian_classifier(Ztr, ytr, reg=reg)

    # Determine K robustly from union of train+test labels
    _, yte_clean = images_to_matrix(samples, list(map(int, split.test)), image_size, "none", fill)
    classes_all = np.unique(np.concatenate([ytr, yte_clean]))
    K = int(classes_all.max()) + 1  # assumes labels are 0..K-1 (your bins code likely does this)

    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    # ---- Eval on test for each occlusion ----
    for occ in occ_types:
        Xte, yte = images_to_matrix(samples, list(map(int, split.test)), image_size, occ, fill)
        Zte = transform_pca(pca_model, Xte)
        yhat = predict_gaussian(clf, Zte)

        acc = overall_accuracy(yte, yhat)
        mf1 = macro_f1(yte, yhat)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}

        cm = confmat(yte, yhat, num_classes=K)
        confmats[str(occ)] = cm.tolist()

        print(f"[pca_gaussian] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    # ---- Save artifacts (match lda_gaussian structure) ----
    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"pca": pca_model, "gaussian": clf, "config": cfg.raw},
        model_dir / "pca_gaussian.joblib",
    )

    result_obj = {
        "method": "pca_gaussian",
        "split": str(split_path.as_posix()),
        "bins": bins,
        "image_size": image_size,
        "occlusion": {"types": occ_types, "fill": fill},
        "hyperparams": {"pca_components": n_pca, "gaussian_reg": reg},
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