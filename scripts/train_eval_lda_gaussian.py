# scripts/train_eval_lda_gaussian.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from src.config import Config
from src.utils import set_seed
from src.data.splits import load_split
from src.data.utkface import discover_utkface, load_image_gray
from src.data.occlusion import occlude_region
from src.eval.metrics import overall_accuracy, macro_f1, confmat
from src.features.pca_lda import fit_lda, transform_lda
from src.models.gaussian import fit_gaussian_classifier, predict_gaussian


def images_to_matrix(
    samples,
    indices: List[int],
    image_size: int,
    occlusion_type: str,
    fill: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads images for the given indices, optionally applies occlusion,
    flattens into rows, returns (X, y).

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

        img = load_image_gray(samples[i].path, image_size=image_size)

        # Make 'none' truly a no-op even if occlude_region doesn't handle it.
        if occlusion_type is not None and occlusion_type.lower() != "none":
            img = occlude_region(img, occlusion_type, fill=fill)

        X_list.append(img.reshape(-1))
        y_list.append(int(samples[i].y))

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.array(y_list, dtype=np.int64)
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
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"UTKFace root does not exist: {root_path}")

    bins = cfg.get("labels.bins")
    if bins is None:
        raise ValueError("Config missing 'labels.bins' (age bin edges).")

    image_size = int(cfg.get("data.image_size", 128))

    split_path = Path("outputs/splits/utkface_split.json")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split = load_split(str(split_path))

    samples = discover_utkface(str(root_path), bins)
    if len(samples) == 0:
        raise RuntimeError("No UTKFace samples discovered. Check root path and filenames.")

    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")
    reg = float(cfg.get("classical.gaussian_reg", 1e-3))

    # ---- train (always clean) ----
    Xtr, ytr = images_to_matrix(
        samples, split.train, image_size, occlusion_type="none", fill=fill
    )

    classes_tr = np.unique(ytr)
    if classes_tr.size < 2:
        raise RuntimeError(
            f"Need at least 2 classes in training split for LDA, got {classes_tr.size}: {classes_tr.tolist()}"
        )

    # Number of classes we'll evaluate over: use union of train+test labels so CM is consistent.
    # (Safer than using max over all discovered samples.)
    _, yte_clean = images_to_matrix(
        samples, split.test, image_size, occlusion_type="none", fill=fill
    )
    classes_all = np.unique(np.concatenate([ytr, yte_clean]))
    K = int(classes_all.max()) + 1  # assumes labels are 0..K-1; typical for binned UTKFace

    # ---- fit LDA (max dim is C-1) ----
    # If your fit_lda supports n_components, you could pass:
    # n_components = min(classes_tr.size - 1, Xtr.shape[1])
    # lda = fit_lda(Xtr, ytr, n_components=n_components)
    lda = fit_lda(Xtr, ytr)

    Ztr = transform_lda(lda, Xtr)

    # ---- fit Gaussian on LDA features ----
    g = fit_gaussian_classifier(Ztr, ytr, reg=reg)

    # ---- eval on test for each occlusion ----
    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    for occ in occ_types:
        Xte, yte = images_to_matrix(samples, split.test, image_size, occlusion_type=occ, fill=fill)
        Zte = transform_lda(lda, Xte)
        yhat = predict_gaussian(g, Zte)

        acc = overall_accuracy(yte, yhat)
        mf1 = macro_f1(yte, yhat)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}

        cm = confmat(yte, yhat, num_classes=K)
        confmats[str(occ)] = cm.tolist()

        print(f"[lda_gaussian] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    # ---- save artifacts ----
    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"lda": lda, "gaussian": g, "config": cfg.raw},
        model_dir / "lda_gaussian.joblib",
    )

    result_obj = {
        "method": "lda_gaussian",
        "split": str(split_path.as_posix()),
        "bins": bins,
        "image_size": image_size,
        "occlusion": {"types": occ_types, "fill": fill},
        "hyperparams": {"gaussian_reg": reg},
        "metrics": metrics,
        "confusion_matrices": confmats,
        "train_classes": classes_tr.tolist(),
        "eval_classes_union_train_test": classes_all.tolist(),
    }
    save_json(result_obj, res_dir / "lda_gaussian.json")

    print(f"Saved model -> {model_dir / 'lda_gaussian.joblib'}")
    print(f"Saved results -> {res_dir / 'lda_gaussian.json'}")


if __name__ == "__main__":
    main()