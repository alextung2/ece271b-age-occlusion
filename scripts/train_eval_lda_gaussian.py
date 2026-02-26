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
from src.features.pca_lda import fit_pca, transform_pca, fit_lda, transform_lda
from src.models.gaussian import fit_gaussian_classifier, predict_gaussian


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

        img = load_image_gray(samples[i].path, image_size=image_size)

        if occlusion_type is not None and str(occlusion_type).lower() != "none":
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

    # Gaussian reg
    reg = float(cfg.get("classical.gaussian_reg", 1e-3))

    # ---- train (clean) ----
    Xtr, ytr = images_to_matrix(samples, list(map(int, split.train)), image_size, "none", fill)

    classes_tr = np.unique(ytr)
    if classes_tr.size < 2:
        raise RuntimeError(f"Need at least 2 classes in training split for LDA, got {classes_tr.tolist()}")

    # Determine K from union of train+test
    _, yte_clean = images_to_matrix(samples, list(map(int, split.test)), image_size, "none", fill)
    classes_all = np.unique(np.concatenate([ytr, yte_clean]))
    K = int(classes_all.max()) + 1

    # ---- Fisherfaces: PCA -> LDA ----
    # Recommended: PCA to a few hundred dims first
    fisher_pca = int(cfg.get("fisher.pca_components", 300))
    pca_model = fit_pca(Xtr, n_components=fisher_pca, whiten=True)
    Ptr = transform_pca(pca_model, Xtr)

    lda = fit_lda(Ptr, ytr)
    Ztr = transform_lda(lda, Ptr)

    # ---- Gaussian on LDA features ----
    # shared_cov tends to be more stable in LDA space
    g = fit_gaussian_classifier(Ztr, ytr, reg=reg, shared_cov=True, diagonal=False)

    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    for occ in occ_types:
        Xte, yte = images_to_matrix(samples, list(map(int, split.test)), image_size, str(occ), fill)
        Pte = transform_pca(pca_model, Xte)
        Zte = transform_lda(lda, Pte)
        yhat = predict_gaussian(g, Zte)

        acc = overall_accuracy(yte, yhat)
        mf1 = macro_f1(yte, yhat)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}

        cm = confmat(yte, yhat, num_classes=K)
        confmats[str(occ)] = cm.tolist()

        print(f"[lda_gaussian] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"pca": pca_model, "lda": lda, "gaussian": g, "config": cfg.raw},
        model_dir / "lda_gaussian.joblib",
    )

    result_obj = {
        "method": "lda_gaussian",
        "split": str(split_path.as_posix()),
        "bins": bins,
        "image_size": image_size,
        "occlusion": {"types": occ_types, "fill": fill},
        "hyperparams": {
            "fisher_pca_components": fisher_pca,
            "gaussian_reg": reg,
            "gaussian_shared_cov": True,
            "gaussian_diagonal": False,
        },
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