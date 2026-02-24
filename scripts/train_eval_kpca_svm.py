from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import Config
from src.data.occlusion import occlude_region
from src.data.splits import load_split
from src.data.utkface import discover_utkface, load_image_gray
from src.eval.metrics import confmat, macro_f1, overall_accuracy
from src.features.kernel_pca import fit_kpca, suggest_gamma_rbf, transform_kpca
from src.utils import set_seed


def images_to_matrix(
    samples,
    indices: List[int],
    image_size: int,
    occlusion_type: str,
    fill: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in indices:
        img = load_image_gray(samples[i].path, image_size=image_size)

        # Ensure "none" is always a no-op
        if occlusion_type is not None and str(occlusion_type).lower() != "none":
            img = occlude_region(img, occlusion_type, fill=fill)

        X.append(img.reshape(-1))
        y.append(samples[i].y)
    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# EDIT HERE (Luke Wittemann): This is the ONLY place you should change code.
#
# Goal:
#   Pick hyperparameter combinations that maximize validation macro-F1
#   (if split.val exists). If there is no val split, the script will just use
#   the first combo in your lists.
#
# Start simple:
#   gamma_multipliers=[1.0], C_list=[10.0], svm_kernels=["linear"]
# Then expand gradually by uncommenting lines.
# =============================================================================
def choose_hyperparams() -> Dict[str, object]:
    # ---- switches ----
    use_scaler = True
    use_class_weight = False  # set True if macro-F1 << accuracy (imbalance)

    # ---- RBF-KPCA gamma candidates ----
    # actual gamma used = suggest_gamma_rbf(Xtr) * multiplier
    gamma_multipliers = [
        1.0,  # baseline
        # 1/3,
        # 3.0,
        # 10.0,
    ]

    # ---- SVM C candidates ----
    C_list = [
        10.0,  # baseline
        # 1.0,
        # 100.0,
    ]

    # ---- SVM kernel candidates ----
    svm_kernels = [
        "linear",  # baseline
        # "rbf",
    ]

    return {
        "use_scaler": use_scaler,
        "use_class_weight": use_class_weight,
        "gamma_multipliers": gamma_multipliers,
        "C_list": C_list,
        "svm_kernels": svm_kernels,
    }


def _get_val_indices(split) -> Optional[List[int]]:
    if hasattr(split, "val"):
        v = getattr(split, "val")
        return list(v) if v is not None else None
    if isinstance(split, dict) and "val" in split:
        return list(split["val"]) if split["val"] is not None else None
    return None


def main() -> None:
    cfg = Config.load("configs/default.yaml")
    seed = int(cfg.get("seed", 271))
    set_seed(seed)

    root = cfg.get("data.utkface_root")
    if root is None or not Path(root).exists():
        raise FileNotFoundError(f"UTKFace root does not exist: {root}")

    bins = cfg.get("labels.bins")
    if bins is None:
        raise ValueError("Config missing 'labels.bins'.")

    image_size = int(cfg.get("data.image_size", 128))

    split_path = Path("outputs/splits/utkface_split.json")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split = load_split(str(split_path))

    samples = discover_utkface(root, bins)
    if len(samples) == 0:
        raise RuntimeError("No UTKFace samples discovered. Check root path and filtering.")

    # occlusion settings
    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")

    # KPCA settings
    kpca_k = int(cfg.get("classical.kpca_components", 200))
    kpca_kernel = cfg.get("classical.kpca_kernel", "rbf")
    kpca_gamma_cfg = cfg.get("classical.kpca_gamma", None)

    # Luke's editable settings
    hp = choose_hyperparams()
    use_scaler = bool(hp["use_scaler"])
    use_class_weight = bool(hp["use_class_weight"])
    gamma_multipliers = list(hp["gamma_multipliers"])
    C_list = list(hp["C_list"])
    svm_kernels = list(hp["svm_kernels"])

    # Train data (always clean)
    Xtr, ytr = images_to_matrix(samples, list(map(int, split.train)), image_size, occlusion_type="none", fill=fill)

    # Optional scaling
    scaler = None
    if use_scaler:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(Xtr)

    # Val split (if available)
    val_indices = _get_val_indices(split)
    have_val = val_indices is not None and len(val_indices) > 0

    Xva = yva = None
    if have_val:
        Xva, yva = images_to_matrix(samples, list(map(int, val_indices)), image_size, occlusion_type="none", fill=fill)
        if scaler is not None:
            Xva = scaler.transform(Xva)

    # Gamma candidates
    if kpca_gamma_cfg is not None:
        gamma_candidates = [float(kpca_gamma_cfg)]
    else:
        if kpca_kernel == "rbf":
            g0 = float(suggest_gamma_rbf(Xtr))
            gamma_candidates = [g0 * float(m) for m in gamma_multipliers]
        else:
            gamma_candidates = [1.0]

    class_weight = "balanced" if use_class_weight else None

    # Pick best combo by val macro-F1 (if val exists), else first combo
    best = {
        "val_macro_f1": -1.0,
        "kpca_gamma": gamma_candidates[0],
        "svm_C": float(C_list[0]),
        "svm_kernel": str(svm_kernels[0]),
        "kpca": None,
        "svm": None,
    }

    for gamma in gamma_candidates:
        kpca = fit_kpca(Xtr, n_components=kpca_k, kernel=kpca_kernel, gamma=gamma, random_state=seed)
        Ztr = transform_kpca(kpca, Xtr)

        for C in C_list:
            for sk in svm_kernels:
                svm = SVC(C=float(C), kernel=str(sk), class_weight=class_weight)
                svm.fit(Ztr, ytr)

                if have_val:
                    Zva = transform_kpca(kpca, Xva)  # type: ignore[arg-type]
                    yhat_va = svm.predict(Zva)
                    mf1 = float(macro_f1(yva, yhat_va))  # type: ignore[arg-type]
                else:
                    mf1 = 0.0

                if (not have_val and best["kpca"] is None) or (have_val and mf1 > best["val_macro_f1"]):
                    best.update(
                        {
                            "val_macro_f1": mf1,
                            "kpca_gamma": float(gamma),
                            "svm_C": float(C),
                            "svm_kernel": str(sk),
                            "kpca": kpca,
                            "svm": svm,
                        }
                    )

        if not have_val:
            break

    kpca = best["kpca"]
    svm = best["svm"]
    assert kpca is not None and svm is not None

    if have_val:
        print(
            "[kpca_svm] picked via VAL:",
            f"gamma={best['kpca_gamma']:.6g}",
            f"C={best['svm_C']:.6g}",
            f"svm_kernel={best['svm_kernel']}",
            f"val_macro_f1={best['val_macro_f1']:.4f}",
        )
    else:
        print(
            "[kpca_svm] no val split; using first combo:",
            f"gamma={best['kpca_gamma']:.6g}",
            f"C={best['svm_C']:.6g}",
            f"svm_kernel={best['svm_kernel']}",
        )

    # Robust K from union of train+test labels (clean)
    Xte_clean, yte_clean = images_to_matrix(samples, list(map(int, split.test)), image_size, occlusion_type="none", fill=fill)
    classes_all = np.unique(np.concatenate([ytr, yte_clean]))
    K = int(classes_all.max()) + 1  # assumes contiguous 0..K-1

    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    for occ in occ_types:
        Xte, yte = images_to_matrix(samples, list(map(int, split.test)), image_size, occlusion_type=occ, fill=fill)

        if scaler is not None:
            Xte = scaler.transform(Xte)

        Zte = transform_kpca(kpca, Xte)
        yhat = svm.predict(Zte)

        acc = overall_accuracy(yte, yhat)
        mf1 = macro_f1(yte, yhat)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}

        cm = confmat(yte, yhat, num_classes=K)
        confmats[str(occ)] = cm.tolist()

        print(f"[kpca_svm] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    # Save artifacts
    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "kpca": kpca,
            "svm": svm,
            "scaler": scaler,
            "config": cfg.raw,
            "notes": {
                "use_scaler": use_scaler,
                "use_class_weight": use_class_weight,
                "picked": {
                    "kpca_gamma": best["kpca_gamma"],
                    "svm_C": best["svm_C"],
                    "svm_kernel": best["svm_kernel"],
                    "val_macro_f1": best["val_macro_f1"] if have_val else None,
                },
            },
        },
        model_dir / "kpca_svm.joblib",
    )

    result_obj = {
        "method": "kpca_svm",
        "split": str(split_path.as_posix()),
        "bins": bins,
        "image_size": image_size,
        "occlusion": {"types": occ_types, "fill": fill},
        "hyperparams": {
            "kpca_components": kpca_k,
            "kpca_kernel": kpca_kernel,
            "kpca_gamma": best["kpca_gamma"],
            "svm_C": best["svm_C"],
            "svm_kernel": best["svm_kernel"],
            "use_scaler": use_scaler,
            "use_class_weight": use_class_weight,
        },
        "metrics": metrics,
        "confusion_matrices": confmats,
        "train_classes": np.unique(ytr).tolist(),
        "eval_classes_union_train_test": classes_all.tolist(),
    }

    save_json(result_obj, res_dir / "kpca_svm.json")
    print(f"Saved model -> {model_dir / 'kpca_svm.joblib'}")
    print(f"Saved results -> {res_dir / 'kpca_svm.json'}")


if __name__ == "__main__":
    main()