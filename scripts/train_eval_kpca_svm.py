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


#HELPER FUNCTIONS
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
def _get_val_indices(split) -> Optional[List[int]]:
    if hasattr(split, "val"):
        v = getattr(split, "val")
        return list(v) if v is not None else None
    if isinstance(split, dict) and "val" in split:
        return list(split["val"]) if split["val"] is not None else None
    return None


#main train/test function to find best performing SVM of given paramater canidates and save it
def main(use_scaler:bool, use_class_weight:bool, gamma_multipliers:list, C_list:list, svm_kernels:list) -> None:

    #read config
    cfg = Config.load("configs/default.yaml")
    seed = int(cfg.get("seed", 271))
    set_seed(seed)
    #check for path to dataset 
    root = cfg.get("data.utkface_root")
    if root is None or not Path(root).exists():
        raise FileNotFoundError(f"UTKFace root does not exist: {root}")
    #get labels
    bins = cfg.get("labels.bins")
    if bins is None:
        raise ValueError("Config missing 'labels.bins'.")
    image_size = int(cfg.get("data.image_size", 128))
    #set output path for splits
    split_path = Path("outputs/splits/utkface_split.json")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split = load_split(str(split_path))
    #import samples
    samples = discover_utkface(root, bins)
    if len(samples) == 0:
        raise RuntimeError("No UTKFace samples discovered. Check root path and filtering.")

    #occlusion settings
    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")
    #KPCA settings
    kpca_k = int(cfg.get("classical.kpca_components", 300))
    kpca_kernel = cfg.get("classical.kpca_kernel", "rbf")
    kpca_gamma_cfg = cfg.get("classical.kpca_gamma", None)

    

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

    #set best to first candidates by default
    best = {"val_macro_f1": -1.0, "kpca_gamma": gamma_candidates[0], "svm_C": float(C_list[0]),
        "svm_kernel": str(svm_kernels[0]), "kpca": None, "svm": None,}

    #Run through all permutations of parameter candidates 
    for gamma in gamma_candidates:
        print("Checking gamma = "+str(gamma))
        kpca = fit_kpca(Xtr, n_components=kpca_k, kernel=kpca_kernel, gamma=gamma, random_state=seed)
        Ztr = transform_kpca(kpca, Xtr)
        for C in C_list:
            print("     Checking C = "+str(C))
            for sk in svm_kernels:
                svm = SVC(C=float(C), kernel=str(sk), class_weight=class_weight)
                svm.fit(Ztr, ytr)

                if have_val:
                    Zva = transform_kpca(kpca, Xva)  # type: ignore[arg-type]
                    yhat_va = svm.predict(Zva)
                    mf1 = float(macro_f1(yva, yhat_va))  # type: ignore[arg-type]
                else:
                    mf1 = 0.0
                print("         Checking kernel "+str(sk)+" with f1: "+str(best["val_macro_f1"]))
                if (not have_val and best["kpca"] is None) or (have_val and mf1 > best["val_macro_f1"]):
                    best.update({ "val_macro_f1": mf1, "kpca_gamma": float(gamma), "svm_C": float(C), "svm_kernel": str(sk), "kpca": kpca, "svm": svm,} )

        if not have_val:
            break

    kpca = best["kpca"]
    svm = best["svm"]
    assert kpca is not None and svm is not None

    if have_val:
        print("[kpca_svm] picked via VAL:", f"gamma={best['kpca_gamma']:.6g}", f"C={best['svm_C']:.6g}", f"svm_kernel={best['svm_kernel']}", f"val_macro_f1={best['val_macro_f1']:.4f}")
    else:
        print("[kpca_svm] no val split; using first combo:", f"gamma={best['kpca_gamma']:.6g}", f"C={best['svm_C']:.6g}", f"svm_kernel={best['svm_kernel']}")

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
    save_json(result_obj, res_dir / f"kpca_svm_scaler{use_scaler}_classweight{use_class_weight}.json")
    print(f"Saved model -> {model_dir / 'kpca_svm.joblib'}")
    print(f"Saved results -> {res_dir / 'kpca_svm.json'}")


#========================================================================
#HYPER-PARAMETERS
#use_scaler = False
#use_class_weight = True
gamma_multipliers = [ 1, 3, 10] #gamma multipliers for PCA
C_list = [0.1, 1, 10, 100]  #c canidates for SVM
svm_kernels = ["linear", "rbf"] #kernels for SVM
#========================================================================

#run KPCA and SVM for all binary combinations of use_scaler and use_class_weight booleans
for use_scaler in [True, False]:
    for use_class_weight in [True, False]:
        print("Trying use_scaler="+str(use_scaler)+" and use_class_weight="+str(use_class_weight))
        main(use_scaler, use_class_weight, gamma_multipliers, C_list, svm_kernels)