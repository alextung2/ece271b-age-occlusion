from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.svm import SVC

from src.config import Config
from src.utils import set_seed
from src.data.utkface import discover_utkface, load_image_gray, load_image_rgb
from src.data.splits import load_split
from src.data.occlusion import occlude_region
from src.features.pca_lda import transform_pca, transform_lda
from src.features.kernel_pca import transform_kpca
from src.models.gaussian import predict_gaussian
from src.models.mlp import MLP
from src.models.cnn import build_cnn
from src.eval.metrics import overall_accuracy, macro_f1

def build_test_matrices(samples, indices, image_size, occlusion_type, fill):
    X = []
    y = []
    for i in indices:
        img = load_image_gray(samples[i].path, image_size=image_size)
        img = occlude_region(img, region=occlusion_type, fill=fill)
        X.append(img.reshape(-1))
        y.append(samples[i].y)
    return np.stack(X, axis=0).astype(np.float32), np.array(y, dtype=np.int64)

@torch.no_grad()
def eval_cnn(samples, indices, image_size, occlusion_type, fill, cfg, device):
    num_classes = len(cfg.get("labels.bin_names"))
    model = build_cnn(cfg.get("cnn.backbone", "resnet18"), num_classes=num_classes, pretrained=False)
    state = torch.load("outputs/models/cnn.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)

    y_true = []
    y_pred = []
    for i in indices:
        img = load_image_rgb(samples[i].path, image_size=image_size)  # HWC [0,1]
        # occlusion in grayscale; for RGB, apply mask per-channel using grayscale mask
        gray = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)
        gray_occ = occlude_region(gray, region=occlusion_type, fill=fill)
        # apply same occlusion to all channels (simple + consistent)
        occ = img.copy()
        mask = (gray != gray_occ)
        for c in range(3):
            occ[..., c][mask] = occ[..., c].mean()
        x = torch.from_numpy(occ).permute(2,0,1).float().unsqueeze(0).to(device)
        pred = model(x).argmax(dim=1).item()
        y_true.append(samples[i].y)
        y_pred.append(pred)

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    return overall_accuracy(y_true, y_pred), macro_f1(y_true, y_pred)

def main():
    cfg = Config.load("configs/default.yaml")
    set_seed(int(cfg.get("seed", 271)))

    root = cfg.get("data.utkface_root")
    bins = cfg.get("labels.bins")
    image_size = int(cfg.get("data.image_size", 128))
    fill = cfg.get("occlusion.fill", "mean")

    samples = discover_utkface(root, bins)
    split = load_split("outputs/splits/utkface_split.json")
    test_idx = split.test

    out_rows = []

    # Load trained classical artifacts
    pca_gauss = joblib.load("outputs/models/pca_gaussian.joblib")
    lda_gauss = joblib.load("outputs/models/lda_gaussian.joblib")
    svm_on_pca = joblib.load("outputs/models/svm_on_pca.joblib")
    svm_on_lda = joblib.load("outputs/models/svm_on_lda.joblib")
    kpca_svm = joblib.load("outputs/models/kpca_svm.joblib")

    occlusions = cfg.get("occlusion.types", ["none","eyes","mouth","center"])

    for occ in occlusions:
        Xte, yte = build_test_matrices(samples, test_idx, image_size, occ, fill)

        # PCA+Gaussian
        Z = transform_pca(pca_gauss["pca"], Xte)
        pred = predict_gaussian(pca_gauss["gaussian"], Z)
        out_rows.append(("pca_gaussian", occ, overall_accuracy(yte, pred), macro_f1(yte, pred)))

        # LDA+Gaussian
        Z = transform_lda(lda_gauss["lda"], Xte)
        pred = predict_gaussian(lda_gauss["gaussian"], Z)
        out_rows.append(("lda_gaussian", occ, overall_accuracy(yte, pred), macro_f1(yte, pred)))

        # SVM on PCA
        Z = transform_pca(svm_on_pca["pca"], Xte)
        pred = svm_on_pca["svm"].predict(Z)
        out_rows.append(("svm_on_pca", occ, overall_accuracy(yte, pred), macro_f1(yte, pred)))

        # SVM on LDA
        Z = transform_lda(svm_on_lda["lda"], Xte)
        pred = svm_on_lda["svm"].predict(Z)
        out_rows.append(("svm_on_lda", occ, overall_accuracy(yte, pred), macro_f1(yte, pred)))

        # kPCA + SVM
        Z = transform_kpca(kpca_svm["kpca"], Xte)
        pred = kpca_svm["svm"].predict(Z)
        out_rows.append(("kpca_svm", occ, overall_accuracy(yte, pred), macro_f1(yte, pred)))

    # CNN evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for occ in occlusions:
        acc, mf1 = eval_cnn(samples, test_idx, image_size, occ, fill, cfg, device)
        out_rows.append(("cnn", occ, acc, mf1))

    df = pd.DataFrame(out_rows, columns=["method","occlusion","accuracy","macro_f1"])
    out_path = Path("outputs/results/results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df)
    print(f"Saved results to {out_path}")

    # TODO: compute and save robustness drop table (clean minus occluded)
    # TODO: create plots (bar charts) from df

if __name__ == "__main__":
    main()