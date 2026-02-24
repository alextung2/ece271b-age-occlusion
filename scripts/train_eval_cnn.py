from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import Config
from src.utils import set_seed
from src.data.utkface import discover_utkface, load_image_rgb
from src.data.splits import load_split
from src.data.occlusion import occlude_region
from src.models.cnn import build_cnn
from src.train.torch_train import train_classifier
from src.eval.metrics import overall_accuracy, macro_f1, confmat


def _to_int_list(x) -> List[int]:
    return list(map(int, list(x)))


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def occlude_rgb(img_hwc: np.ndarray, occlusion_type: str, fill: str) -> np.ndarray:
    """
    Apply the same occlusion region to an RGB image by applying occlude_region
    channel-wise. Assumes occlude_region works on 2D arrays.

    img_hwc: (H, W, 3) float32 in [0,1]
    """
    if occlusion_type is None or str(occlusion_type).lower() == "none":
        return img_hwc

    assert isinstance(img_hwc, np.ndarray) and img_hwc.ndim == 3 and img_hwc.shape[2] == 3
    out = img_hwc.copy()
    # Apply the region mask separately to each channel (same region geometry).
    for c in range(3):
        out[..., c] = occlude_region(out[..., c], occlusion_type, fill=fill)
    return out


class RgbFaceDataset(Dataset):
    def __init__(
        self,
        samples,
        indices: List[int],
        image_size: int,
        *,
        occlusion_type: str = "none",
        fill: str = "mean",
    ):
        self.samples = samples
        self.indices = _to_int_list(indices)
        self.image_size = int(image_size)
        self.occlusion_type = str(occlusion_type)
        self.fill = str(fill)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        i = self.indices[idx]
        s = self.samples[i]
        img = load_image_rgb(s.path, image_size=self.image_size)  # (H,W,3) float32 [0,1]

        # Occlusion (no-op if "none")
        img = occlude_rgb(img, self.occlusion_type, fill=self.fill)

        # TODO: add augmentation here if desired (flip, jitter, etc.)
        x = torch.from_numpy(img).permute(2, 0, 1).float()  # (3,H,W)
        y = int(s.y)
        return x, y


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    ys: List[int] = []
    yhats: List[int] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        ys.extend([int(v) for v in yb.numpy()])
        yhats.extend([int(v) for v in pred])

    y = np.asarray(ys, dtype=np.int64)
    yhat = np.asarray(yhats, dtype=np.int64)

    acc = float(overall_accuracy(y, yhat))
    mf1 = float(macro_f1(y, yhat))
    cm = confmat(y, yhat, num_classes=int(num_classes))
    return acc, mf1, cm


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

    samples = discover_utkface(root, bins)
    if len(samples) == 0:
        raise RuntimeError("No UTKFace samples discovered. Check root path and filtering.")

    tr_idx = _to_int_list(split.train)
    va_idx = _to_int_list(split.val)
    te_idx = _to_int_list(split.test)

    # Infer num_classes safely from the data (robust to config mismatch)
    y_tr = np.array([int(samples[i].y) for i in tr_idx], dtype=np.int64)
    train_classes = np.unique(y_tr)
    if train_classes.size < 2:
        raise RuntimeError(f"Need >=2 classes in training split; got {train_classes.tolist()}")

    # If you really want to tie to cfg labels.bin_names, you can assert they agree:
    bin_names = cfg.get("labels.bin_names")
    if bin_names is not None:
        expected = len(bin_names)
        inferred = int(train_classes.max()) + 1
        if expected != inferred:
            raise ValueError(
                f"num_classes mismatch: len(labels.bin_names)={expected} but inferred from train labels is {inferred}. "
                "Fix your config or label mapping."
            )
        num_classes = expected
    else:
        num_classes = int(train_classes.max()) + 1

    # Occlusion config (for TEST evaluation only)
    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")

    # ---- Datasets / loaders (train/val clean only) ----
    ds_tr = RgbFaceDataset(samples, tr_idx, image_size=image_size, occlusion_type="none", fill=fill)
    ds_va = RgbFaceDataset(samples, va_idx, image_size=image_size, occlusion_type="none", fill=fill)

    batch_size = int(cfg.get("cnn.batch_size", 64))
    num_workers = int(cfg.get("cnn.num_workers", 2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    # ---- Build model ----
    backbone = cfg.get("cnn.backbone", "resnet18")
    pretrained = bool(cfg.get("cnn.pretrained", True))
    model = build_cnn(backbone=backbone, num_classes=num_classes, pretrained=pretrained)

    # ---- Train ----
    epochs = int(cfg.get("cnn.epochs", 10))
    lr = float(cfg.get("cnn.lr", 3e-4))
    weight_decay = float(cfg.get("cnn.weight_decay", 1e-4))

    result = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    # Load best weights before evaluation
    model.load_state_dict(result.best_state_dict)

    # ---- Evaluate on TEST for each occlusion ----
    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    # Determine union of train+test classes for reporting (like your PCA/LDA)
    y_te_clean = np.array([int(samples[i].y) for i in te_idx], dtype=np.int64)
    eval_classes_union = np.unique(np.concatenate([y_tr, y_te_clean]))

    for occ in occ_types:
        ds_te = RgbFaceDataset(
            samples,
            te_idx,
            image_size=image_size,
            occlusion_type=str(occ),
            fill=fill,
        )
        te_loader = DataLoader(
            ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        acc, mf1, cm = eval_model(model, te_loader, device=device, num_classes=num_classes)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}
        confmats[str(occ)] = cm.tolist()

        print(f"[cnn] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    # ---- Save artifacts ----
    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Save weights + minimal metadata (so you can reload)
    torch.save(
        {
            "state_dict": result.best_state_dict,
            "backbone": backbone,
            "pretrained": pretrained,
            "num_classes": num_classes,
            "image_size": image_size,
            "bins": bins,
            "seed": seed,
            "config": cfg.raw,
        },
        model_dir / "cnn.pt",
    )

    result_obj = {
        "method": "cnn",
        "split": str(split_path.as_posix()),
        "bins": bins,
        "image_size": image_size,
        "occlusion": {"types": occ_types, "fill": fill},
        "hyperparams": {
            "backbone": backbone,
            "pretrained": pretrained,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "metrics": metrics,
        "confusion_matrices": confmats,
        "train_classes": train_classes.tolist(),
        "eval_classes_union_train_test": eval_classes_union.tolist(),
        "best_val_metric": float(getattr(result, "best_val_metric", float("nan"))),
        "device": device,
    }
    save_json(result_obj, res_dir / "cnn.json")

    print(f"Saved model -> {model_dir / 'cnn.pt'}")
    print(f"Saved results -> {res_dir / 'cnn.json'}")


if __name__ == "__main__":
    main()