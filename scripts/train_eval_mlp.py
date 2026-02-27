from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.config import Config
from src.utils import set_seed
from src.data.utkface import discover_utkface, load_image_gray
from src.data.splits import load_split
from src.data.occlusion import occlude_region
from src.models.mlp import build_mlp
from src.train.torch_train import train_classifier
from src.eval.metrics import overall_accuracy, macro_f1, confmat


def _to_int_list(x) -> List[int]:
    return list(map(int, list(x)))


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _normalize_flat(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Per-sample normalization: (x - mean)/std.
    Works well for MLP on raw pixels and avoids needing dataset-wide stats.
    """
    mu = x.mean()
    sd = x.std(unbiased=False).clamp_min(eps)
    return (x - mu) / sd


class FlatFaceDataset(Dataset):
    def __init__(
        self,
        samples,
        indices,
        image_size: int,
        *,
        occlusion_type: str = "none",
        fill: str = "mean",
        normalize: bool = True,
    ):
        self.samples = samples
        self.indices = _to_int_list(indices)
        self.image_size = int(image_size)
        self.occlusion_type = str(occlusion_type)
        self.fill = str(fill)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        img = load_image_gray(self.samples[i].path, image_size=self.image_size)  # (H,W) float32 [0,1]

        # Occlusion (no-op if "none")
        # Occlusion (random during training if occlusion_type == "random")
        if self.occlusion_type.lower() == "random":
            occ = np.random.choice(["none", "eyes", "mouth", "center"])
            if occ != "none":
                img = occlude_region(img, occ, fill=self.fill)
        elif self.occlusion_type.lower() != "none":
            img = occlude_region(img, self.occlusion_type, fill=self.fill)

        x = torch.from_numpy(img.reshape(-1)).float()

        if self.normalize:
            x = _normalize_flat(x)

        y = int(self.samples[i].y)
        return x, y


@torch.no_grad()
def eval_model(model: torch.nn.Module, loader: DataLoader, device: str, num_classes: int):
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

    # Infer num_classes robustly from train labels; verify against config if present
    y_tr = np.array([int(samples[i].y) for i in tr_idx], dtype=np.int64)
    train_classes = np.unique(y_tr)
    if train_classes.size < 2:
        raise RuntimeError(f"Need >=2 classes in training split; got {train_classes.tolist()}")
    inferred_num_classes = int(train_classes.max()) + 1

    bin_names = cfg.get("labels.bin_names")
    if bin_names is not None:
        expected = len(bin_names)
        if expected != inferred_num_classes:
            raise ValueError(
                f"num_classes mismatch: len(labels.bin_names)={expected} but inferred from train labels is {inferred_num_classes}."
            )
        num_classes = expected
    else:
        num_classes = inferred_num_classes

    # Occlusion config for TEST eval
    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")

    # ---- Train/val loaders (clean) ----
    batch_size = int(cfg.get("mlp.batch_size", 128))
    num_workers = int(cfg.get("mlp.num_workers", 2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    ds_tr = FlatFaceDataset(samples, tr_idx, image_size=image_size, occlusion_type="random", fill=fill, normalize=True)
    ds_va = FlatFaceDataset(samples, va_idx, image_size=image_size, occlusion_type="none", fill=fill, normalize=True)

        # ---- Balanced sampling to improve macro-F1 (no trainer changes) ----
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    class_w = counts.sum() / np.clip(counts, 1.0, None)   # inverse frequency
    sample_w = class_w[y_tr]                               # aligns with ds_tr/tr_idx ordering

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )

    train_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        sampler=sampler,   # sampler replaces shuffle
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    # ---- Model ----
    input_dim = image_size * image_size
    hidden_sizes = cfg.get("mlp.hidden_sizes", [512, 256])
    dropout = float(cfg.get("mlp.dropout", 0.2))
    activation = cfg.get("mlp.activation", "relu")
    use_batchnorm = bool(cfg.get("mlp.use_batchnorm", False))

    model = build_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
        use_batchnorm=use_batchnorm,
    )

    # ---- Train ----
    epochs = int(cfg.get("mlp.epochs", 20))
    lr = float(cfg.get("mlp.lr", 1e-3))
    weight_decay = float(cfg.get("mlp.weight_decay", 0.0))

    result = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    # Load best weights before test evaluation
    model.load_state_dict(result.best_state_dict)

    # Determine union classes train+test for reporting
    y_te_clean = np.array([int(samples[i].y) for i in te_idx], dtype=np.int64)
    eval_classes_union = np.unique(np.concatenate([y_tr, y_te_clean]))

    # ---- Evaluate test under occlusions ----
    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    for occ in occ_types:
        ds_te = FlatFaceDataset(samples, te_idx, image_size=image_size, occlusion_type=str(occ), fill=fill, normalize=True)
        te_loader = DataLoader(
            ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        acc, mf1, cm = eval_model(model, te_loader, device=device, num_classes=num_classes)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}
        confmats[str(occ)] = cm.tolist()

        print(f"[mlp] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    # ---- Save ----
    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": result.best_state_dict,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "activation": activation,
            "use_batchnorm": use_batchnorm,
            "image_size": image_size,
            "bins": bins,
            "seed": seed,
            "config": cfg.raw,
        },
        model_dir / "mlp.pt",
    )

    result_obj = {
        "method": "mlp",
        "split": str(split_path.as_posix()),
        "bins": bins,
        "image_size": image_size,
        "occlusion": {"types": occ_types, "fill": fill},
        "hyperparams": {
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "activation": activation,
            "use_batchnorm": use_batchnorm,
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
    save_json(result_obj, res_dir / "mlp.json")

    print(f"Saved model -> {model_dir / 'mlp.pt'}")
    print(f"Saved results -> {res_dir / 'mlp.json'}")


if __name__ == "__main__":
    main()