from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import Config
from src.utils import set_seed
from src.data.utkface import discover_utkface, load_image_rgb
from src.data.splits import load_split
from src.data.occlusion import occlude_region
from src.models.cnn import build_cnn
from src.train.torch_train import train_classifier
from src.eval.metrics import overall_accuracy, macro_f1, confmat


# ImageNet normalization (IMPORTANT when using pretrained=True)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _to_int_list(x) -> List[int]:
    return list(map(int, list(x)))


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def occlude_rgb(img_hwc: np.ndarray, occlusion_type: str, fill: str) -> np.ndarray:
    """
    Apply occlusion to an RGB image by applying occlude_region channel-wise.
    img_hwc: (H, W, 3) float32 in [0,1]
    """
    if occlusion_type is None or str(occlusion_type).lower() == "none":
        return img_hwc

    assert isinstance(img_hwc, np.ndarray) and img_hwc.ndim == 3 and img_hwc.shape[2] == 3
    out = img_hwc.copy()
    for c in range(3):
        out[..., c] = occlude_region(out[..., c], occlusion_type, fill=fill)
    return out


def _rand_color_jitter(
    x: torch.Tensor, brightness: float = 0.10, contrast: float = 0.10
) -> torch.Tensor:
    """
    Lightweight color jitter without torchvision.transforms dependency.
    x: (3,H,W) in [0,1]
    """
    b = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * brightness
    x = x * b

    c = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * contrast
    mean = x.mean(dim=(1, 2), keepdim=True)
    x = (x - mean) * c + mean

    return x.clamp(0.0, 1.0)


def _random_resized_crop(
    x: torch.Tensor,
    out_size: int,
    scale: Tuple[float, float] = (0.90, 1.0),
    ratio: Tuple[float, float] = (0.90, 1.10),
) -> torch.Tensor:
    """
    Gentler random resized crop on a tensor image.
    x: (3,H,W) float in [0,1]
    returns: (3,out_size,out_size)

    NOTE:
      - UTKFace framing is inconsistent; aggressive crops can cut off age cues.
      - Default is intentionally mild: scale in [0.90, 1.0], near-square ratio.
    """
    _, H, W = x.shape
    img_area = H * W

    for _ in range(10):
        s = float(torch.empty(1).uniform_(scale[0], scale[1]).item())
        target_area = s * img_area
        r = float(torch.empty(1).uniform_(ratio[0], ratio[1]).item())

        crop_h = int(round(np.sqrt(target_area / r)))
        crop_w = int(round(np.sqrt(target_area * r)))

        if 1 <= crop_h <= H and 1 <= crop_w <= W:
            top = 0 if crop_h == H else int(torch.randint(0, H - crop_h + 1, (1,)).item())
            left = 0 if crop_w == W else int(torch.randint(0, W - crop_w + 1, (1,)).item())
            x2 = x[:, top : top + crop_h, left : left + crop_w]
            x2 = x2.unsqueeze(0)
            x2 = F.interpolate(x2, size=(out_size, out_size), mode="bilinear", align_corners=False)
            return x2.squeeze(0)

    # Fallback: just resize (no crop)
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def _center_crop(x: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Center crop a (3,H,W) tensor. If crop_size >= min(H,W), returns x unchanged.
    """
    _, H, W = x.shape
    if crop_size >= H or crop_size >= W:
        return x
    top = (H - crop_size) // 2
    left = (W - crop_size) // 2
    return x[:, top : top + crop_size, left : left + crop_size]


def _random_erasing(
    x: torch.Tensor,
    p: float = 0.0,
    area: Tuple[float, float] = (0.02, 0.12),
    aspect: Tuple[float, float] = (0.3, 3.3),
) -> torch.Tensor:
    """
    Random erasing on tensor image (3,H,W) in [0,1]. Fills with per-image mean.
    """
    if p <= 0.0 or torch.rand(1).item() > p:
        return x

    C, H, W = x.shape
    img_area = H * W

    a = float(torch.empty(1).uniform_(area[0], area[1]).item())
    target_area = a * img_area
    asp = float(torch.empty(1).uniform_(aspect[0], aspect[1]).item())

    h = int(round(np.sqrt(target_area * asp)))
    w = int(round(np.sqrt(target_area / asp)))
    if h < 1 or w < 1 or h >= H or w >= W:
        return x

    top = int(torch.randint(0, H - h + 1, (1,)).item())
    left = int(torch.randint(0, W - w + 1, (1,)).item())

    fill = x.mean(dim=(1, 2), keepdim=True)
    x[:, top : top + h, left : left + w] = fill
    return x


def compute_class_weights(y: np.ndarray, num_classes: int, kind: str = "effective") -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.

    kind:
      - "none": all ones
      - "inverse": 1/freq
      - "sqrt_inv": 1/sqrt(freq)
      - "effective": effective number of samples (Cui et al.)
    """
    kind = str(kind).lower()
    if kind == "none":
        return torch.ones(int(num_classes), dtype=torch.float32)

    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    if kind == "inverse":
        w = 1.0 / counts
    elif kind == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)
    elif kind == "effective":
        beta = 0.9995
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / eff
    else:
        raise ValueError(f"Unknown kind={kind}")

    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32)


class RgbFaceDataset(Dataset):
    def __init__(
        self,
        samples,
        indices: List[int],
        image_size: int,
        *,
        occlusion_type: str = "none",
        fill: str = "mean",
        augment: bool = False,
        imagenet_norm: bool = True,
        # occlusion augmentation (TRAIN ONLY)
        occlusion_aug_prob: float = 0.0,
        occlusion_aug_types: List[str] | None = None,
        # stronger aug (TRAIN ONLY)
        crop_aug: bool = True,
        crop_scale: Tuple[float, float] = (0.90, 1.0),
        erasing_p: float = 0.0,
        # eval-time stabilization
        eval_center_crop_frac: float = 0.0,  # e.g. 0.90 to crop 90% center region before resize
    ):
        self.samples = samples
        self.indices = _to_int_list(indices)
        self.image_size = int(image_size)

        self.occlusion_type = str(occlusion_type)
        self.fill = str(fill)

        self.augment = bool(augment)
        self.imagenet_norm = bool(imagenet_norm)

        self.occlusion_aug_prob = float(occlusion_aug_prob)
        self.occlusion_aug_types = occlusion_aug_types or ["eyes", "mouth", "center"]
        self.occlusion_aug_types = [str(t) for t in self.occlusion_aug_types if str(t).lower() != "none"]

        self.crop_aug = bool(crop_aug)
        self.crop_scale = (float(crop_scale[0]), float(crop_scale[1]))
        self.erasing_p = float(erasing_p)

        self.eval_center_crop_frac = float(eval_center_crop_frac)

        if not (0.0 <= self.occlusion_aug_prob <= 1.0):
            raise ValueError("occlusion_aug_prob must be in [0,1].")
        if not (0.0 <= self.erasing_p <= 1.0):
            raise ValueError("erasing_p must be in [0,1].")
        if not (0.0 <= self.eval_center_crop_frac < 1.0):
            raise ValueError("eval_center_crop_frac must be in [0,1).")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        i = self.indices[idx]
        s = self.samples[i]

        img = load_image_rgb(s.path, image_size=self.image_size)  # (H,W,3) float32 [0,1]
        y = int(s.y)

        # TRAIN: structured occlusion augmentation
        if self.augment and self.occlusion_aug_prob > 0.0 and len(self.occlusion_aug_types) > 0:
            if random.random() < self.occlusion_aug_prob:
                occ = random.choice(self.occlusion_aug_types)
                img = occlude_rgb(img, occ, fill=self.fill)

        # EVAL: fixed occlusion
        if self.occlusion_type is not None and self.occlusion_type.lower() != "none":
            img = occlude_rgb(img, self.occlusion_type, fill=self.fill)

        x = torch.from_numpy(img).permute(2, 0, 1).float()  # (3,H,W) in [0,1]

        # EVAL stabilization: optional mild center crop (helps if UTKFace has background noise)
        if (not self.augment) and self.eval_center_crop_frac > 0.0:
            _, H, W = x.shape
            crop_sz = int(round(min(H, W) * self.eval_center_crop_frac))
            if crop_sz > 0:
                x = _center_crop(x, crop_sz)
                x = x.unsqueeze(0)
                x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
                x = x.squeeze(0)

        # TRAIN: augmentations
        if self.augment:
            if self.crop_aug and torch.rand(1).item() < 0.9:
                x = _random_resized_crop(
                    x,
                    out_size=self.image_size,
                    scale=self.crop_scale,
                    ratio=(0.90, 1.10),
                )

            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])

            x = _rand_color_jitter(x, brightness=0.10, contrast=0.10)
            x = _random_erasing(x, p=self.erasing_p)

        # ImageNet normalization
        if self.imagenet_norm:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD

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


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _pick_device(cfg: Config) -> str:
    req = str(cfg.get("cnn.device", "cuda")).lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[warn] cnn.device=cuda requested but CUDA not available; using cpu")
        return "cpu"
    if req == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    cfg = Config.load("configs/default.yaml")
    seed = int(cfg.get("seed", 271))
    set_seed(seed)

    print(f"[python] {sys.executable}")
    print(f"[torch] version={torch.__version__} cuda_build={torch.version.cuda}")
    print(f"[cuda] available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[cuda] gpu0={torch.cuda.get_device_name(0)}")

    root = cfg.get("data.utkface_root")
    if root is None:
        raise ValueError("Config missing 'data.utkface_root'.")
    if not Path(root).exists():
        raise FileNotFoundError(f"UTKFace root does not exist: {root}")

    bins = cfg.get("labels.bins")
    if bins is None:
        raise ValueError("Config missing 'labels.bins'.")
    bins = list(map(int, list(bins)))
    num_classes = len(bins) - 1
    if num_classes < 2:
        raise RuntimeError(f"Need >=2 classes; bins={bins}")

    image_size = int(cfg.get("cnn.image_size", cfg.get("data.image_size", 128)))

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

    y_tr = np.array([int(samples[i].y) for i in tr_idx], dtype=np.int64)
    train_classes = np.unique(y_tr)

    # --- debug: class imbalance visibility ---
    counts = np.bincount(y_tr, minlength=num_classes)
    print(f"[train] n_train={len(y_tr)} class_counts={counts.tolist()}")
    if train_classes.size < 2:
        raise RuntimeError(f"Need >=2 classes in training split; got {train_classes.tolist()}")

    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")

    occ_aug_prob = float(cfg.get("cnn.occlusion_aug_prob", 0.0))
    occ_aug_types = cfg.get("cnn.occlusion_aug_types", ["eyes", "mouth", "center"])

    crop_aug = bool(cfg.get("cnn.crop_aug", True))
    crop_scale_min = float(cfg.get("cnn.crop_scale_min", 0.90))
    crop_scale_max = float(cfg.get("cnn.crop_scale_max", 1.00))
    erasing_p = float(cfg.get("cnn.erasing_p", 0.0))
    eval_center_crop_frac = float(cfg.get("cnn.eval_center_crop_frac", 0.0))

    ds_tr = RgbFaceDataset(
        samples,
        tr_idx,
        image_size=image_size,
        occlusion_type="none",
        fill=fill,
        augment=True,
        imagenet_norm=True,
        occlusion_aug_prob=occ_aug_prob,
        occlusion_aug_types=occ_aug_types,
        crop_aug=crop_aug,
        crop_scale=(crop_scale_min, crop_scale_max),
        erasing_p=erasing_p,
        eval_center_crop_frac=0.0,
    )
    ds_va = RgbFaceDataset(
        samples,
        va_idx,
        image_size=image_size,
        occlusion_type="none",
        fill=fill,
        augment=False,
        imagenet_norm=True,
        occlusion_aug_prob=0.0,
        crop_aug=False,
        erasing_p=0.0,
        eval_center_crop_frac=eval_center_crop_frac,
    )

    batch_size = int(cfg.get("cnn.batch_size", 64))
    num_workers = int(cfg.get("cnn.num_workers", 2))

    device = _pick_device(cfg)
    pin_memory = device == "cuda"
    persistent_workers = (num_workers > 0)

    print(f"[device] selected={device}")
    if device == "cuda":
        print(f"[device] gpu={torch.cuda.get_device_name(0)}")

    g = torch.Generator()
    g.manual_seed(seed)

    prefetch_factor = int(cfg.get("cnn.prefetch_factor", 2)) if num_workers > 0 else None

    train_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    backbone = cfg.get("cnn.backbone", "resnet18")
    pretrained = bool(cfg.get("cnn.pretrained", True))
    model = build_cnn(backbone=backbone, num_classes=num_classes, pretrained=pretrained)

    epochs = int(cfg.get("cnn.epochs", 50))
    lr = float(cfg.get("cnn.lr", 3e-4))
    weight_decay = float(cfg.get("cnn.weight_decay", 1e-4))

    class_weight_kind = str(cfg.get("cnn.class_weight_kind", "none"))
    class_weights = compute_class_weights(y_tr, num_classes=num_classes, kind=class_weight_kind)
    print(
    f"[train] class_weight_kind={class_weight_kind} "
    f"class_weights={class_weights.numpy().round(3).tolist()}")

    label_smoothing = float(cfg.get("cnn.label_smoothing", 0.0))
    grad_clip = float(cfg.get("cnn.grad_clip", 1.0))
    amp = bool(cfg.get("cnn.amp", True))
    scheduler_name = str(cfg.get("cnn.scheduler", "cosine"))
    select_metric = str(cfg.get("cnn.select_metric", "acc"))

    result = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        grad_clip=grad_clip,
        use_amp=amp,
        select_metric=select_metric,
        scheduler_name=scheduler_name,
    )

    model.load_state_dict(result.best_state_dict)

    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    y_te_clean = np.array([int(samples[i].y) for i in te_idx], dtype=np.int64)
    eval_classes_union = np.unique(np.concatenate([y_tr, y_te_clean]))

    for occ in occ_types:
        ds_te = RgbFaceDataset(
            samples,
            te_idx,
            image_size=image_size,
            occlusion_type=str(occ),
            fill=fill,
            augment=False,
            imagenet_norm=True,
            occlusion_aug_prob=0.0,
            crop_aug=False,
            erasing_p=0.0,
            eval_center_crop_frac=eval_center_crop_frac,
        )
        te_loader = DataLoader(
            ds_te,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=g,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        acc, mf1, cm = eval_model(model, te_loader, device=device, num_classes=num_classes)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}
        confmats[str(occ)] = cm.tolist()
        print(f"[cnn] occ={str(occ):>6s} acc={acc:.4f} macro_f1={mf1:.4f}")

    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    res_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

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
            "imagenet_norm": True,
            "augment": True,
            "occlusion_aug_prob": occ_aug_prob,
            "occlusion_aug_types": occ_aug_types,
            "crop_aug": crop_aug,
            "crop_scale_min": crop_scale_min,
            "crop_scale_max": crop_scale_max,
            "erasing_p": erasing_p,
            "eval_center_crop_frac": eval_center_crop_frac,
            "class_weight_kind": class_weight_kind,
            "label_smoothing": label_smoothing,
            "grad_clip": grad_clip,
            "amp": amp,
            "scheduler": scheduler_name,
            "select_metric": select_metric,
            "device_request": str(cfg.get("cnn.device", "cuda")),
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
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