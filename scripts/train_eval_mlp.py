from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
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
    """Per-sample normalization: (x - mean)/std."""
    mu = x.mean()
    sd = x.std(unbiased=False).clamp_min(eps)
    return (x - mu) / sd


# ============================================================
# ✨ NEW PLOTTING FUNCTIONS FOR POWERPOINT
# ============================================================

def plot_training_curves(history: Dict[str, List[float]], save_path: Path):
    """Generates Loss and Accuracy curves to show the model learning."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Curve
    ax1.plot(history['train_loss'], label='Train Loss', color='#2ecc71', lw=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='#e74c3c', lw=2)
    ax1.set_title('Training vs Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy Curve
    ax2.plot(history['train_acc'], label='Train Acc', color='#2ecc71', lw=2)
    ax2.plot(history['val_acc'], label='Val Acc', color='#e74c3c', lw=2)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix_sns(cm: np.ndarray, labels: List[str], title: str, save_path: Path):
    """Generates a professional heatmap for the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {title}', fontsize=16)
    plt.ylabel('True Age Bin', fontsize=12)
    plt.xlabel('Predicted Age Bin', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_occlusion_robustness(metrics: Dict[str, Dict[str, float]], save_path: Path):
    """Shows how performance degrades under different facial occlusions."""
    labels = list(metrics.keys())
    accs = [m['acc'] for m in metrics.values()]
    f1s = [m['macro_f1'] for m in metrics.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, accs, width, label='Accuracy', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, f1s, width, label='Macro-F1', color='#9b59b6', alpha=0.8)

    ax.set_title('Model Performance Under Various Occlusions', fontsize=16)
    ax.set_ylabel('Score (0.0 - 1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============================================================


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
        img = load_image_gray(self.samples[i].path, image_size=self.image_size)

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
    image_size = int(cfg.get("data.image_size", 128))
    bins = cfg.get("labels.bins")
    
    # Use explicit bin names for cleaner graph labels
    bin_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]

    split_path = Path("outputs/splits/utkface_split.json")
    split = load_split(str(split_path))
    samples = discover_utkface(root, bins)

    tr_idx = _to_int_list(split.train)
    va_idx = _to_int_list(split.val)
    te_idx = _to_int_list(split.test)

    y_tr = np.array([int(samples[i].y) for i in tr_idx], dtype=np.int64)
    num_classes = int(y_tr.max()) + 1

    occ_types = cfg.get("occlusion.types", ["none", "eyes", "mouth", "center"])
    fill = cfg.get("occlusion.fill", "mean")

    batch_size = int(cfg.get("mlp.batch_size", 128))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_tr = FlatFaceDataset(samples, tr_idx, image_size=image_size, occlusion_type="random", fill=fill)
    ds_va = FlatFaceDataset(samples, va_idx, image_size=image_size, occlusion_type="none", fill=fill)

    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    class_w = counts.sum() / np.clip(counts, 1.0, None)
    sample_w = class_w[y_tr]

    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), len(sample_w), replacement=True)

    train_loader = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=(device == "cuda"))
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---- Model ----
    model = build_mlp(
        input_dim=image_size * image_size,
        num_classes=num_classes,
        hidden_sizes=cfg.get("mlp.hidden_sizes", [512, 256]),
        dropout=float(cfg.get("mlp.dropout", 0.4)),
        activation=cfg.get("mlp.activation", "relu"),
        use_batchnorm=bool(cfg.get("mlp.use_batchnorm", False)),
    )

    # ---- Train ----
    result = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg.get("mlp.epochs", 80)),
        lr=float(cfg.get("mlp.lr", 0.001)),
        weight_decay=float(cfg.get("mlp.weight_decay", 0.0001)),
        device=device,
    )

    # Plot Training Curves
    out_dir = Path("outputs")
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(result, 'history'):
        plot_training_curves(result.history, res_dir / "mlp_training_curves.png")

    model.load_state_dict(result.best_state_dict)

    # ---- Evaluate & Plot Confusion Matrices ----
    metrics: Dict[str, Dict[str, float]] = {}
    confmats: Dict[str, List[List[int]]] = {}

    for occ in occ_types:
        ds_te = FlatFaceDataset(samples, te_idx, image_size=image_size, occlusion_type=str(occ), fill=fill)
        te_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        acc, mf1, cm = eval_model(model, te_loader, device=device, num_classes=num_classes)
        metrics[str(occ)] = {"acc": float(acc), "macro_f1": float(mf1)}
        confmats[str(occ)] = cm.tolist()
        
        # Save a heatmap for each occlusion type
        plot_confusion_matrix_sns(cm, bin_labels, f"Occlusion={occ}", res_dir / f"cm_{occ}.png")

    # Save summary bar chart
    plot_occlusion_robustness(metrics, res_dir / "robustness_comparison.png")

    # ---- Standard Saving ----
    save_json(metrics, res_dir / "mlp_metrics.json")
    print(f"✅ All graphs saved to {res_dir}")


if __name__ == "__main__":
    main()