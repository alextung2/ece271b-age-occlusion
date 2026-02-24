from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.utils import set_seed
from src.data.utkface import discover_utkface
from src.data.splits import make_split, save_split


def main():
    cfg = Config.load("configs/default.yaml")
    seed = int(cfg.get("seed", 271))
    set_seed(seed)

    root = cfg.get("data.utkface_root")
    bins = cfg.get("labels.bins")
    min_age = cfg.get("labels.min_age", None)
    max_age = cfg.get("labels.max_age", None)

    samples = discover_utkface(
        root=root,
        bins=bins,
        min_age=min_age,
        max_age=max_age,
        debug=True,
    )

    split = make_split(
        n=len(samples),
        train_frac=float(cfg.get("data.split.train")),
        val_frac=float(cfg.get("data.split.val")),
        seed=seed,
    )

    out_dir = Path("outputs/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "utkface_split.json"
    save_split(split, out_path)
    print(f"Saved split to {out_path} with N={len(samples)}")


if __name__ == "__main__":
    main()