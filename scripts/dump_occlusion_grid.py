from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

from src.config import Config
from src.data.splits import load_split
from src.data.utkface import discover_utkface, load_image_gray
from src.data.occlusion import occlude_region


def to_u8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0).round().astype(np.uint8)


def main():
    cfg = Config.load("configs/default.yaml")
    image_size = int(cfg.get("data.image_size", 128))
    fill = str(cfg.get("occlusion.fill", "mean"))

    split_path = Path("outputs/splits/utkface_split.json")
    s = load_split(split_path)

    root = cfg.get("data.utkface_root")
    bins = cfg.get("labels.bins")
    samples = discover_utkface(root, bins, debug=True)

    # Try to collect N examples even if a few are unreadable (OneDrive placeholders etc.)
    target = 8
    regions = ["none", "eyes", "mouth", "center"]

    rows = []
    skipped = 0

    for i in s.test:
        if len(rows) >= target:
            break

        sample = samples[int(i)]
        try:
            img = load_image_gray(sample.path, image_size=image_size)
        except Exception as e:
            skipped += 1
            # Print a short message and continue
            print(f"[skip] unreadable: {sample.path.name}")
            continue

        cols = []
        for r in regions:
            o = occlude_region(img, region=r, fill=fill)
            cols.append(to_u8(o))

        row = np.concatenate(cols, axis=1)  # (H, 4W)
        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError(
            "Could not load any images for the grid.\n"
            "Most likely they are OneDrive cloud-only placeholders.\n"
            "Fix: Right-click UTKFace folder -> OneDrive -> 'Always keep on this device'."
        )

    grid = np.concatenate(rows, axis=0)

    out_dir = Path("outputs/occlusion_examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "occlusion_grid.png"
    cv2.imwrite(str(out_path), grid)

    print(f"Saved: {out_path.resolve()}")
    print(f"Loaded rows: {len(rows)} (skipped unreadable: {skipped})")


if __name__ == "__main__":
    main()