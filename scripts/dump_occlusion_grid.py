from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

from src.config import Config
from src.data.splits import load_split
from src.data.utkface import discover_utkface


def load_image_bgr(path: Path, image_size: int) -> np.ndarray:
    """
    Robust color loader for Windows/OneDrive paths:
    - reads raw bytes with np.fromfile
    - decodes with cv2.imdecode
    Returns BGR uint8 image resized to (image_size, image_size).
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty file or unreadable bytes: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR uint8 or None
    if img is None:
        raise ValueError(f"cv2.imdecode failed for: {path}")
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img


def _region_box(h: int, w: int, region: str) -> tuple[int, int, int, int]:
    """
    Occlusion boxes (y0, y1, x0, x1).
    NOTE: These are reasonable defaults. If you want exact matching with your
    src.data.occlusion.occlude_region geometry, paste that function and I’ll mirror it.
    """
    if region == "none":
        return 0, 0, 0, 0

    if region == "eyes":
        y0 = int(0.25 * h)
        y1 = int(0.45 * h)
        x0 = int(0.15 * w)
        x1 = int(0.85 * w)
        return y0, y1, x0, x1

    if region == "mouth":
        y0 = int(0.60 * h)
        y1 = int(0.80 * h)
        x0 = int(0.20 * w)
        x1 = int(0.80 * w)
        return y0, y1, x0, x1

    if region == "center":
        y0 = int(0.33 * h)
        y1 = int(0.67 * h)
        x0 = int(0.33 * w)
        x1 = int(0.67 * w)
        return y0, y1, x0, x1

    raise ValueError(f"Unknown region: {region}")


def occlude_region_bgr(img_bgr: np.ndarray, region: str, fill: str = "mean") -> np.ndarray:
    """
    Apply occlusion to a BGR uint8 image.
    fill options:
      - "mean": fill with whole-image mean color
      - "black": (0,0,0)
      - "white": (255,255,255)
      - anything else: fallback to "mean"
    """
    if region == "none":
        return img_bgr

    h, w = img_bgr.shape[:2]
    y0, y1, x0, x1 = _region_box(h, w, region)
    out = img_bgr.copy()

    if fill == "black":
        color = np.array([0, 0, 0], dtype=np.uint8)
    elif fill == "white":
        color = np.array([255, 255, 255], dtype=np.uint8)
    else:
        # "mean" or fallback
        color = np.clip(out.mean(axis=(0, 1)), 0, 255).round().astype(np.uint8)

    out[y0:y1, x0:x1, :] = color.reshape(1, 1, 3)
    return out


def main():
    cfg = Config.load("configs/default.yaml")
    image_size = int(cfg.get("data.image_size", 128))
    fill = str(cfg.get("occlusion.fill", "mean"))

    root = Path(cfg.get("data.utkface_root"))
    print("[utkface_root]", root)
    print("[exists]", root.exists(), "[is_dir]", root.is_dir())

    bins = cfg.get("labels.bins")
    num_classes = len(bins) - 1  # 7 classes for 8 bin edges

    split_path = Path("outputs/splits/utkface_split.json")
    s = load_split(split_path)

    samples = discover_utkface(str(root), bins, debug=True)

    # 4 columns per row
    regions = ["none", "eyes", "mouth", "center"]

    # RNG: set to int for reproducible output; set to None for different each run
    seed = 0  # change to None if you want a new random grid every run
    rng = np.random.default_rng(seed)

    # Build pools by class for test set
    by_class_test: list[list[int]] = [[] for _ in range(num_classes)]
    for idx in s.test:
        idx = int(idx)
        if 0 <= idx < len(samples):
            y = int(samples[idx].y)
            if 0 <= y < num_classes:
                by_class_test[y].append(idx)

    # Build pools by class for full dataset (fallback)
    by_class_all: list[list[int]] = [[] for _ in range(num_classes)]
    for idx, smp in enumerate(samples):
        y = int(smp.y)
        if 0 <= y < num_classes:
            by_class_all[y].append(idx)

    # Choose an initial random index per class (prefer test, fallback all)
    chosen: list[int] = []
    for y in range(num_classes):
        pool = by_class_test[y] if len(by_class_test[y]) > 0 else by_class_all[y]
        if len(pool) == 0:
            raise RuntimeError(f"No samples found for class y={y}. Check bins / dataset filtering.")
        chosen.append(int(rng.choice(pool)))

    def try_indices(indices: list[int], max_tries: int = 200) -> tuple[np.ndarray | None, object | None]:
        """
        Try loading from a list of sample indices. Returns (img_bgr, sample) or (None, None).
        """
        tries = 0
        for alt_idx in indices:
            if tries >= max_tries:
                break
            tries += 1
            smp = samples[int(alt_idx)]
            try:
                img = load_image_bgr(Path(smp.path), image_size=image_size)
                return img, smp
            except Exception:
                continue
        return None, None

    rows = []
    unreadable_bins = []

    for y in range(num_classes):
        test_pool = by_class_test[y]
        all_pool = by_class_all[y]

        # Primary attempt order: chosen -> shuffled test pool -> shuffled all pool
        order: list[int] = []

        if chosen[y] is not None:
            order.append(int(chosen[y]))

        if len(test_pool) > 0:
            tmp = test_pool.copy()
            rng.shuffle(tmp)
            order.extend(tmp)

        # Ensure we also try the full pool (in case the test pool for that bin is all unreadable)
        if len(all_pool) > 0:
            tmp = all_pool.copy()
            rng.shuffle(tmp)
            order.extend(tmp)

        # Deduplicate while preserving order
        seen = set()
        order_dedup = []
        for idx in order:
            if idx in seen:
                continue
            seen.add(idx)
            order_dedup.append(idx)

        img, smp = try_indices(order_dedup, max_tries=300)
        if img is None or smp is None:
            unreadable_bins.append(y)

            # Print a few diagnostics
            print(f"\n[ERROR] No readable image for class y={y}. Some candidates:")
            for cand in order_dedup[:10]:
                p = Path(samples[int(cand)].path)
                try:
                    size = p.stat().st_size
                except Exception:
                    size = None
                print(f"  - {p} | exists={p.exists()} | size={size}")

            raise RuntimeError(
                f"Could not load any readable image for class y={y}.\n"
                "Most likely causes:\n"
                "  (1) UTKFace files are OneDrive cloud-only placeholders (not downloaded)\n"
                "  (2) data.utkface_root points to the wrong folder\n"
                "Fix: Right-click UTKFace folder -> OneDrive -> 'Always keep on this device',\n"
                "and verify configs/default.yaml data.utkface_root."
            )

        cols = []
        for r in regions:
            o = occlude_region_bgr(img, region=r, fill=fill)  # BGR uint8
            cols.append(o)

        row = np.concatenate(cols, axis=1)  # (H, 4W, 3)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)  # (7H, 4W, 3)

    out_dir = Path("outputs/occlusion_examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "occlusion_grid_color_random_per_bin.png"
    cv2.imwrite(str(out_path), grid)

    print(f"\nSaved: {out_path.resolve()}")
    print(f"Rows: {len(rows)} (expected {num_classes})")
    print(f"Seed: {seed}")
    if unreadable_bins:
        print(f"Unreadable bins encountered (should be empty if saved): {unreadable_bins}")


if __name__ == "__main__":
    main()