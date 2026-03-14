from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("outputs/results")
FIG_DIR = Path("outputs/figures")


# Pretty names + stable plotting order (your 5 methods)
METHOD_ORDER = ["pca_gaussian", "lda_gaussian", "kpca_svm", "mlp", "cnn"]
METHOD_PRETTY = {
    "pca_gaussian": "PCA",
    "lda_gaussian": "LDA",
    "kpca_svm": "KPCA+SVM",
    "mlp": "MLP",
    "cnn": "CNN (ResNet-18)",
}

OCC_ORDER = ["none", "eyes", "mouth", "center"]
OCC_PRETTY = {"none": "Clean", "eyes": "Eyes", "mouth": "Mouth", "center": "Center"}

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_result_jsons(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {results_dir.resolve()}")
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files found in: {results_dir.resolve()}")
    return paths


def _extract_method_id(obj: Dict[str, Any], path: Path) -> str:
    m = obj.get("method", None)
    if isinstance(m, str) and m.strip():
        return m.strip()
    # fallback: use filename stem
    return path.stem


def _extract_metrics(obj: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Expected format:
      obj["metrics"][occ]["acc"], obj["metrics"][occ]["macro_f1"]
    """
    metrics = obj.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("JSON missing 'metrics' dict.")
    out: Dict[str, Dict[str, float]] = {}
    for occ, d in metrics.items():
        if not isinstance(d, dict):
            continue
        if "acc" in d:
            out[str(occ)] = {
                "acc": float(d.get("acc", float("nan"))),
                "macro_f1": float(d.get("macro_f1", float("nan"))),
            }
    return out


def _extract_confmat(obj: Dict[str, Any], occ: str = "none") -> np.ndarray | None:
    cms = obj.get("confusion_matrices", None)
    if not isinstance(cms, dict):
        return None
    cm = cms.get(occ, None)
    if cm is None:
        return None
    arr = np.array(cm, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


def _sort_methods_present(methods_present: List[str]) -> List[str]:
    # keep your preferred order; unknowns go at end
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(methods_present, key=lambda m: (rank.get(m, 10_000), m))


def _pretty_method(m: str) -> str:
    return METHOD_PRETTY.get(m, m)


def plot_clean_bar(
    results: Dict[str, Dict[str, Dict[str, float]]],
    fig_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    """
    Bar plot for clean ('none') metric across methods.
    results[method]['none'][metric]
    """
    methods = _sort_methods_present([m for m in results.keys() if "none" in results[m]])
    if not methods:
        raise ValueError("No methods with clean ('none') metrics found.")

    vals = [results[m]["none"].get(metric, float("nan")) for m in methods]
    labels = [_pretty_method(m) for m in methods]

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / filename

    plt.figure(figsize=(8.5, 4.8))
    x = np.arange(len(methods))
    plt.bar(x, vals)
    for xi, v in zip(x, vals):
        if not np.isnan(v):
            plt.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    plt.xticks(x, labels, rotation=0)
    plt.ylim(0.3, 0.7)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_robustness_line(
    results: Dict[str, Dict[str, Dict[str, float]]],
    fig_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    """
    Line plot: metric vs occlusion type for each method.
    """
    methods = _sort_methods_present(list(results.keys()))
    if not methods:
        raise ValueError("No methods found in results.")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / filename

    plt.figure(figsize=(9.0, 5.2))
    x = np.arange(len(OCC_ORDER))
    xticks = [OCC_PRETTY[o] for o in OCC_ORDER]

    for m in methods:
        ys = []
        for occ in OCC_ORDER:
            if occ in results[m]:
                ys.append(results[m][occ].get(metric, float("nan")))
            else:
                ys.append(np.nan)
        lw = 3 if m == "cnn" else 2
        plt.plot(x, ys, marker="o", linewidth=lw, label=_pretty_method(m))

    plt.xticks(x, xticks)
    plt.ylim(0.3, 0.7)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_robustness_drop(
    results: Dict[str, Dict[str, Dict[str, float]]],
    fig_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    """
    Bar plot: absolute drop from clean for each occlusion (eyes/mouth/center).
    Uses: drop = clean_metric - occ_metric
    """
    methods = _sort_methods_present([m for m in results.keys() if "none" in results[m]])
    if not methods:
        raise ValueError("No methods with clean ('none') metrics found.")

    occs = [o for o in OCC_ORDER if o != "none"]

    drops = np.zeros((len(methods), len(occs)), dtype=float)
    for i, m in enumerate(methods):
        clean = results[m]["none"].get(metric, float("nan"))
        for j, occ in enumerate(occs):
            if occ in results[m]:
                drops[i, j] = clean - results[m][occ].get(metric, float("nan"))
            else:
                drops[i, j] = np.nan

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / filename

    plt.figure(figsize=(9.5, 5.2))
    x = np.arange(len(methods))
    width = 0.22

    for j, occ in enumerate(occs):
        plt.bar(x + (j - 1) * width, drops[:, j], width=width, label=f"Drop: {OCC_PRETTY[occ]}")

    plt.xticks(x, [_pretty_method(m) for m in methods], rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    out_path: Path,
    class_names: List[str] | None = None,
) -> None:
    plt.figure(figsize=(7.8, 6.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    n = cm.shape[0]
    ticks = np.arange(n)
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate counts (small)
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            if v != 0:
                plt.text(j, i, str(v), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _bins_to_class_names(bins: Any) -> List[str] | None:
    if not isinstance(bins, list) or len(bins) < 2:
        return None
    try:
        b = list(map(int, bins))
        names: List[str] = []
        for i in range(len(b) - 1):
            lo, hi = b[i], b[i + 1]
            if i == len(b) - 2:
                names.append(f"{lo}+")
            else:
                names.append(f"{lo}–{hi-1}")
        return names
    except Exception:
        return None


def main() -> None:
    paths = _find_result_jsons(RESULTS_DIR)

    # results[method][occ] = {"acc":..., "macro_f1":...}
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    raw_by_method: Dict[str, Dict[str, Any]] = {}

    for p in paths:
        obj = _read_json(p)
        method = _extract_method_id(obj, p)
        try:
            metrics = _extract_metrics(obj)
        except Exception as e:
            print(f"[skip] {p.name}: failed to parse metrics ({e})")
            continue
        results[method] = metrics
        raw_by_method[method] = obj

    if not results:
        raise RuntimeError("No usable results parsed from JSON files.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Accuracy figures -----
    p1 = plot_clean_bar(
        results,
        FIG_DIR,
        metric="acc",
        title="Clean Accuracy (UTKFace 7-bin)",
        ylabel="Accuracy",
        filename="clean_accuracy.png",
    )
    print(f"Saved: {p1}")

    p2 = plot_robustness_line(
        results,
        FIG_DIR,
        metric="acc",
        title="Accuracy Under Structured Occlusion",
        ylabel="Accuracy",
        filename="robustness_accuracy.png",
    )
    print(f"Saved: {p2}")

    p3 = plot_robustness_drop(
        results,
        FIG_DIR,
        metric="acc",
        title="Occlusion Sensitivity (Absolute Drop from Clean)",
        ylabel="Accuracy Drop (Clean - Occluded)",
        filename="robustness_drop.png",
    )
    print(f"Saved: {p3}")

    # ----- Macro-F1 figures -----
    f1_1 = plot_clean_bar(
        results,
        FIG_DIR,
        metric="macro_f1",
        title="Clean Macro-F1 (UTKFace 7-bin)",
        ylabel="Macro-F1",
        filename="clean_macro_f1.png",
    )
    print(f"Saved: {f1_1}")

    f1_2 = plot_robustness_line(
        results,
        FIG_DIR,
        metric="macro_f1",
        title="Macro-F1 Under Structured Occlusion",
        ylabel="Macro-F1",
        filename="robustness_macro_f1.png",
    )
    print(f"Saved: {f1_2}")

    f1_3 = plot_robustness_drop(
        results,
        FIG_DIR,
        metric="macro_f1",
        title="Macro-F1 Sensitivity (Absolute Drop from Clean)",
        ylabel="Macro-F1 Drop (Clean - Occluded)",
        filename="robustness_drop_macro_f1.png",
    )
    print(f"Saved: {f1_3}")

    # ----- Confusion matrices for CNN -----
    if "cnn" in raw_by_method:
        bins = raw_by_method["cnn"].get("bins", None)
        class_names = _bins_to_class_names(bins)

        # Clean confusion
        cm_clean = _extract_confmat(raw_by_method["cnn"], occ="none")
        if cm_clean is not None:
            out_cm = FIG_DIR / "confusion_cnn_none.png"
            plot_confusion_matrix(
                cm_clean,
                title="CNN Confusion Matrix (Clean)",
                out_path=out_cm,
                class_names=class_names,
            )
            print(f"Saved: {out_cm}")
        else:
            print("[info] cnn present but no confusion matrix found for occ='none'.")

        # Eyes occlusion confusion  ✅ NEW
        cm_eyes = _extract_confmat(raw_by_method["cnn"], occ="eyes")
        if cm_eyes is not None:
            out_cm_eyes = FIG_DIR / "confusion_cnn_eyes.png"
            plot_confusion_matrix(
                cm_eyes,
                title="CNN Confusion Matrix (Eyes Occlusion)",
                out_path=out_cm_eyes,
                class_names=class_names,
            )
            print(f"Saved: {out_cm_eyes}")
        else:
            print("[info] cnn present but no confusion matrix found for occ='eyes'.")

    print(f"\nAll figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()