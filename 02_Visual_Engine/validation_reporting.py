"""
Deterministic validation reporting for front/profile/rear fidelity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        a = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])
    if b.ndim == 3:
        b = (0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2])
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_x = a.mean()
    mu_y = b.mean()
    sig_x = ((a - mu_x) ** 2).mean()
    sig_y = ((b - mu_y) ** 2).mean()
    sig_xy = ((a - mu_x) * (b - mu_y)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return float(((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sig_x + sig_y + c2)))


def _bucket(progress: float) -> str:
    if progress < 0.20 or progress > 0.80:
        return "front"
    if 0.35 <= progress <= 0.65:
        return "profile"
    return "rear"


def _find_latest_train_dir(model_path: Path) -> Path:
    train_dir = model_path / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    dirs = [p for p in train_dir.iterdir() if p.is_dir() and p.name.startswith("ours_")]
    if not dirs:
        raise FileNotFoundError(f"No ours_* directories in {train_dir}")
    return sorted(dirs, key=lambda p: int(p.name.split("_")[-1]), reverse=True)[0]


def generate_report(model_path: Path, deterministic_frames_dir: Path, output_dir: Path):
    latest = _find_latest_train_dir(model_path)
    renders_dir = latest / "renders"
    gt_dir = latest / "gt"
    if not renders_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing renders/gt directories in {latest}")

    manifest = deterministic_frames_dir / "deterministic_indices_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing deterministic manifest: {manifest}")
    with open(manifest, "r", encoding="utf-8") as f:
        idx_manifest = json.load(f)
    rows = idx_manifest.get("exports", [])

    metrics = []
    max_index = max((int(r.get("index", 0)) for r in rows), default=1)
    for row in rows:
        idx = int(row["index"])
        src_name = row["source"]
        render_path = renders_dir / src_name
        gt_path = gt_dir / src_name
        if not render_path.exists() or not gt_path.exists():
            continue
        a = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32)
        progress = idx / max(1, max_index)
        metrics.append(
            {
                "index": idx,
                "frame": src_name,
                "progress": progress,
                "bucket": _bucket(progress),
                "psnr": psnr(a, b),
                "ssim": ssim_global(a, b),
            }
        )

    summary = {"count": len(metrics), "by_bucket": {}}
    for bucket in ("front", "profile", "rear"):
        vals = [m for m in metrics if m["bucket"] == bucket]
        if not vals:
            summary["by_bucket"][bucket] = {"count": 0, "psnr": None, "ssim": None}
            continue
        summary["by_bucket"][bucket] = {
            "count": len(vals),
            "psnr": float(np.mean([v["psnr"] for v in vals])),
            "ssim": float(np.mean([v["ssim"] for v in vals])),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "strict_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": metrics}, f, indent=2)

    checklist_path = output_dir / "human_review_checklist.md"
    checklist = """# Human Review Checklist

- [ ] Jawline continuity in profile views.
- [ ] Ear geometry plausibility in left/right profile.
- [ ] Neck-head transition remains stable across motion.
- [ ] No visible shimmer/flicker in slow turns.
- [ ] Maxilla/mandible changes remain anatomically plausible.
"""
    checklist_path.write_text(checklist, encoding="utf-8")
    print(f"[validation_reporting] Wrote strict report: {scores_path}")
    print(f"[validation_reporting] Wrote checklist: {checklist_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic validation report.")
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--deterministic_frames_dir", required=True, type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("02_Visual_Engine/output/model/eval_strict/reports"),
    )
    args = parser.parse_args()
    generate_report(args.model_path, args.deterministic_frames_dir, args.output_dir)


if __name__ == "__main__":
    main()
