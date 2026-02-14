"""
Evaluate coarse front/profile/rear coverage from frame indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_head_coverage(n_frames: int) -> dict:
    if n_frames <= 0:
        return {"front": 0, "profile": 0, "rear": 0, "n_frames": 0}

    # Deterministic temporal buckets. This is a scaffold metric useful for repeatable reporting.
    front = 0
    profile = 0
    rear = 0
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        if t < 0.20 or t > 0.80:
            front += 1
        elif 0.35 <= t <= 0.65:
            profile += 1
        else:
            rear += 1
    return {"front": front, "profile": profile, "rear": rear, "n_frames": n_frames}


def main():
    parser = argparse.ArgumentParser(description="Compute coarse head coverage report.")
    parser.add_argument("--transforms", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("02_Visual_Engine/output/head_recon/head_coverage.json"),
    )
    args = parser.parse_args()

    with open(args.transforms, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_frames = len(data.get("frames", []))
    report = evaluate_head_coverage(n_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[head_recon] Wrote coverage report: {args.output}")


if __name__ == "__main__":
    main()
