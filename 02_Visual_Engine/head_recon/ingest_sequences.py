"""
Ingest multi-sequence captures for full-head reconstruction scaffolding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _count_frames(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return len([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")])


def ingest_sequences(capture_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = []
    for seq_dir in sorted([p for p in capture_root.iterdir() if p.is_dir()]):
        transforms = seq_dir / "transforms_train.json"
        images = seq_dir / "images"
        if not transforms.exists() and not images.exists():
            continue
        sequences.append(
            {
                "name": seq_dir.name,
                "path": str(seq_dir.resolve()),
                "transforms_train": str(transforms.resolve()) if transforms.exists() else "",
                "image_count": _count_frames(images),
            }
        )

    manifest = {
        "capture_root": str(capture_root.resolve()),
        "sequence_count": len(sequences),
        "sequences": sequences,
    }
    out_path = output_dir / "sequence_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[head_recon] Wrote sequence manifest: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Ingest multi-sequence captures.")
    parser.add_argument("--capture_root", required=True, type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("02_Visual_Engine/output/head_recon"),
    )
    args = parser.parse_args()
    ingest_sequences(args.capture_root, args.output_dir)


if __name__ == "__main__":
    main()
