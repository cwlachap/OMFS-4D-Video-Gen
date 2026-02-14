"""
Build a lightweight registration table for multiple capture sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def register_sequences(manifest_path: Path, output_dir: Path) -> Path:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sequences = manifest.get("sequences", [])
    if not sequences:
        raise RuntimeError("No sequences found in manifest.")

    # Placeholder alignment policy:
    # we choose the first sequence as canonical and mark others with identity transform.
    canonical_name = sequences[0]["name"]
    registrations = []
    for seq in sequences:
        registrations.append(
            {
                "sequence": seq["name"],
                "to_canonical_transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "confidence": 1.0 if seq["name"] == canonical_name else 0.7,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "registration.json"
    payload = {
        "canonical_sequence": canonical_name,
        "manifest_path": str(manifest_path.resolve()),
        "registrations": registrations,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[head_recon] Wrote registration: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Register capture sequences to canonical frame.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("02_Visual_Engine/output/head_recon"),
    )
    args = parser.parse_args()
    register_sequences(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
