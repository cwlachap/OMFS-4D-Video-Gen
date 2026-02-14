"""
Build a canonical full-head asset scaffold from registered sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_canonical_head(registration_path: Path, output_dir: Path) -> tuple[Path, Path]:
    with open(registration_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    canonical = reg.get("canonical_sequence", "unknown")
    registrations = reg.get("registrations", [])

    # Placeholder canonical asset for downstream hybrid pipeline wiring.
    # The array data is intentionally minimal and acts as versioned metadata carrier.
    canonical_asset = output_dir / "canonical_head_asset.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        canonical_asset,
        version=np.array([1], dtype=np.int32),
        canonical_sequence=np.array([canonical]),
        registration_count=np.array([len(registrations)], dtype=np.int32),
    )

    manifest = {
        "canonical_sequence": canonical,
        "registration_source": str(registration_path.resolve()),
        "asset_path": str(canonical_asset.resolve()),
        "notes": "Scaffold asset for hybrid_full_head mode. Replace with dense head geometry/field in future iterations.",
    }
    manifest_path = output_dir / "canonical_head_asset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[head_recon] Wrote canonical head scaffold: {canonical_asset}")
    print(f"[head_recon] Wrote canonical asset manifest: {manifest_path}")
    return canonical_asset, manifest_path


def main():
    parser = argparse.ArgumentParser(description="Build canonical full-head scaffold asset.")
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("02_Visual_Engine/output/head_recon"),
    )
    args = parser.parse_args()
    build_canonical_head(args.registration, args.output_dir)


if __name__ == "__main__":
    main()
