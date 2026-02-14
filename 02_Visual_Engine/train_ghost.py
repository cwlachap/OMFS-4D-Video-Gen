"""
train_ghost.py — Train a GaussianAvatars model from preprocessed video data.

Wraps the GaussianAvatars train.py with the correct arguments for
single-camera monocular video (typical patient selfie).

Usage:
    python 02_Visual_Engine/train_ghost.py \
        --data_dir 02_Visual_Engine/data \
        --output_dir 02_Visual_Engine/output/model \
        --iterations 30000

This script is called from app.py via subprocess.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# Path to the GaussianAvatars repo
REPO_DIR = Path(__file__).parent.parent / "gaussian_avatars_repo"
TRAIN_SCRIPT = REPO_DIR / "train.py"


def validate_setup():
    """Check that all required files/dirs exist."""
    if not REPO_DIR.exists():
        raise FileNotFoundError(
            f"GaussianAvatars repo not found at: {REPO_DIR}\n"
            "Clone it with:\n"
            "  git clone https://github.com/ShenhanQian/GaussianAvatars.git gaussian_avatars_repo"
        )
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"train.py not found at: {TRAIN_SCRIPT}\n"
            "The GaussianAvatars repo may be incomplete."
        )


def validate_data(data_dir: str):
    """Check that the preprocessed data is ready."""
    required = ["transforms_train.json", "transforms_test.json", "flame_param.npz"]
    for f in required:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run preprocess_video.py first to prepare the dataset."
            )
    images_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Run preprocess_video.py first."
        )
    n_images = len([f for f in os.listdir(images_dir) if f.endswith(".png")])
    if n_images == 0:
        raise FileNotFoundError("No PNG frames found in images directory.")
    print(f"[train_ghost] Dataset validated: {n_images} frames")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_dataset_fingerprint(data_dir: str) -> dict:
    """Build a reproducible fingerprint for the dataset used in training."""
    data_path = Path(data_dir)
    key_files = [
        "transforms_train.json",
        "transforms_test.json",
        "transforms_val.json",
        "flame_param.npz",
        "canonical_flame_param.npz",
    ]
    file_hashes = {}
    for rel in key_files:
        p = data_path / rel
        if p.exists():
            file_hashes[rel] = _sha256_file(p)

    aggregate = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {"files": file_hashes, "dataset_hash": aggregate}


def run_quality_gates(data_dir: str):
    """Run light-weight pre-train quality gates to fail fast on bad data."""
    data_path = Path(data_dir)
    train_json = data_path / "transforms_train.json"
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    frames = train_data.get("frames", [])
    if len(frames) < 50:
        raise RuntimeError(
            f"Quality gate failed: only {len(frames)} training frames; need at least 50."
        )

    # Ensure frame indices are mostly contiguous (drop-heavy datasets often regress quality).
    timestep_indices = [
        int(fr.get("timestep_index", i)) for i, fr in enumerate(frames)
    ]
    gaps = sum(
        1 for i in range(1, len(timestep_indices))
        if (timestep_indices[i] - timestep_indices[i - 1]) > 1
    )
    if gaps > max(10, len(timestep_indices) // 10):
        raise RuntimeError(
            f"Quality gate failed: too many timeline gaps in train split ({gaps})."
        )

    masks_dir = data_path / "fg_masks"
    if masks_dir.exists():
        n_masks = len([f for f in masks_dir.iterdir() if f.suffix.lower() == ".png"])
        if n_masks < len(frames) // 2:
            raise RuntimeError(
                f"Quality gate failed: only {n_masks} fg masks for {len(frames)} train frames."
            )

    print(
        "[train_ghost] Quality gates passed:"
        f" frames={len(frames)}, timeline_gaps={gaps}"
    )


def _collect_checkpoint_lineage(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return []
    checkpoints = []
    for ckpt in sorted(out.glob("chkpnt*.pth")):
        checkpoints.append(
            {
                "name": ckpt.name,
                "size_bytes": ckpt.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    ckpt.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return checkpoints


def write_experiment_manifest(
    data_dir: str,
    output_dir: str,
    iterations: int,
    resolution: int,
    cmd: list[str],
    extra: dict,
):
    out = Path(output_dir)
    manifests_dir = out / "experiment_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifests_dir / f"{now}.json"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(Path(data_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "iterations": iterations,
        "resolution": resolution,
        "command": cmd,
        "dataset_fingerprint": build_dataset_fingerprint(data_dir),
        "checkpoint_lineage": _collect_checkpoint_lineage(output_dir),
        "extra": extra,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[train_ghost] Wrote experiment manifest: {manifest_path}")
    return manifest_path


def train(
    data_dir: str,
    output_dir: str,
    iterations: int = 30000,
    resolution: int = -1,
):
    """Launch GaussianAvatars training.

    Parameters
    ----------
    data_dir : str
        Path to preprocessed dataset (with transforms_train.json etc.)
    output_dir : str
        Path to save the trained model.
    iterations : int
        Number of training iterations.
    resolution : int
        Training resolution. -1 means native resolution (recommended).
    """
    validate_setup()
    validate_data(data_dir)
    run_quality_gates(data_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Build the training command
    # Calculate save iterations - save at end and a few checkpoints along the way
    save_iters = [iterations]  # Always save at final iteration
    if iterations >= 5000:
        save_iters.insert(0, iterations // 2)  # Midpoint checkpoint
    if iterations >= 10000:
        save_iters.insert(0, iterations // 4)  # Quarter checkpoint
    
    # Check if fg_masks exist - only use white_background if we have masks
    fg_masks_dir = os.path.join(data_dir, "fg_masks")
    has_masks = os.path.isdir(fg_masks_dir) and len(os.listdir(fg_masks_dir)) > 0

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--source_path", os.path.abspath(data_dir),
        "--model_path", os.path.abspath(output_dir),
        "--bind_to_mesh",
        "--iterations", str(iterations),
        "--resolution", str(resolution),
        "--save_iterations", *[str(i) for i in save_iters],
        "--checkpoint_iterations", *[str(i) for i in save_iters],
    ]

    if has_masks:
        cmd.append("--white_background")
        print(f"[train_ghost] Using white background (fg_masks found)")
    else:
        print(f"[train_ghost] Using original background (no fg_masks)")

    write_experiment_manifest(
        data_dir=data_dir,
        output_dir=output_dir,
        iterations=iterations,
        resolution=resolution,
        cmd=cmd,
        extra={"has_masks": has_masks},
    )

    print(f"[train_ghost] Starting training:")
    print(f"  Data:       {data_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Iterations: {iterations}")
    print(f"  Command:    {' '.join(cmd)}")
    print()

    # Run training — stream output in real time
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        text=True,
        capture_output=False,  # stream to console
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed with exit code {result.returncode}."
        )

    print(f"\n[train_ghost] Training complete! Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GaussianAvatars model.")
    parser.add_argument(
        "--data_dir", type=str, default="02_Visual_Engine/data",
        help="Path to preprocessed dataset.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="02_Visual_Engine/output/model",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--iterations", type=int, default=5000,
        help="Training iterations (default 5000 for quick test, 30000 for good quality, 600000 for full).",
    )
    parser.add_argument(
        "--resolution", type=int, default=-1,
        help="Training resolution. -1 = native resolution (recommended).",
    )
    args = parser.parse_args()

    train(args.data_dir, args.output_dir, args.iterations, args.resolution)


if __name__ == "__main__":
    main()
