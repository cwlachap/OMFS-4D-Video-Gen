"""
train_ghost.py — Train a GaussianAvatars model from a monocular video.

This script is called from the omfs_visual conda environment via subprocess.

Usage:
    python 02_Visual_Engine/train_ghost.py --data_dir 02_Visual_Engine/data --output_dir 02_Visual_Engine/output
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Train GaussianAvatars model from video data."
    )
    parser.add_argument(
        "--data_dir", type=str, default="02_Visual_Engine/data",
        help="Path to preprocessed video data directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="02_Visual_Engine/output",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--iterations", type=int, default=30000,
        help="Number of training iterations.",
    )
    args = parser.parse_args()

    # --- Validate inputs ---
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            "Please place your preprocessed video data there first."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Import training modules ---
    try:
        from scene import GaussianModel
        from gaussian_renderer import render
    except ImportError as exc:
        raise ImportError(
            "Cannot import GaussianAvatars modules. "
            "Ensure the scene/ and gaussian_renderer/ submodules are "
            "installed in the omfs_visual environment.\n"
            f"Details: {exc}"
        ) from exc

    print(f"[train_ghost] Data directory : {args.data_dir}")
    print(f"[train_ghost] Output directory: {args.output_dir}")
    print(f"[train_ghost] Iterations     : {args.iterations}")
    print("[train_ghost] Starting training …")

    # ─── Training loop ────────────────────────────────────────
    # This is a scaffold. The real training loop integrates with
    # the GaussianAvatars codebase (scene, cameras, optimiser).
    # Implement here once the gaussian_renderer submodule is set up.
    # ──────────────────────────────────────────────────────────

    raise NotImplementedError(
        "Training logic requires the GaussianAvatars submodule. "
        "Please clone and install gaussian_renderer/ and scene/ "
        "into 02_Visual_Engine/, then implement the training loop."
    )


if __name__ == "__main__":
    main()
