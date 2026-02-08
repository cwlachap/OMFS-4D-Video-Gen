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
import os
import subprocess
import sys
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


def train(
    data_dir: str,
    output_dir: str,
    iterations: int = 30000,
    resolution: int = 512,
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
        Training resolution.
    """
    validate_setup()
    validate_data(data_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Build the training command
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--source_path", os.path.abspath(data_dir),
        "--model_path", os.path.abspath(output_dir),
        "--bind_to_mesh",
        "--iterations", str(iterations),
        "--resolution", str(resolution),
        "--white_background",
    ]

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
        "--iterations", type=int, default=30000,
        help="Training iterations (default 30000 for quick test, 600000 for full).",
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Training resolution.",
    )
    args = parser.parse_args()

    train(args.data_dir, args.output_dir, args.iterations, args.resolution)


if __name__ == "__main__":
    main()
