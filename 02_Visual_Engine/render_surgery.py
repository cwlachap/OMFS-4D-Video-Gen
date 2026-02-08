"""
render_surgery.py — Render a post-surgical prediction video.

Takes surgical movement values (mm) from the planning tab,
converts them to FLAME parameter offsets, modifies the FLAME
params in the dataset, and renders the prediction using
GaussianAvatars.

CLI Arguments:
    --lefort_mm   : Le Fort I advancement in mm
    --bsso_mm     : BSSO advancement in mm
    --sensitivity : Multiplier for clinical-to-FLAME mapping
    --model_path  : Path to trained GaussianAvatars model
    --data_dir    : Path to preprocessed dataset
    --output      : Output video path

This script is called from app.py via subprocess.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SCALE_FACTOR = 0.001  # mm → FLAME internal units
REPO_DIR = Path(__file__).parent.parent / "gaussian_avatars_repo"
RENDER_SCRIPT = REPO_DIR / "render.py"


def compute_offset(input_mm: float, sensitivity: float) -> float:
    """Convert clinical mm to FLAME-space offset."""
    return input_mm * sensitivity * SCALE_FACTOR


def _get_ffmpeg_path() -> str:
    """Return the path to ffmpeg — bundled or system."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise FileNotFoundError(
        "ffmpeg not found. Install via: pip install imageio-ffmpeg"
    )


def modify_flame_params(
    source_npz: str,
    output_npz: str,
    lefort_offset: float,
    bsso_offset: float,
) -> None:
    """Create a modified FLAME parameter file with surgical offsets.

    Le Fort I → shifts translation along Y (anterior advancement).
    BSSO → modifies jaw_pose[0] (X rotation = jaw opening/advancement).

    Parameters
    ----------
    source_npz : str
        Original FLAME parameter file.
    output_npz : str
        Path to save modified parameters.
    lefort_offset : float
        Offset in FLAME units for maxilla advancement.
    bsso_offset : float
        Offset in FLAME units for mandible advancement.
    """
    data = dict(np.load(source_npz, allow_pickle=True))

    # Le Fort I: translate maxilla anteriorly (Y-axis in FLAME space)
    if "translation" in data:
        trans = data["translation"].copy()
        trans[:, 1] += lefort_offset  # Y = anterior
        data["translation"] = trans

    # BSSO: modify jaw pose (X rotation = jaw opening/advancement)
    if "jaw_pose" in data:
        jaw = data["jaw_pose"].copy()
        jaw[:, 0] += bsso_offset  # X rotation
        data["jaw_pose"] = jaw

    np.savez(output_npz, **data)
    print(f"[render_surgery] Modified FLAME params saved: {output_npz}")
    print(f"  Le Fort offset (translation Y): {lefort_offset:.6f}")
    print(f"  BSSO offset (jaw_pose X):       {bsso_offset:.6f}")


def create_modified_dataset(
    data_dir: str,
    lefort_offset: float,
    bsso_offset: float,
) -> str:
    """Create a temporary dataset with modified FLAME parameters.

    Copies the original dataset structure but replaces flame_param.npz
    with modified params. Returns the path to the temp dataset.
    """
    temp_dir = tempfile.mkdtemp(prefix="surgical_render_")

    # Copy images (symlink to save disk space, fallback to copy)
    src_images = os.path.join(data_dir, "images")
    dst_images = os.path.join(temp_dir, "images")
    try:
        os.symlink(os.path.abspath(src_images), dst_images, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(src_images, dst_images)

    # Modify FLAME params
    src_flame = os.path.join(data_dir, "flame_param.npz")
    dst_flame = os.path.join(temp_dir, "flame_param.npz")
    modify_flame_params(src_flame, dst_flame, lefort_offset, bsso_offset)

    # Copy and update transforms JSONs
    for json_name in ("transforms_train.json", "transforms_test.json"):
        src_json = os.path.join(data_dir, json_name)
        dst_json = os.path.join(temp_dir, json_name)
        if os.path.exists(src_json):
            with open(src_json, "r") as f:
                transforms = json.load(f)
            # Update flame_param paths to point to the modified file
            for frame in transforms.get("frames", []):
                frame["flame_param_path"] = "flame_param.npz"
            with open(dst_json, "w") as f:
                json.dump(transforms, f, indent=2)

    print(f"[render_surgery] Modified dataset at: {temp_dir}")
    return temp_dir


def render_with_gaussians(
    model_path: str,
    data_dir: str,
    output_dir: str,
    iteration: int = -1,
) -> str:
    """Run GaussianAvatars render.py to produce frames.

    Returns path to the rendered frames directory.
    """
    if not RENDER_SCRIPT.exists():
        raise FileNotFoundError(
            f"GaussianAvatars render.py not found at: {RENDER_SCRIPT}"
        )

    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--source_path", os.path.abspath(data_dir),
        "--model_path", os.path.abspath(model_path),
        "--bind_to_mesh",
        "--skip_val",
        "--skip_test",
    ]
    if iteration > 0:
        cmd.extend(["--iteration", str(iteration)])

    print(f"[render_surgery] Rendering with GaussianAvatars...")
    print(f"  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Rendering failed:\n{result.stderr[-2000:]}"
        )

    # Find the rendered frames
    # GaussianAvatars outputs to model_path/train/ours_XXXXX/renders/
    renders_dir = None
    train_dir = os.path.join(model_path, "train")
    if os.path.isdir(train_dir):
        for d in sorted(os.listdir(train_dir), reverse=True):
            renders = os.path.join(train_dir, d, "renders")
            if os.path.isdir(renders):
                renders_dir = renders
                break

    if renders_dir is None:
        raise FileNotFoundError(
            "No rendered frames found after GaussianAvatars rendering."
        )

    print(f"[render_surgery] Frames rendered to: {renders_dir}")
    return renders_dir


def stitch_video(frames_dir: str, output_path: str, fps: int = 30):
    """Stitch PNG frames into H.264 MP4 using ffmpeg."""
    ffmpeg_bin = _get_ffmpeg_path()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get list of frame files
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not frames:
        raise FileNotFoundError(f"No PNG frames in {frames_dir}")

    # Rename frames to sequential pattern for ffmpeg
    temp_frames = tempfile.mkdtemp(prefix="stitch_")
    for i, fname in enumerate(frames):
        src = os.path.join(frames_dir, fname)
        dst = os.path.join(temp_frames, f"frame_{i:05d}.png")
        shutil.copy2(src, dst)

    pattern = os.path.join(temp_frames, "frame_%05d.png")
    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(temp_frames, ignore_errors=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    print(f"[render_surgery] Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render post-surgical prediction video."
    )
    parser.add_argument("--lefort_mm", type=float, required=True)
    parser.add_argument("--bsso_mm", type=float, required=True)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--model_path", type=str, default="02_Visual_Engine/output/model")
    parser.add_argument("--data_dir", type=str, default="02_Visual_Engine/data")
    parser.add_argument("--output", type=str, default="final_prediction.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Compute FLAME offsets
    lefort_offset = compute_offset(args.lefort_mm, args.sensitivity)
    bsso_offset = compute_offset(args.bsso_mm, args.sensitivity)

    print(f"[render_surgery] Le Fort: {args.lefort_mm} mm → offset {lefort_offset:.6f}")
    print(f"[render_surgery] BSSO:    {args.bsso_mm} mm → offset {bsso_offset:.6f}")

    # Create modified dataset with surgical offsets
    modified_dir = create_modified_dataset(
        args.data_dir, lefort_offset, bsso_offset
    )

    try:
        # Render frames
        frames_dir = render_with_gaussians(
            args.model_path, modified_dir
        )

        # Stitch into video
        stitch_video(frames_dir, args.output, fps=args.fps)

    finally:
        # Clean up temp dataset
        shutil.rmtree(modified_dir, ignore_errors=True)

    print("[render_surgery] Done.")


if __name__ == "__main__":
    main()
