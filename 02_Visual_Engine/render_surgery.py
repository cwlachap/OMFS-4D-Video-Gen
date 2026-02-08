"""
render_surgery.py — Render a post-surgical prediction video using
Gaussian Splatting + FLAME deformation.

CLI Arguments:
    --lefort_mm   : Le Fort I advancement in mm  (float)
    --bsso_mm     : BSSO advancement in mm       (float)
    --sensitivity : Multiplier for clinical→visual mapping (float, default 1.0)
    --model_path  : Path to trained Gaussian model checkpoint
    --data_dir    : Path to input video data directory
    --output      : Output video path (default: final_prediction.mp4)

This script is called from the omfs_visual conda environment via subprocess.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SCALE_FACTOR = 0.001  # mm → FLAME internal units


def compute_offset(input_mm: float, sensitivity: float) -> float:
    """Convert a clinical mm value to a FLAME-space offset.

    Parameters
    ----------
    input_mm    : displacement in millimetres from the surgical plan.
    sensitivity : user-tuneable multiplier (1.0 = no scaling).

    Returns
    -------
    float : offset in FLAME internal units.
    """
    return input_mm * sensitivity * SCALE_FACTOR


def load_gaussian_model(model_path: str):
    """Load a trained 3D Gaussian Splatting model.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint directory.

    Returns
    -------
    model : The loaded Gaussian model object.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint does not exist.
    ImportError
        If the gaussian_renderer package is not available.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {model_path}\n"
            "Please train the model first using train_ghost.py."
        )

    try:
        from scene import GaussianModel
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'scene.GaussianModel'. "
            "Ensure the GaussianAvatars scene/ submodule is installed "
            "in the omfs_visual environment."
        ) from exc

    model = GaussianModel(sh_degree=3)
    model.load_ply(os.path.join(model_path, "point_cloud.ply"))
    return model


def get_flame_params(data_dir: str, frame_idx: int) -> dict:
    """Retrieve FLAME parameters for a specific frame.

    Parameters
    ----------
    data_dir  : Root data directory containing FLAME param files.
    frame_idx : Zero-based frame index.

    Returns
    -------
    dict with keys: jaw_pose (np.ndarray), translation (np.ndarray),
                    expression, shape, etc.
    """
    params_dir = os.path.join(data_dir, "flame_params")
    param_file = os.path.join(params_dir, f"frame_{frame_idx:05d}.npz")

    if not os.path.exists(param_file):
        raise FileNotFoundError(
            f"FLAME parameters not found: {param_file}"
        )

    data = np.load(param_file, allow_pickle=True)
    return {
        "jaw_pose": data["jaw_pose"].copy(),
        "translation": data["translation"].copy(),
        "expression": data["expression"].copy(),
        "shape": data["shape"].copy(),
    }


def apply_surgical_offsets(
    flame_params: dict,
    lefort_offset: float,
    bsso_offset: float,
) -> dict:
    """Modify FLAME params to reflect surgical movements.

    - Le Fort I  → shifts translation along Y (anterior)
    - BSSO       → modifies jaw_pose to simulate mandibular advancement

    Parameters
    ----------
    flame_params  : dict of FLAME parameter arrays.
    lefort_offset : offset in FLAME units for Le Fort I.
    bsso_offset   : offset in FLAME units for BSSO.

    Returns
    -------
    dict : modified FLAME parameters.
    """
    modified = {k: v.copy() for k, v in flame_params.items()}

    # Le Fort I: translate the maxilla anteriorly (Y-axis in FLAME space)
    modified["translation"][1] += lefort_offset

    # BSSO: open/close jaw — jaw_pose is a 3D rotation vector;
    # rotating about X-axis simulates mandibular advancement.
    modified["jaw_pose"][0] += bsso_offset

    return modified


def render_frame(model, flame_params: dict, frame_idx: int, output_dir: str):
    """Deform Gaussians with FLAME params and render one frame.

    Parameters
    ----------
    model        : Loaded GaussianModel.
    flame_params : Modified FLAME parameter dict.
    frame_idx    : Frame number (for naming the output image).
    output_dir   : Directory to save rendered frames.
    """
    try:
        from gaussian_renderer import render
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'gaussian_renderer.render'. "
            "Ensure diff-gaussian-rasterization is installed "
            "in the omfs_visual environment."
        ) from exc

    # Deform the Gaussian model with the modified FLAME parameters
    rendered_image = render(model, flame_params)

    # Save the frame
    out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    import cv2
    img_np = (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


def stitch_video(frames_dir: str, output_path: str, fps: int = 30):
    """Use ffmpeg to stitch PNG frames into an H.264 MP4.

    Parameters
    ----------
    frames_dir  : Directory containing frame_00000.png, frame_00001.png, …
    output_path : Output .mp4 file path.
    fps         : Frames per second.
    """
    pattern = os.path.join(frames_dir, "frame_%05d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\n{result.stderr}"
        )
    print(f"[render_surgery] Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render post-surgical prediction video."
    )
    parser.add_argument(
        "--lefort_mm", type=float, required=True,
        help="Le Fort I advancement in millimetres.",
    )
    parser.add_argument(
        "--bsso_mm", type=float, required=True,
        help="BSSO advancement in millimetres.",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=1.0,
        help="Sensitivity multiplier (default 1.0).",
    )
    parser.add_argument(
        "--model_path", type=str, default="02_Visual_Engine/output/model",
        help="Path to trained Gaussian model checkpoint.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="02_Visual_Engine/data",
        help="Path to input video data directory.",
    )
    parser.add_argument(
        "--output", type=str, default="final_prediction.mp4",
        help="Output video file path.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for output video.",
    )
    args = parser.parse_args()

    # --- Compute offsets ---
    lefort_offset = compute_offset(args.lefort_mm, args.sensitivity)
    bsso_offset = compute_offset(args.bsso_mm, args.sensitivity)

    print(f"[render_surgery] Le Fort offset : {lefort_offset:.6f}")
    print(f"[render_surgery] BSSO offset    : {bsso_offset:.6f}")

    # --- Load model ---
    model = load_gaussian_model(args.model_path)

    # --- Determine frame count ---
    params_dir = os.path.join(args.data_dir, "flame_params")
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(
            f"FLAME params directory not found: {params_dir}"
        )
    frame_files = sorted(Path(params_dir).glob("frame_*.npz"))
    num_frames = len(frame_files)
    if num_frames == 0:
        raise FileNotFoundError("No FLAME parameter files found.")

    print(f"[render_surgery] Rendering {num_frames} frames …")

    # --- Render each frame ---
    frames_dir = os.path.join(
        os.path.dirname(args.output), "render_frames_tmp"
    )
    os.makedirs(frames_dir, exist_ok=True)

    for idx in range(num_frames):
        flame_params = get_flame_params(args.data_dir, idx)
        modified = apply_surgical_offsets(flame_params, lefort_offset, bsso_offset)
        render_frame(model, modified, idx, frames_dir)

        if (idx + 1) % 50 == 0 or idx == num_frames - 1:
            print(f"  … rendered {idx + 1}/{num_frames}")

    # --- Stitch into video ---
    stitch_video(frames_dir, args.output, fps=args.fps)

    # --- Cleanup temp frames ---
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    print("[render_surgery] Done.")


if __name__ == "__main__":
    main()
