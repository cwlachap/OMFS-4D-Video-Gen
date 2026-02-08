"""
Render a post-surgical prediction video using Gaussian Splatting + FLAME deformation.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

SCALE_FACTOR = 0.001  # mm -> FLAME internal units


def _ensure_ffmpeg_available() -> None:
    """Fail early if ffmpeg is not available on PATH."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg and restart the shell."
        )


def compute_offset(input_mm: float, sensitivity: float) -> float:
    """Convert a clinical mm value to a FLAME-space offset."""
    return input_mm * sensitivity * SCALE_FACTOR


def load_gaussian_model(model_path: str):
    """Load a trained 3D Gaussian Splatting model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {model_path}\n"
            "Please train the model first using train_ghost.py."
        )
    point_cloud_path = os.path.join(model_path, "point_cloud.ply")
    if not os.path.isfile(point_cloud_path):
        raise FileNotFoundError(
            f"Model file not found: {point_cloud_path}\n"
            "Expected a trained Gaussian checkpoint with point_cloud.ply."
        )

    try:
        from scene import GaussianModel
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'scene.GaussianModel'. Ensure GaussianAvatars modules are installed."
        ) from exc

    model = GaussianModel(sh_degree=3)
    model.load_ply(point_cloud_path)
    return model


def get_flame_params(data_dir: str, frame_idx: int) -> dict:
    """Retrieve FLAME parameters for a specific frame index."""
    params_dir = os.path.join(data_dir, "flame_params")
    param_file = os.path.join(params_dir, f"frame_{frame_idx:05d}.npz")
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"FLAME parameters not found: {param_file}")

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
    """Modify FLAME params to reflect surgical movements."""
    modified = {k: v.copy() for k, v in flame_params.items()}
    modified["translation"][1] += lefort_offset
    modified["jaw_pose"][0] += bsso_offset
    return modified


def _load_render_callable():
    """Import renderer once and return callable."""
    try:
        from gaussian_renderer import render
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'gaussian_renderer.render'. "
            "Ensure diff-gaussian-rasterization is installed."
        ) from exc
    return render


def render_frame(render_fn, model, flame_params: dict, frame_idx: int, output_dir: str):
    """Deform Gaussians with FLAME params and render one frame."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to write PNG frames.") from exc

    rendered_image = render_fn(model, flame_params)
    out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    img_np = (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


def stitch_video(frames_dir: str, output_path: str, fps: int = 30):
    """Use ffmpeg to stitch PNG frames into an H.264 MP4."""
    _ensure_ffmpeg_available()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(frames_dir, "frame_%05d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    print(f"[render_surgery] Video saved to {output_path}")


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")
    if args.sensitivity < 0:
        raise ValueError("--sensitivity must be >= 0.")
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    params_dir = os.path.join(args.data_dir, "flame_params")
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(f"FLAME params directory not found: {params_dir}")
    if not list(Path(params_dir).glob("frame_*.npz")):
        raise FileNotFoundError("No FLAME parameter files found.")


def main():
    parser = argparse.ArgumentParser(description="Render post-surgical prediction video.")
    parser.add_argument("--lefort_mm", type=float, required=True)
    parser.add_argument("--bsso_mm", type=float, required=True)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--model_path", type=str, default="02_Visual_Engine/output/model")
    parser.add_argument("--data_dir", type=str, default="02_Visual_Engine/data")
    parser.add_argument("--output", type=str, default="final_prediction.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    _validate_inputs(args)
    lefort_offset = compute_offset(args.lefort_mm, args.sensitivity)
    bsso_offset = compute_offset(args.bsso_mm, args.sensitivity)
    print(f"[render_surgery] Le Fort offset : {lefort_offset:.6f}")
    print(f"[render_surgery] BSSO offset    : {bsso_offset:.6f}")

    model = load_gaussian_model(args.model_path)
    render_fn = _load_render_callable()

    params_dir = os.path.join(args.data_dir, "flame_params")
    frame_files = sorted(Path(params_dir).glob("frame_*.npz"))
    num_frames = len(frame_files)
    print(f"[render_surgery] Rendering {num_frames} frames ...")

    output_dir = os.path.dirname(args.output)
    frames_dir = os.path.join(output_dir if output_dir else ".", "render_frames_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    try:
        for idx in range(num_frames):
            flame_params = get_flame_params(args.data_dir, idx)
            modified = apply_surgical_offsets(flame_params, lefort_offset, bsso_offset)
            render_frame(render_fn, model, modified, idx, frames_dir)
            if (idx + 1) % 50 == 0 or idx == num_frames - 1:
                print(f"  ... rendered {idx + 1}/{num_frames}")

        stitch_video(frames_dir, args.output, fps=args.fps)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print("[render_surgery] Done.")


if __name__ == "__main__":
    main()
