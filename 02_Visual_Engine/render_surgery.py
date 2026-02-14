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
from typing import Any

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


def load_deformation_map(path: str | None) -> dict[str, Any]:
    """Load optional region-aware deformation controls from JSON."""
    if not path:
        return {}
    map_path = Path(path)
    if not map_path.exists():
        raise FileNotFoundError(f"Deformation map not found: {map_path}")
    with open(map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Deformation map JSON must contain an object at the top level.")
    return payload


def choose_rig_mode(
    requested_mode: str,
    canonical_head_asset: str | None,
) -> tuple[str, str]:
    """
    Return effective rig mode and a human-readable reason.
    """
    if requested_mode == "flame_only":
        return "flame_only", "explicitly requested"
    if canonical_head_asset and Path(canonical_head_asset).exists():
        return "hybrid_full_head", "canonical head asset found"
    return "flame_only", "hybrid requested but canonical head asset missing"


def modify_flame_params(
    source_npz: str,
    output_npz: str,
    lefort_offset: float,
    bsso_offset: float,
    deformation_map: dict[str, Any] | None = None,
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

    deformation_map = deformation_map or {}
    trans_axis = int(deformation_map.get("translation_axis", 1))
    jaw_axis = int(deformation_map.get("jaw_axis", 0))
    lefort_scale = float(deformation_map.get("lefort_scale", 1.0))
    bsso_scale = float(deformation_map.get("bsso_scale", 1.0))

    # Le Fort I: translate maxilla anteriorly (default Y-axis in FLAME space)
    if "translation" in data:
        trans = data["translation"].copy()
        if trans.ndim == 1:
            # Single frame: shape (3,)
            trans[trans_axis] += lefort_offset * lefort_scale
        else:
            # Batched: shape (T, 3)
            trans[:, trans_axis] += lefort_offset * lefort_scale
        data["translation"] = trans

    # BSSO: modify jaw pose (default X rotation = jaw opening/advancement)
    if "jaw_pose" in data:
        jaw = data["jaw_pose"].copy()
        if jaw.ndim == 1:
            # Single frame: shape (3,)
            jaw[jaw_axis] += bsso_offset * bsso_scale
        else:
            # Batched: shape (T, 3)
            jaw[:, jaw_axis] += bsso_offset * bsso_scale
        data["jaw_pose"] = jaw

    np.savez(output_npz, **data)


def create_modified_dataset(
    data_dir: str,
    lefort_offset: float,
    bsso_offset: float,
    deformation_map: dict[str, Any] | None = None,
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

    # Copy flame_param directory if it exists (individual per-frame params)
    # Note: VHAP uses "flame_param" (singular), not "flame_params"
    src_flame_params = os.path.join(data_dir, "flame_param")
    dst_flame_params = os.path.join(temp_dir, "flame_param")
    if os.path.isdir(src_flame_params):
        # Modify each individual flame param file
        os.makedirs(dst_flame_params, exist_ok=True)
        for fname in os.listdir(src_flame_params):
            if fname.endswith(".npz"):
                src_f = os.path.join(src_flame_params, fname)
                dst_f = os.path.join(dst_flame_params, fname)
                modify_flame_params(
                    src_f, dst_f, lefort_offset, bsso_offset, deformation_map=deformation_map
                )

    # Copy/modify main flame_param.npz if it exists
    src_flame = os.path.join(data_dir, "flame_param.npz")
    dst_flame = os.path.join(temp_dir, "flame_param.npz")
    if os.path.exists(src_flame):
        modify_flame_params(
            src_flame, dst_flame, lefort_offset, bsso_offset, deformation_map=deformation_map
        )

    # Copy points3d.ply if it exists
    src_points = os.path.join(data_dir, "points3d.ply")
    dst_points = os.path.join(temp_dir, "points3d.ply")
    if os.path.exists(src_points):
        shutil.copy2(src_points, dst_points)
    
    # CRITICAL: Copy canonical_flame_param.npz - required to trigger DynamicNerf loader
    src_canonical = os.path.join(data_dir, "canonical_flame_param.npz")
    dst_canonical = os.path.join(temp_dir, "canonical_flame_param.npz")
    if os.path.exists(src_canonical):
        shutil.copy2(src_canonical, dst_canonical)
        print(f"[render_surgery] Copied canonical_flame_param.npz")

    # Copy and update transforms JSONs (including val)
    # Update flame_param_path to point to individual per-frame files if they exist
    for json_name in ("transforms_train.json", "transforms_test.json", "transforms_val.json"):
        src_json = os.path.join(data_dir, json_name)
        dst_json = os.path.join(temp_dir, json_name)
        if os.path.exists(src_json):
            with open(src_json, "r") as f:
                transforms = json.load(f)
            
            # Update each frame to point to individual flame_param files
            for frame in transforms.get("frames", []):
                timestep = frame.get("timestep_index", 0)
                individual_param = f"flame_param/{timestep:05d}.npz"
                if os.path.exists(os.path.join(temp_dir, individual_param)):
                    frame["flame_param_path"] = individual_param
            
            with open(dst_json, "w") as f:
                json.dump(transforms, f, indent=2)

    print(f"[render_surgery] Modified dataset at: {temp_dir}")
    
    # Debug: list what was created
    print(f"[render_surgery] Contents of temp_dir:")
    for item in os.listdir(temp_dir):
        item_path = os.path.join(temp_dir, item)
        if os.path.isdir(item_path):
            count = len(os.listdir(item_path))
            print(f"  {item}/ ({count} files)")
        else:
            print(f"  {item}")
    
    # Debug: check transforms
    test_json = os.path.join(temp_dir, "transforms_train.json")
    if os.path.exists(test_json):
        with open(test_json) as f:
            t = json.load(f)
        print(f"[render_surgery] transforms_train.json: {len(t.get('frames', []))} frames")
        if t.get('frames'):
            f0 = t['frames'][0]
            print(f"  First frame: timestep_index={f0.get('timestep_index')}, flame_param_path={f0.get('flame_param_path')}")
    
    return temp_dir


def render_with_gaussians(
    model_path: str,
    data_dir: str,
    iteration: int = -1,
    clear_old_renders: bool = True,
) -> str:
    """Run GaussianAvatars render.py to produce frames.

    Returns path to the rendered frames directory.
    """
    if not RENDER_SCRIPT.exists():
        raise FileNotFoundError(
            f"GaussianAvatars render.py not found at: {RENDER_SCRIPT}"
        )

    # CRITICAL: Clear old renders to avoid using stale frames
    train_dir = os.path.join(model_path, "train")
    if clear_old_renders and os.path.isdir(train_dir):
        for d in os.listdir(train_dir):
            renders = os.path.join(train_dir, d, "renders")
            if os.path.isdir(renders):
                print(f"[render_surgery] Clearing old renders: {renders}")
                shutil.rmtree(renders)

    # Find the best iteration to use
    # Prefer the most recent checkpoint that matches training with current data
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    best_iteration = None
    if os.path.isdir(point_cloud_dir):
        iterations = []
        for d in os.listdir(point_cloud_dir):
            if d.startswith("iteration_"):
                try:
                    it = int(d.split("_")[1])
                    iterations.append(it)
                except (ValueError, IndexError):
                    pass
        if iterations:
            # Use the highest iteration, but cap at what was just trained
            # if --iteration not specified, use highest available
            best_iteration = max(iterations)
            print(f"[render_surgery] Available iterations: {sorted(iterations)}")
            print(f"[render_surgery] Using iteration: {best_iteration}")

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
    elif best_iteration:
        cmd.extend(["--iteration", str(best_iteration)])

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
        print(f"[render_surgery] Render stderr: {result.stderr[-2000:]}")
        print(f"[render_surgery] Render stdout: {result.stdout[-2000:]}")
        raise RuntimeError(
            f"Rendering failed:\n{result.stderr[-2000:]}"
        )

    # Find the rendered frames
    # GaussianAvatars outputs to model_path/train/ours_XXXXX/renders/
    # We want the highest iteration number
    renders_dir = None
    if os.path.isdir(train_dir):
        if iteration > 0:
            preferred = f"ours_{iteration}"
            preferred_renders = os.path.join(train_dir, preferred, "renders")
            if os.path.isdir(preferred_renders):
                frame_count = len([f for f in os.listdir(preferred_renders) if f.endswith(".png")])
                print(f"[render_surgery] Found renders at {preferred}: {frame_count} frames")
                renders_dir = preferred_renders
                print(f"[render_surgery] Frames rendered to: {renders_dir}")
                return renders_dir

        # Sort by iteration number (extract number from "ours_XXXXX")
        def get_iteration_num(dirname):
            try:
                return int(dirname.split("_")[-1])
            except (ValueError, IndexError):
                return 0
        
        dirs = sorted(os.listdir(train_dir), key=get_iteration_num, reverse=True)
        for d in dirs:
            renders = os.path.join(train_dir, d, "renders")
            if os.path.isdir(renders):
                # Verify it has the right number of frames
                frame_count = len([f for f in os.listdir(renders) if f.endswith(".png")])
                print(f"[render_surgery] Found renders at {d}: {frame_count} frames")
                renders_dir = renders
                break

    if renders_dir is None:
        raise FileNotFoundError(
            "No rendered frames found after GaussianAvatars rendering."
        )

    print(f"[render_surgery] Frames rendered to: {renders_dir}")
    return renders_dir


def export_deterministic_frames(
    frames_dir: str,
    output_dir: str,
    index_file: str | None = None,
    max_frames: int = 24,
) -> str:
    """
    Export a deterministic subset of rendered frames for strict A/B comparisons.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not frames:
        raise FileNotFoundError(f"No PNG frames in {frames_dir}")

    if index_file:
        with open(index_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        indices = payload.get("indices", payload)
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ValueError("index_file must contain a JSON list of frame indices or {'indices': [...]} ")
        selected = [i for i in indices if 0 <= i < len(frames)]
    else:
        sample_count = max(1, min(max_frames, len(frames)))
        if sample_count == 1:
            selected = [0]
        else:
            selected = sorted(set(
                int(round(i * (len(frames) - 1) / (sample_count - 1)))
                for i in range(sample_count)
            ))

    manifest = {"source_frames_dir": frames_dir, "selected_indices": selected, "exports": []}
    for i in selected:
        src_name = frames[i]
        src = os.path.join(frames_dir, src_name)
        dst_name = f"idx_{i:05d}.png"
        dst = os.path.join(output_dir, dst_name)
        shutil.copy2(src, dst)
        manifest["exports"].append({"index": i, "source": src_name, "exported": dst_name})

    manifest_path = os.path.join(output_dir, "deterministic_indices_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[render_surgery] Deterministic frame export written to: {output_dir}")
    return output_dir


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
    parser.add_argument("--iteration", type=int, default=-1, help="Explicit model iteration to render.")
    parser.add_argument(
        "--rig_mode",
        type=str,
        default="flame_only",
        choices=("flame_only", "hybrid_full_head"),
        help="Rendering rig mode. hybrid_full_head falls back to flame_only when asset is absent.",
    )
    parser.add_argument(
        "--canonical_head_asset",
        type=str,
        default="",
        help="Path to canonical full-head asset used by hybrid_full_head mode.",
    )
    parser.add_argument(
        "--deformation_map",
        type=str,
        default="",
        help="Optional JSON deformation map controlling region-aware scaling/axes.",
    )
    parser.add_argument(
        "--export_frames_dir",
        type=str,
        default="",
        help="If set, exports deterministic frame subset for strict A/B evaluation.",
    )
    parser.add_argument(
        "--deterministic_indices",
        type=str,
        default="",
        help="Optional JSON file with deterministic frame indices.",
    )
    parser.add_argument(
        "--deterministic_max_frames",
        type=int,
        default=24,
        help="Max deterministic frames when indices are auto-generated.",
    )
    args = parser.parse_args()

    # Compute FLAME offsets
    lefort_offset = compute_offset(args.lefort_mm, args.sensitivity)
    bsso_offset = compute_offset(args.bsso_mm, args.sensitivity)
    effective_mode, mode_reason = choose_rig_mode(args.rig_mode, args.canonical_head_asset)
    deformation_map = load_deformation_map(args.deformation_map if effective_mode == "hybrid_full_head" else None)

    print(f"[render_surgery] Le Fort: {args.lefort_mm} mm -> offset {lefort_offset:.6f}")
    print(f"[render_surgery] BSSO:    {args.bsso_mm} mm -> offset {bsso_offset:.6f}")
    print(f"[render_surgery] Rig mode: {effective_mode} ({mode_reason})")
    if args.iteration > 0:
        print(f"[render_surgery] Pinned iteration: {args.iteration}")

    # Create modified dataset with surgical offsets
    modified_dir = create_modified_dataset(
        args.data_dir, lefort_offset, bsso_offset, deformation_map=deformation_map
    )

    try:
        # Render frames - returns the actual renders directory path
        frames_dir = render_with_gaussians(
            args.model_path, modified_dir, iteration=args.iteration
        )

        if args.export_frames_dir:
            export_deterministic_frames(
                frames_dir=frames_dir,
                output_dir=args.export_frames_dir,
                index_file=args.deterministic_indices or None,
                max_frames=args.deterministic_max_frames,
            )

        # Stitch into video
        stitch_video(frames_dir, args.output, fps=args.fps)

    finally:
        # Clean up temp dataset
        shutil.rmtree(modified_dir, ignore_errors=True)

    print("[render_surgery] Done.")


if __name__ == "__main__":
    main()
