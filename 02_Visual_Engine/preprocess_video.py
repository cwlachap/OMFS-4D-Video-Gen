"""
preprocess_video.py — Preprocess patient video using VHAP for accurate FLAME tracking.

Pipeline:
1. Run VHAP preprocessing (extract frames + background matting)
2. Run VHAP tracking (FLAME fitting with photometric optimization)
3. Export to GaussianAvatars-compatible format

Produces:
    data_dir/
    ├── images/
    │   ├── 00000.png
    │   └── ...
    ├── fg_masks/
    │   ├── 00000.png
    │   └── ...
    ├── flame_param.npz           (batched FLAME params for all frames)
    ├── canonical_flame_param.npz (canonical/neutral pose)
    └── transforms_train.json     (camera + per-frame metadata)

Usage:
    python 02_Visual_Engine/preprocess_video.py --video path/to/video.mp4 --output_dir 02_Visual_Engine/data
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Add VHAP to path
REPO_ROOT = Path(__file__).parent.parent.resolve()
VHAP_ROOT = REPO_ROOT / "vhap_repo"
sys.path.insert(0, str(VHAP_ROOT))

# Python executable in current venv
PYTHON_EXE = sys.executable

# Add ffmpeg to PATH if not already available
FFMPEG_PATHS = [
    r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]
for ffmpeg_path in FFMPEG_PATHS:
    if os.path.isdir(ffmpeg_path) and ffmpeg_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
        print(f"[VHAP] Added ffmpeg to PATH: {ffmpeg_path}")
        break


def run_command(cmd: list, description: str, cwd: str = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[VHAP] {description}")
    print(f"[VHAP] Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        cmd,
        cwd=cwd or str(REPO_ROOT),
        capture_output=False,  # Stream output directly
        env=os.environ.copy(),  # Pass updated environment with ffmpeg PATH
    )
    
    if result.returncode != 0:
        print(f"\n[VHAP] ERROR: {description} failed with code {result.returncode}")
        return False
    
    print(f"\n[VHAP] {description} completed successfully.")
    return True


def setup_vhap_data_structure(video_path: str, vhap_data_dir: Path) -> str:
    """
    Set up the directory structure VHAP expects.
    
    VHAP expects:
        data/monocular/<sequence_name>.mp4
    
    After preprocessing, VHAP creates:
        data/monocular/<sequence_name>/<sequence_name>/images/
        data/monocular/<sequence_name>/<sequence_name>/alpha_maps/
    
    Then for tracking, VHAP expects sequence at:
        data/monocular/<sequence_name>/<sequence_name>/
    
    Returns the sequence name.
    """
    video_path = Path(video_path)
    sequence_name = video_path.stem  # e.g., "patient_video"
    
    # VHAP preprocess expects the video directly in the monocular folder
    target_video = vhap_data_dir / f"{sequence_name}.mp4"
    if not target_video.exists():
        print(f"[VHAP] Copying video to {target_video}")
        shutil.copy2(video_path, target_video)
    else:
        print(f"[VHAP] Video already exists at {target_video}")
    
    return sequence_name


def run_vhap_preprocess(sequence_name: str, vhap_data_dir: Path, downsample: int = None, skip_matting: bool = False) -> bool:
    """
    Run VHAP preprocessing: extract frames + background matting.
    
    Parameters
    ----------
    skip_matting : bool
        If True, skip background matting entirely. Useful when matting
        fails (complex backgrounds) and you want to keep full frames.
    """
    video_path = vhap_data_dir / f"{sequence_name}.mp4"
    
    cmd = [
        PYTHON_EXE,
        str(VHAP_ROOT / "vhap" / "preprocess_video.py"),
        "--input", str(video_path),
    ]
    
    if skip_matting:
        # Don't pass matting_method at all to skip matting
        print("[VHAP] Skipping background matting (using full frames)")
    else:
        cmd.extend(["--matting_method", "robust_video_matting"])
    
    if downsample and downsample > 1:
        cmd.extend(["--downsample_scales", str(downsample)])
    
    return run_command(cmd, "VHAP Preprocessing (frames + matting)", cwd=str(VHAP_ROOT))


def run_vhap_tracking(
    sequence_name: str,
    vhap_data_dir: Path,
    output_dir: Path,
    use_static_offset: bool = True,
    downsample: int = None,
    skip_matting: bool = False,
) -> bool:
    """
    Run VHAP FLAME tracking.
    
    After preprocessing, VHAP creates:
        <data_dir>/<sequence_name>/images/
        <data_dir>/<sequence_name>/images_2/ (if downsampled)
    
    So for tracking, the sequence is just: <sequence_name>
    """
    cmd = [
        PYTHON_EXE,
        str(VHAP_ROOT / "vhap" / "track.py"),
        "--data.root-folder", str(vhap_data_dir),
        "--data.sequence", sequence_name,
        "--exp.output-folder", str(output_dir),
        "--data.landmark-source", "face-alignment",  # More reliable on Windows
        "--data.landmark-detector-njobs", "1",  # Single process to avoid race conditions on Windows
    ]
    
    if skip_matting:
        # Don't use background color replacement (requires alpha maps)
        cmd.extend(["--data.background-color", "None"])
        print("[VHAP] Skipping background color (no alpha maps)")
    
    if not use_static_offset:
        cmd.append("--model.no-use-static-offset")
    
    if downsample and downsample > 1:
        cmd.extend(["--data.n-downsample-rgb", str(downsample)])
    
    return run_command(cmd, "VHAP FLAME Tracking", cwd=str(VHAP_ROOT))


def run_vhap_export(
    tracking_output_dir: Path,
    export_dir: Path,
    background_color: str = "white",
) -> bool:
    """
    Export VHAP tracking results to NeRF-style dataset.
    """
    cmd = [
        PYTHON_EXE,
        str(VHAP_ROOT / "vhap" / "export_as_nerf_dataset.py"),
        "--src-folder", str(tracking_output_dir),
        "--tgt-folder", str(export_dir),
        "--background-color", background_color,
    ]
    
    return run_command(cmd, "VHAP Export to NeRF format", cwd=str(VHAP_ROOT))


def convert_to_gaussianavatars_format(vhap_export_dir: Path, output_dir: Path) -> dict:
    """
    Convert VHAP export format to GaussianAvatars-compatible format.
    
    VHAP exports individual flame_param/*.npz files per frame.
    GaussianAvatars expects per-frame flame_param files + proper camera intrinsics.
    
    Key fixes applied:
    1. Use per-frame camera intrinsics (not top-level normalized values)
    2. Copy individual flame_param files (GaussianAvatars loads per-frame)
    3. Preserve correct file naming from VHAP
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VHAP transforms
    transforms_path = vhap_export_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"VHAP export not found: {transforms_path}")
    
    with open(transforms_path, "r") as f:
        vhap_data = json.load(f)
    
    frames = vhap_data.get("frames", [])
    if not frames:
        raise ValueError("No frames in VHAP export")
    
    num_frames = len(frames)
    print(f"[convert] Converting {num_frames} frames to GaussianAvatars format")
    
    # Get camera intrinsics from FIRST FRAME (not top-level, which has wrong values)
    first_frame = frames[0]
    fl_x = first_frame.get("fl_x", 1000.0)
    fl_y = first_frame.get("fl_y", fl_x)
    cx = first_frame.get("cx", 480.0)
    cy = first_frame.get("cy", 270.0)
    w = first_frame.get("w", 960)
    h = first_frame.get("h", 540)
    fov_x = first_frame.get("camera_angle_x", 2 * math.atan(w / (2 * fl_x)))
    fov_y = first_frame.get("camera_angle_y", 2 * math.atan(h / (2 * fl_y)))
    
    print(f"[convert] Camera intrinsics from first frame:")
    print(f"  fl_x={fl_x:.1f}, fl_y={fl_y:.1f}")
    print(f"  cx={cx:.1f}, cy={cy:.1f}")
    print(f"  w={w}, h={h}")
    print(f"  fov_x={math.degrees(fov_x):.1f} deg")
    
    # Copy/link images directory
    images_src = vhap_export_dir / "images"
    images_dst = output_dir / "images"
    if images_dst.exists():
        shutil.rmtree(images_dst)
    if images_src.exists():
        shutil.copytree(images_src, images_dst)
        print(f"[convert] Copied images: {len(list(images_dst.glob('*.png')))} files")
    
    # Copy foreground masks
    fg_masks_src = vhap_export_dir / "fg_masks"
    fg_masks_dst = output_dir / "fg_masks"
    if fg_masks_dst.exists():
        shutil.rmtree(fg_masks_dst)
    if fg_masks_src.exists():
        shutil.copytree(fg_masks_src, fg_masks_dst)
        print(f"[convert] Copied fg_masks: {len(list(fg_masks_dst.glob('*.png')))} files")
    
    # Copy individual flame_param files (GaussianAvatars loads these per-frame)
    flame_param_src = vhap_export_dir / "flame_param"
    flame_param_dst = output_dir / "flame_param"
    if flame_param_dst.exists():
        shutil.rmtree(flame_param_dst)
    if flame_param_src.exists():
        shutil.copytree(flame_param_src, flame_param_dst)
        print(f"[convert] Copied flame_param: {len(list(flame_param_dst.glob('*.npz')))} files")
    
    # Also create batched flame_param.npz for compatibility
    all_params = {
        "shape": None,
        "expr": [],
        "rotation": [],
        "neck_pose": [],
        "jaw_pose": [],
        "eyes_pose": [],
        "translation": [],
        "static_offset": None,
        "dynamic_offset": [],
    }
    
    for i, frame in enumerate(frames):
        # Determine flame_param path
        timestep_idx = frame.get("timestep_index", i)
        flame_path_rel = frame.get("flame_param_path", f"flame_param/{timestep_idx:05d}.npz")
        flame_path = vhap_export_dir / flame_path_rel
        
        if flame_path.exists():
            params = dict(np.load(flame_path))
            
            if all_params["shape"] is None:
                all_params["shape"] = params["shape"]
            
            all_params["expr"].append(params["expr"][0] if params["expr"].ndim > 1 else params["expr"])
            all_params["rotation"].append(params["rotation"][0] if params["rotation"].ndim > 1 else params["rotation"])
            all_params["neck_pose"].append(params["neck_pose"][0] if params["neck_pose"].ndim > 1 else params["neck_pose"])
            all_params["jaw_pose"].append(params["jaw_pose"][0] if params["jaw_pose"].ndim > 1 else params["jaw_pose"])
            all_params["eyes_pose"].append(params["eyes_pose"][0] if params["eyes_pose"].ndim > 1 else params["eyes_pose"])
            all_params["translation"].append(params["translation"][0] if params["translation"].ndim > 1 else params["translation"])
            
            if "static_offset" in params and all_params["static_offset"] is None:
                all_params["static_offset"] = params["static_offset"]
            
            if "dynamic_offset" in params:
                all_params["dynamic_offset"].append(params["dynamic_offset"][0] if params["dynamic_offset"].ndim > 2 else params["dynamic_offset"])
    
    # Stack into batched arrays
    T = len(all_params["expr"])
    
    batched_params = {
        "shape": all_params["shape"] if all_params["shape"] is not None else np.zeros(300, dtype=np.float32),
        "expr": np.stack(all_params["expr"]) if all_params["expr"] else np.zeros((T, 100), dtype=np.float32),
        "rotation": np.stack(all_params["rotation"]) if all_params["rotation"] else np.zeros((T, 3), dtype=np.float32),
        "neck_pose": np.stack(all_params["neck_pose"]) if all_params["neck_pose"] else np.zeros((T, 3), dtype=np.float32),
        "jaw_pose": np.stack(all_params["jaw_pose"]) if all_params["jaw_pose"] else np.zeros((T, 3), dtype=np.float32),
        "eyes_pose": np.stack(all_params["eyes_pose"]) if all_params["eyes_pose"] else np.zeros((T, 6), dtype=np.float32),
        "translation": np.stack(all_params["translation"]) if all_params["translation"] else np.zeros((T, 3), dtype=np.float32),
    }
    
    # Handle static/dynamic offsets
    if all_params["static_offset"] is not None:
        batched_params["static_offset"] = all_params["static_offset"]
    else:
        batched_params["static_offset"] = np.zeros((1, 5143, 3), dtype=np.float32)
    
    if all_params["dynamic_offset"]:
        batched_params["dynamic_offset"] = np.stack(all_params["dynamic_offset"])
    else:
        batched_params["dynamic_offset"] = np.zeros((T, 5143, 3), dtype=np.float32)
    
    # Save batched FLAME params
    batched_path = output_dir / "flame_param.npz"
    np.savez(batched_path, **batched_params)
    print(f"[convert] Saved batched FLAME params: {batched_path}")
    print(f"  Shape: {batched_params['shape'].shape}")
    print(f"  Expr:  {batched_params['expr'].shape}")
    print(f"  Translation range: [{batched_params['translation'].min():.3f}, {batched_params['translation'].max():.3f}]")
    
    # Create canonical FLAME params (neutral pose)
    canonical_params = {
        "shape": batched_params["shape"],
        "expr": np.zeros((1, batched_params["expr"].shape[1]), dtype=np.float32),
        "rotation": np.zeros((1, 3), dtype=np.float32),
        "neck_pose": np.zeros((1, 3), dtype=np.float32),
        "jaw_pose": np.zeros((1, 3), dtype=np.float32),
        "eyes_pose": np.zeros((1, 6), dtype=np.float32),
        "translation": np.zeros((1, 3), dtype=np.float32),
        "static_offset": batched_params["static_offset"],
        "dynamic_offset": np.zeros((1, 5143, 3), dtype=np.float32),
    }
    canonical_path = output_dir / "canonical_flame_param.npz"
    np.savez(canonical_path, **canonical_params)
    print(f"[convert] Saved canonical FLAME params: {canonical_path}")
    
    # Build GaussianAvatars-compatible transforms JSON
    # Key: use CORRECT camera intrinsics from per-frame data
    ga_frames = []
    for i, frame in enumerate(frames):
        # Use the original file paths from VHAP (e.g., "images/00000_00.png")
        file_path = frame.get("file_path", f"images/{i:05d}_00.png")
        # fg_mask_path is optional - only include if present in source
        fg_mask_path = frame.get("fg_mask_path")
        flame_param_path = frame.get("flame_param_path", f"flame_param/{i:05d}.npz")
        
        # Get transform matrix from frame
        transform = frame.get("transform_matrix", [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
        
        frame_entry = {
            "file_path": file_path,
            "flame_param_path": flame_param_path,
            "transform_matrix": transform,
            "timestep_index": frame.get("timestep_index", i),
            "camera_index": frame.get("camera_index", 0),
            # Include per-frame intrinsics for GaussianAvatars
            "camera_angle_x": frame.get("camera_angle_x", fov_x),
            "w": frame.get("w", w),
            "h": frame.get("h", h),
        }
        # Only include fg_mask_path if it exists in source
        if fg_mask_path:
            frame_entry["fg_mask_path"] = fg_mask_path
        ga_frames.append(frame_entry)
    
    # Create transforms with CORRECT top-level intrinsics
    ga_data = {
        "camera_angle_x": fov_x,  # Use correct value from first frame
        "camera_angle_y": fov_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": ga_frames,
        "timestep_indices": list(range(T)),
        "camera_indices": [0],
    }
    
    # Save train/test/val splits (90/10 split)
    split_idx = max(1, T - T // 10)
    train_data = {**ga_data, "frames": ga_frames[:split_idx]}
    test_data = {**ga_data, "frames": ga_frames[split_idx:]}
    
    with open(output_dir / "transforms_train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(output_dir / "transforms_test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    with open(output_dir / "transforms_val.json", "w") as f:
        json.dump(test_data, f, indent=2)
    # Also save combined transforms.json
    with open(output_dir / "transforms.json", "w") as f:
        json.dump(ga_data, f, indent=2)
    
    print(f"[convert] Created transforms JSON files")
    print(f"  Train frames: {len(train_data['frames'])}")
    print(f"  Test frames:  {len(test_data['frames'])}")
    
    return {
        "num_frames": T,
        "image_size": (w, h),
        "output_dir": str(output_dir),
    }


def preprocess_with_vhap(
    video_path: str,
    output_dir: str,
    max_frames: int = 300,
    target_size: int = 512,
    use_static_offset: bool = True,
) -> dict:
    """
    Full preprocessing pipeline using VHAP.
    
    1. Setup VHAP data structure
    2. Run VHAP preprocessing (frames + matting)
    3. Run VHAP FLAME tracking
    4. Export to NeRF format
    5. Convert to GaussianAvatars format
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Determine downsample factor based on target_size
    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    min_dim = min(orig_w, orig_h)
    # target_size=0 means native resolution (no downsampling)
    downsample = max(1, min_dim // target_size) if target_size > 0 and min_dim > target_size else None
    
    print(f"[VHAP] Video: {orig_w}x{orig_h}")
    print(f"[VHAP] Target size: {target_size}")
    print(f"[VHAP] Downsample factor: {downsample or 'None (original size)'}")
    
    # Setup directories
    vhap_data_dir = REPO_ROOT / "vhap_data" / "monocular"
    vhap_data_dir.mkdir(parents=True, exist_ok=True)
    
    sequence_name = video_path.stem
    tracking_output_dir = REPO_ROOT / "vhap_output" / f"{sequence_name}_tracked"
    export_dir = REPO_ROOT / "vhap_export" / f"{sequence_name}_export"
    
    # Step 1: Setup VHAP data structure
    print("\n" + "="*60)
    print("[VHAP] Step 1: Setting up data structure")
    print("="*60)
    sequence_name = setup_vhap_data_structure(str(video_path), vhap_data_dir)
    
    # Step 2: Run VHAP preprocessing
    print("\n" + "="*60)
    print("[VHAP] Step 2: Preprocessing (frame extraction + matting)")
    print("="*60)
    if not run_vhap_preprocess(sequence_name, vhap_data_dir, downsample, skip_matting=True):
        raise RuntimeError("VHAP preprocessing failed")
    
    # Step 3: Run VHAP tracking
    print("\n" + "="*60)
    print("[VHAP] Step 3: FLAME tracking")
    print("="*60)
    if not run_vhap_tracking(sequence_name, vhap_data_dir, tracking_output_dir, use_static_offset, downsample, skip_matting=True):
        raise RuntimeError("VHAP tracking failed")
    
    # Step 4: Export to NeRF format
    print("\n" + "="*60)
    print("[VHAP] Step 4: Exporting to NeRF format")
    print("="*60)
    if not run_vhap_export(tracking_output_dir, export_dir):
        raise RuntimeError("VHAP export failed")
    
    # Step 5: Convert to GaussianAvatars format
    print("\n" + "="*60)
    print("[VHAP] Step 5: Converting to GaussianAvatars format")
    print("="*60)
    result = convert_to_gaussianavatars_format(export_dir, output_dir)
    
    print("\n" + "="*60)
    print("[VHAP] Preprocessing complete!")
    print(f"[VHAP] Output directory: {output_dir}")
    print(f"[VHAP] Frames: {result['num_frames']}")
    print(f"[VHAP] Image size: {result['image_size']}")
    print("="*60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess patient video using VHAP for accurate FLAME tracking."
    )
    parser.add_argument("--video", type=str, default=None, help="Path to input video (required unless --convert-only).")
    parser.add_argument("--output_dir", type=str, default="02_Visual_Engine/data", help="Output directory.")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames (unused with VHAP, processes all).")
    parser.add_argument("--target_size", type=int, default=0, help="Target image size (shorter side). 0 = native resolution.")
    parser.add_argument("--no-static-offset", action="store_true", help="Disable FLAME static offset.")
    parser.add_argument("--convert-only", action="store_true", help="Only convert existing VHAP export to GaussianAvatars format.")
    parser.add_argument("--vhap_export_dir", type=str, default=None, help="VHAP export folder (required with --convert-only).")
    args = parser.parse_args()

    if args.convert_only:
        if not args.vhap_export_dir:
            parser.error("--convert-only requires --vhap_export_dir")
        vhap_export = Path(args.vhap_export_dir)
        output_dir = Path(args.output_dir)
        if not vhap_export.exists():
            raise FileNotFoundError(f"VHAP export not found: {vhap_export}")
        convert_to_gaussianavatars_format(vhap_export, output_dir)
        return

    if not args.video:
        parser.error("--video is required unless --convert-only")
    preprocess_with_vhap(
        args.video,
        args.output_dir,
        args.max_frames,
        args.target_size,
        use_static_offset=not args.no_static_offset,
    )


if __name__ == "__main__":
    main()
