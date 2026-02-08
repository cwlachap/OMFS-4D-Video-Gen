"""
preprocess_video.py — Extract frames from a patient video and prepare
the dataset in the format expected by GaussianAvatars.

Produces:
    data_dir/
    ├── images/
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── ...
    ├── flame_param.npz          (placeholder — needs FLAME fitting)
    └── transforms_train.json    (camera + per-frame metadata)

Usage:
    python 02_Visual_Engine/preprocess_video.py --video path/to/video.mp4 --output_dir 02_Visual_Engine/data
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np


def extract_frames(
    video_path: str,
    output_dir: str,
    max_frames: int = 0,
    target_size: int = 512,
) -> int:
    """Extract frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to input video.
    output_dir : str
        Directory to save frames as PNGs.
    max_frames : int
        Maximum frames to extract (0 = all).
    target_size : int
        Resize frames so the shorter side = target_size.

    Returns
    -------
    int : number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[preprocess] Video: {width}x{height} @ {fps:.1f} fps, {total} frames")

    if max_frames > 0 and max_frames < total:
        # Sample evenly
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
    else:
        indices = np.arange(total)

    count = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break

        # Resize
        h, w = frame.shape[:2]
        if h < w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out_path = os.path.join(output_dir, f"{count:05d}.png")
        cv2.imwrite(out_path, frame)
        count += 1

    cap.release()
    print(f"[preprocess] Extracted {count} frames to {output_dir}")
    return count


def create_placeholder_flame_params(
    num_frames: int,
    output_path: str,
) -> None:
    """Create a placeholder FLAME parameter file.

    This generates identity/neutral params. For production use,
    these should be replaced with fitted FLAME params from a
    face tracker (e.g. MICA, DECA, or the GaussianAvatars tracker).

    Parameters
    ----------
    num_frames : int
        Number of timesteps.
    output_path : str
        Path to save the .npz file.
    """
    params = {
        "shape": np.zeros(300, dtype=np.float32),
        "expr": np.zeros((num_frames, 100), dtype=np.float32),
        "rotation": np.zeros((num_frames, 3), dtype=np.float32),
        "neck_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "jaw_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "eyes_pose": np.zeros((num_frames, 6), dtype=np.float32),
        "translation": np.zeros((num_frames, 3), dtype=np.float32),
        "static_offset": np.zeros((1, 5143, 3), dtype=np.float32),
        "dynamic_offset": np.zeros((num_frames, 5143, 3), dtype=np.float32),
    }
    np.savez(output_path, **params)
    print(f"[preprocess] Placeholder FLAME params saved: {output_path}")


def create_transforms_json(
    images_dir: str,
    flame_param_path: str,
    output_path: str,
    image_width: int = 512,
    image_height: int = 512,
    fov_deg: float = 25.0,
) -> None:
    """Create the transforms_train.json expected by GaussianAvatars.

    Parameters
    ----------
    images_dir : str
        Directory containing frame PNGs.
    flame_param_path : str
        Path to the FLAME .npz file (relative to dataset root).
    output_path : str
        Path to write transforms_train.json.
    image_width, image_height : int
        Frame dimensions.
    fov_deg : float
        Camera field of view (degrees). 25° is typical for head capture.
    """
    fov_rad = math.radians(fov_deg)

    # List all frame images
    frames_list = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png")
    ])

    # Simple frontal camera — identity transform looking at origin
    # This assumes a static camera setup (typical for selfie video)
    camera_distance = 0.6  # ~60cm from face
    transform_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, camera_distance],
        [0, 0, 0, 1],
    ]

    frames = []
    for idx, fname in enumerate(frames_list):
        frames.append({
            "file_path": f"images/{fname}",
            "transform_matrix": transform_matrix,
            "timestep_index": idx,
            "flame_param_path": flame_param_path,
        })

    data = {
        "camera_angle_x": fov_rad,
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[preprocess] transforms_train.json saved with {len(frames)} frames")


def preprocess(
    video_path: str,
    output_dir: str,
    max_frames: int = 300,
    target_size: int = 512,
):
    """Full preprocessing pipeline.

    1. Extract frames from video
    2. Create placeholder FLAME params
    3. Create transforms_train.json
    """
    images_dir = os.path.join(output_dir, "images")
    flame_path = os.path.join(output_dir, "flame_param.npz")
    transforms_path = os.path.join(output_dir, "transforms_train.json")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract frames
    num_frames = extract_frames(video_path, images_dir, max_frames, target_size)

    if num_frames == 0:
        raise RuntimeError("No frames extracted from video.")

    # Get actual frame size
    sample = cv2.imread(os.path.join(images_dir, "00000.png"))
    h, w = sample.shape[:2]

    # 2. Fit FLAME params (or create placeholder if fitting fails)
    try:
        from flame_fitter import fit_video
        print("[preprocess] Running FLAME fitting (GPU) …")
        fit_video(images_dir, flame_path, device="cuda", n_iters=200)
    except Exception as exc:
        print(f"[preprocess] FLAME fitting failed: {exc}")
        print("[preprocess] Creating placeholder FLAME params instead.")
        create_placeholder_flame_params(num_frames, flame_path)

    # 3. Create transforms JSON
    create_transforms_json(
        images_dir, "flame_param.npz", transforms_path,
        image_width=w, image_height=h,
    )

    # Also create a copy as transforms_test.json (GaussianAvatars expects both)
    test_path = os.path.join(output_dir, "transforms_test.json")
    with open(transforms_path, "r") as f:
        data = json.load(f)
    # Use last 10% as test
    split = max(1, len(data["frames"]) - len(data["frames"]) // 10)
    test_data = {**data, "frames": data["frames"][split:]}
    train_data = {**data, "frames": data["frames"][:split]}

    with open(transforms_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"[preprocess] Dataset ready at: {output_dir}")
    print(f"  Train frames: {len(train_data['frames'])}")
    print(f"  Test frames:  {len(test_data['frames'])}")

    return {
        "num_frames": num_frames,
        "image_size": (w, h),
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess patient video for GaussianAvatars.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--output_dir", type=str, default="02_Visual_Engine/data", help="Output directory.")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to extract (0=all).")
    parser.add_argument("--target_size", type=int, default=512, help="Target image size (shorter side).")
    args = parser.parse_args()

    preprocess(args.video, args.output_dir, args.max_frames, args.target_size)


if __name__ == "__main__":
    main()
