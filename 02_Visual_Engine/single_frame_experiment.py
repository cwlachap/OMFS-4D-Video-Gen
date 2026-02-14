"""
Single-frame experiment: take frame 0 from the pipeline data, train Gaussians
bound to FLAME mesh on that one frame, render it, and save for comparison.

This isolates whether the pipeline can reproduce one view when given
correct FLAME + camera. Run from project root:

  python 02_Visual_Engine/single_frame_experiment.py

Outputs:
  - 02_Visual_Engine/data_single_frame/   (1 image, 1 flame_param, 1 fg_mask, transforms)
  - 02_Visual_Engine/output/model_single_frame/   (trained model)
  - 02_Visual_Engine/single_frame_gt.png   (copy of input frame 0)
  - 02_Visual_Engine/single_frame_render.png   (rendered frame 0)
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VISUAL_DIR = REPO_ROOT / "02_Visual_Engine"
DATA_CONDA = VISUAL_DIR / "data_conda"
DATA_SINGLE = VISUAL_DIR / "data_single_frame"
MODEL_SINGLE = VISUAL_DIR / "output" / "model_single_frame"
VENV_PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"


def build_single_frame_dataset():
    """Create data_single_frame with only frame 0 from data_conda."""
    if DATA_SINGLE.exists():
        shutil.rmtree(DATA_SINGLE)
    DATA_SINGLE.mkdir(parents=True, exist_ok=True)

    (DATA_SINGLE / "images").mkdir(exist_ok=True)
    (DATA_SINGLE / "flame_param").mkdir(exist_ok=True)
    (DATA_SINGLE / "fg_masks").mkdir(exist_ok=True)

    # Copy frame 0 assets
    shutil.copy2(DATA_CONDA / "images" / "00000_00.png", DATA_SINGLE / "images" / "00000_00.png")
    shutil.copy2(DATA_CONDA / "flame_param" / "00000.npz", DATA_SINGLE / "flame_param" / "00000.npz")
    shutil.copy2(DATA_CONDA / "fg_masks" / "00000_00.png", DATA_SINGLE / "fg_masks" / "00000_00.png")

    # Load full transforms and keep only first frame
    with open(DATA_CONDA / "transforms_train.json") as f:
        full = json.load(f)
    frame0 = full["frames"][0]
    single_data = {
        "camera_angle_x": full["camera_angle_x"],
        "camera_angle_y": full["camera_angle_y"],
        "fl_x": full["fl_x"],
        "fl_y": full["fl_y"],
        "cx": full["cx"],
        "cy": full["cy"],
        "w": full["w"],
        "h": full["h"],
        "frames": [frame0],
    }
    for name in ["transforms_train.json", "transforms_test.json", "transforms_val.json"]:
        with open(DATA_SINGLE / name, "w") as f:
            json.dump(single_data, f, indent=2)

    # Batched flame_param.npz (single frame) - GA may use it
    import numpy as np
    p0 = dict(np.load(DATA_CONDA / "flame_param" / "00000.npz", allow_pickle=True))
    batched = {}
    for k, v in p0.items():
        if v.ndim == 1:
            batched[k] = v
        else:
            batched[k] = v[None, ...] if v.shape[0] != 1 else v
    np.savez(DATA_SINGLE / "flame_param.npz", **batched)

    # Canonical (neutral) - copy from data_conda so scene is detected as DynamicNerf
    shutil.copy2(DATA_CONDA / "canonical_flame_param.npz", DATA_SINGLE / "canonical_flame_param.npz")

    print(f"[single_frame] Built {DATA_SINGLE} (1 frame)")
    return DATA_SINGLE


def train_single_frame():
    """Train GaussianAvatars on the single-frame dataset (bind_to_mesh)."""
    cmd = [
        str(VENV_PYTHON),
        str(VISUAL_DIR / "train_ghost.py"),
        "--data_dir", str(DATA_SINGLE),
        "--output_dir", str(MODEL_SINGLE),
        "--iterations", "3000",
        "--resolution", "-1",
    ]
    print(f"[single_frame] Training: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "gaussian_avatars_repo") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Training failed")
    print("[single_frame] Training done.")


def render_single_frame_and_save():
    """Render the single view and save GT + render as PNGs for comparison."""
    # Use render_surgery with zero surgery offsets so we just render the same pose
    cmd = [
        str(VENV_PYTHON),
        str(VISUAL_DIR / "render_surgery.py"),
        "--model_path", str(MODEL_SINGLE),
        "--data_dir", str(DATA_SINGLE),
        "--output", str(VISUAL_DIR / "single_frame_video.mp4"),
        "--lefort_mm", "0",
        "--bsso_mm", "0",
        "--fps", "1",
    ]
    print(f"[single_frame] Rendering: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        raise RuntimeError("Render failed")

    # Find the rendered frame (single frame)
    train_dir = MODEL_SINGLE / "train"
    renders_dir = None
    if train_dir.exists():
        def iter_num(p):
            try:
                return int(p.name.split("_")[-1]) if p.name.startswith("ours_") else -1
            except (ValueError, IndexError):
                return -1
        for d in sorted(train_dir.iterdir(), key=iter_num, reverse=True):
            r = d / "renders"
            if r.exists() and list(r.glob("*.png")):
                renders_dir = r
                break
    if not renders_dir or not list(renders_dir.glob("*.png")):
        subdirs = list(train_dir.iterdir()) if train_dir.exists() else []
        raise FileNotFoundError(
            "No rendered frames found. Looked in model_single_frame/train/*/renders/. "
            f"Subdirs: {[p.name for p in subdirs]}"
        )

    # Ensure output dir exists and use resolved paths so they are easy to find
    out_dir = VISUAL_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dst = out_dir / "single_frame_render.png"
    gt_dst = out_dir / "single_frame_gt.png"

    rendered_pngs = sorted(renders_dir.glob("*.png"))
    shutil.copy2(rendered_pngs[0], render_dst)
    shutil.copy2(DATA_SINGLE / "images" / "00000_00.png", gt_dst)

    print("\n--- OUTPUT FILES (absolute paths) ---")
    print(f"  Render: {render_dst}")
    print(f"  GT:     {gt_dst}")
    print("------------------------------------")
    print("Compare single_frame_gt.png (input) vs single_frame_render.png (Gaussians on mesh).")


def main():
    if not DATA_CONDA.exists():
        print("Run the full conda pipeline first to create 02_Visual_Engine/data_conda")
        sys.exit(1)
    build_single_frame_dataset()
    train_single_frame()
    render_single_frame_and_save()
    out_dir = VISUAL_DIR.resolve()
    print(f"\nDone. Open: {out_dir / 'single_frame_gt.png'} and {out_dir / 'single_frame_render.png'}")


if __name__ == "__main__":
    main()
