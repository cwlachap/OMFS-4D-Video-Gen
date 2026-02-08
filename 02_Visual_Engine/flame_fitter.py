"""
flame_fitter.py — Fit FLAME 2023 parameters to video frames using
MediaPipe face landmarks + PyTorch optimization.

Produces a flame_param.npz file compatible with GaussianAvatars:
    shape      : (300,)
    expr       : (T, 100)
    rotation   : (T, 3)
    neck_pose  : (T, 3)
    jaw_pose   : (T, 3)
    eyes_pose  : (T, 6)
    translation: (T, 3)

Usage:
    python 02_Visual_Engine/flame_fitter.py \
        --images_dir 02_Visual_Engine/data/images \
        --output 02_Visual_Engine/data/flame_param.npz
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).parent.parent / "gaussian_avatars_repo"
FLAME_MODEL_PATH = REPO_DIR / "flame_model" / "assets" / "flame" / "flame2023.pkl"
FLAME_LMK_PATH = REPO_DIR / "flame_model" / "assets" / "flame" / "landmark_embedding_with_eyes.npy"

# MediaPipe → FLAME landmark correspondence
# FLAME has 68 landmarks (standard face landmarks) + eye landmarks
# MediaPipe has 478 landmarks — we map a subset
# These are the 68 standard face landmark indices in MediaPipe
MEDIAPIPE_TO_68 = [
    # Jaw contour (17 points)
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400,
    # Left eyebrow (5 points)
    46, 53, 52, 65, 55,
    # Right eyebrow (5 points)
    285, 295, 282, 283, 276,
    # Nose bridge (4 points)
    6, 197, 195, 5,
    # Nose tip (5 points)
    48, 115, 220, 45, 4,
    # Left eye (6 points)
    33, 160, 158, 133, 153, 144,
    # Right eye (6 points)
    362, 385, 387, 263, 373, 380,
    # Outer lip (12 points)
    61, 40, 37, 0, 267, 270, 291,
    321, 314, 17, 84, 91,
    # Inner lip (8 points)
    78, 82, 13, 312, 308, 317, 14, 87,
]


class SimpleFLAME(nn.Module):
    """Minimal FLAME forward pass for landmark fitting.

    Only needs flame2023.pkl — no masks, no teeth, no template mesh.
    """

    def __init__(self, flame_model_path: str, n_shape: int = 100, n_expr: int = 50):
        super().__init__()
        self.n_shape = n_shape
        self.n_expr = n_expr

        with open(flame_model_path, "rb") as f:
            model = pickle.load(f, encoding="latin1")

        # Template vertices
        self.register_buffer("v_template", torch.tensor(
            np.array(model["v_template"], dtype=np.float32)
        ))

        # Shape + expression blend shapes
        shapedirs = np.array(model["shapedirs"], dtype=np.float32)
        # First 300 = shape, next 100 = expression
        self.register_buffer("shapedirs_shape", torch.tensor(shapedirs[:, :, :n_shape]))
        self.register_buffer("shapedirs_expr", torch.tensor(shapedirs[:, :, 300:300 + n_expr]))

        # Joint regressor
        J_regressor = np.array(model["J_regressor"].todense(), dtype=np.float32)
        self.register_buffer("J_regressor", torch.tensor(J_regressor))

        # Skinning weights
        self.register_buffer("lbs_weights", torch.tensor(
            np.array(model["weights"], dtype=np.float32)
        ))

        # Kinematic tree
        kintree = np.array(model["kintree_table"], dtype=np.int64)
        parents = kintree[0].copy()
        parents[0] = -1
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))

        # Landmark embedding
        lmk_data = np.load(str(FLAME_LMK_PATH), allow_pickle=True)[()]
        self.register_buffer("lmk_faces_idx", torch.tensor(
            lmk_data["full_lmk_faces_idx"], dtype=torch.long
        ))
        self.register_buffer("lmk_bary_coords", torch.tensor(
            np.array(lmk_data["full_lmk_bary_coords"], dtype=np.float32)
        ))

        # Faces for landmark computation
        faces = np.array(model["f"], dtype=np.int64)
        self.register_buffer("faces", torch.tensor(faces, dtype=torch.long))

    def forward(self, shape, expr, rotation, jaw, translation):
        """
        Parameters — all batched (B, ...):
            shape:       (B, n_shape)
            expr:        (B, n_expr)
            rotation:    (B, 3) — global rotation (axis-angle)
            jaw:         (B, 3) — jaw rotation (axis-angle)
            translation: (B, 3)

        Returns:
            landmarks: (B, N_lmk, 3) — 3D face landmarks
        """
        B = shape.shape[0]

        # Shape + expression blend shapes
        v = self.v_template.unsqueeze(0).expand(B, -1, -1)
        v = v + torch.einsum("bijk,bk->bij",
                             self.shapedirs_shape.unsqueeze(0).expand(B, -1, -1, -1),
                             shape)
        v = v + torch.einsum("bijk,bk->bij",
                             self.shapedirs_expr.unsqueeze(0).expand(B, -1, -1, -1),
                             expr)

        # Simple translation (skip full LBS for speed — good enough for fitting)
        v = v + translation.unsqueeze(1)

        # Apply jaw rotation to approximate jaw movement
        # jaw[0] = opening angle around X axis
        jaw_angle = jaw[:, 0:1]  # (B, 1)
        # Move lower face vertices down proportional to jaw angle
        # Vertices below y=0 in template space are roughly the lower jaw
        lower_mask = (self.v_template[:, 1] < self.v_template[:, 1].mean()).float()
        jaw_offset = torch.zeros_like(v)
        jaw_offset[:, :, 1] = -jaw_angle * lower_mask.unsqueeze(0) * 0.1

        v = v + jaw_offset

        # Compute landmarks from vertices via barycentric interpolation
        lmk_faces = self.faces[self.lmk_faces_idx]  # (N_lmk, 3) — vertex indices
        lmk_verts = v[:, lmk_faces]  # (B, N_lmk, 3, 3)
        bary = self.lmk_bary_coords.unsqueeze(0).unsqueeze(-1)  # (1, N_lmk, 3, 1)
        landmarks = (lmk_verts * bary).sum(dim=2)  # (B, N_lmk, 3)

        return landmarks


def detect_landmarks_mediapipe(images_dir: str) -> list:
    """Detect 2D face landmarks using MediaPipe for all frames.

    Returns list of (N, 2) arrays or None for frames with no face.
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    frames = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    all_landmarks = []

    print(f"[flame_fitter] Detecting landmarks in {len(frames)} frames …")
    for fname in tqdm(frames, desc="MediaPipe"):
        img = cv2.imread(os.path.join(images_dir, fname))
        if img is None:
            all_landmarks.append(None)
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            all_landmarks.append(None)
            continue

        face = results.multi_face_landmarks[0]
        # Extract the 68-point subset
        lmk_2d = np.zeros((len(MEDIAPIPE_TO_68), 2), dtype=np.float32)
        for i, mp_idx in enumerate(MEDIAPIPE_TO_68):
            lm = face.landmark[mp_idx]
            lmk_2d[i, 0] = lm.x * w
            lmk_2d[i, 1] = lm.y * h

        all_landmarks.append(lmk_2d)

    face_mesh.close()

    n_detected = sum(1 for l in all_landmarks if l is not None)
    print(f"[flame_fitter] Landmarks detected: {n_detected}/{len(frames)}")
    return all_landmarks


def fit_flame_to_landmarks(
    landmarks_2d_list: list,
    image_size: tuple,
    flame_model_path: str,
    n_shape: int = 100,
    n_expr: int = 50,
    lr: float = 0.01,
    n_iters: int = 200,
    device: str = "cuda",
) -> dict:
    """Fit FLAME parameters to 2D landmarks.

    Parameters
    ----------
    landmarks_2d_list : list of (68, 2) arrays (or None for missing frames)
    image_size : (width, height)
    flame_model_path : path to flame2023.pkl

    Returns
    -------
    dict with FLAME params in GaussianAvatars format.
    """
    T = len(landmarks_2d_list)
    W, H = image_size

    # Filter out None frames
    valid_indices = [i for i, l in enumerate(landmarks_2d_list) if l is not None]
    if not valid_indices:
        raise ValueError("No faces detected in any frame.")

    flame = SimpleFLAME(flame_model_path, n_shape, n_expr).to(device)

    # Optimizable parameters
    shape = nn.Parameter(torch.zeros(1, n_shape, device=device))
    expr = nn.Parameter(torch.zeros(T, n_expr, device=device))
    rotation = nn.Parameter(torch.zeros(T, 3, device=device))
    jaw = nn.Parameter(torch.zeros(T, 3, device=device))
    translation = nn.Parameter(torch.zeros(T, 3, device=device))

    # Initialize translation to roughly center the face
    with torch.no_grad():
        translation[:, 2] = -5.0  # move face back from camera

    optimizer = optim.Adam([
        {"params": shape, "lr": lr * 0.1},
        {"params": expr, "lr": lr},
        {"params": rotation, "lr": lr * 0.5},
        {"params": jaw, "lr": lr},
        {"params": translation, "lr": lr},
    ])

    # Prepare target landmarks
    target_lmks = torch.zeros(T, len(MEDIAPIPE_TO_68), 2, device=device)
    valid_mask = torch.zeros(T, dtype=torch.bool, device=device)
    for i in valid_indices:
        lmk = landmarks_2d_list[i]
        # Normalize to [-1, 1] range
        target_lmks[i, :, 0] = torch.tensor(lmk[:, 0] / W * 2 - 1, device=device)
        target_lmks[i, :, 1] = torch.tensor(lmk[:, 1] / H * 2 - 1, device=device)
        valid_mask[i] = True

    n_lmk = min(len(MEDIAPIPE_TO_68), flame.lmk_faces_idx.shape[0])

    print(f"[flame_fitter] Fitting FLAME to {len(valid_indices)} frames …")
    for iteration in tqdm(range(n_iters), desc="Fitting"):
        optimizer.zero_grad()

        # Forward pass — batch all frames
        shape_expanded = shape.expand(T, -1)
        landmarks_3d = flame(shape_expanded, expr, rotation, jaw, translation)

        # Simple orthographic projection
        proj_x = landmarks_3d[:, :n_lmk, 0] / (-landmarks_3d[:, :n_lmk, 2] + 1e-8)
        proj_y = landmarks_3d[:, :n_lmk, 1] / (-landmarks_3d[:, :n_lmk, 2] + 1e-8)
        projected = torch.stack([proj_x, proj_y], dim=-1)

        # Landmark loss (only on valid frames)
        target = target_lmks[:, :n_lmk, :]
        diff = (projected - target) ** 2
        lmk_loss = (diff * valid_mask.unsqueeze(-1).unsqueeze(-1)).sum() / max(valid_mask.sum() * n_lmk, 1)

        # Regularization
        reg_shape = (shape ** 2).mean() * 0.001
        reg_expr = (expr ** 2).mean() * 0.0001
        reg_jaw = (jaw ** 2).mean() * 0.001

        # Temporal smoothness
        if T > 1:
            smooth_expr = ((expr[1:] - expr[:-1]) ** 2).mean() * 0.01
            smooth_jaw = ((jaw[1:] - jaw[:-1]) ** 2).mean() * 0.01
            smooth_rot = ((rotation[1:] - rotation[:-1]) ** 2).mean() * 0.01
            smooth_trans = ((translation[1:] - translation[:-1]) ** 2).mean() * 0.01
        else:
            smooth_expr = smooth_jaw = smooth_rot = smooth_trans = 0

        loss = lmk_loss + reg_shape + reg_expr + reg_jaw + smooth_expr + smooth_jaw + smooth_rot + smooth_trans
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            tqdm.write(f"  iter {iteration+1}/{n_iters} — loss: {loss.item():.6f}")

    # Collect results
    with torch.no_grad():
        # Pad shape to 300
        shape_full = torch.zeros(300, device=device)
        shape_full[:n_shape] = shape[0]

        # Pad expression to 100
        expr_full = torch.zeros(T, 100, device=device)
        expr_full[:, :n_expr] = expr

        result = {
            "shape": shape_full.cpu().numpy(),
            "expr": expr_full.cpu().numpy(),
            "rotation": rotation.cpu().numpy(),
            "neck_pose": np.zeros((T, 3), dtype=np.float32),
            "jaw_pose": jaw.cpu().numpy(),
            "eyes_pose": np.zeros((T, 6), dtype=np.float32),
            "translation": translation.cpu().numpy(),
            "static_offset": np.zeros((1, 5143, 3), dtype=np.float32),
            "dynamic_offset": np.zeros((T, 5143, 3), dtype=np.float32),
        }

    print(f"[flame_fitter] Fitting complete.")
    return result


def fit_video(
    images_dir: str,
    output_path: str,
    device: str = "cuda",
    n_iters: int = 200,
):
    """Full pipeline: detect landmarks → fit FLAME → save .npz."""
    if not FLAME_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FLAME model not found at: {FLAME_MODEL_PATH}\n"
            "Copy flame2023.pkl to gaussian_avatars_repo/flame_model/assets/flame/"
        )

    # Detect landmarks
    landmarks = detect_landmarks_mediapipe(images_dir)

    # Get image size
    frames = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    sample = cv2.imread(os.path.join(images_dir, frames[0]))
    h, w = sample.shape[:2]

    # Fit FLAME
    result = fit_flame_to_landmarks(
        landmarks, (w, h), str(FLAME_MODEL_PATH),
        n_iters=n_iters, device=device,
    )

    # Save
    np.savez(output_path, **result)
    print(f"[flame_fitter] Saved FLAME params: {output_path}")
    print(f"  Frames: {len(landmarks)}")
    print(f"  Shape params: {result['shape'].shape}")
    print(f"  Expr params:  {result['expr'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Fit FLAME 2023 to video frames.")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_iters", type=int, default=200)
    args = parser.parse_args()

    fit_video(args.images_dir, args.output, args.device, args.n_iters)


if __name__ == "__main__":
    main()
