"""
app.py — 4D OMFS Surgical Planning & Prediction Platform.

Streamlit dashboard with two tabs:
  Tab 1 (Clinical Engine)  – Bone segmentation, cutting, and movement.
  Tab 2 (Visual Engine)    – Video-based skin prediction driven by Tab 1 values.

Run:
    streamlit run app.py
"""

import os
import subprocess
import tempfile
from pathlib import Path

import sys
import streamlit as st
import numpy as np
import pyvista as pv
from stpyvista import stpyvista

# Ensure Clinical Engine modules are importable
sys.path.insert(0, str(Path(__file__).parent / "01_Clinical_Engine"))

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
# Path to the Python executable for the Visual Engine.
# By default, uses the same venv as the app itself.
# If you use a separate environment for GPU/torch, update this path.
VISUAL_PYTHON_PATH = str(
    Path(__file__).parent / "venv" / "Scripts" / "python.exe"
)

PROJECT_ROOT = Path(__file__).parent.resolve()
CLINICAL_DIR = PROJECT_ROOT / "01_Clinical_Engine"
VISUAL_DIR = PROJECT_ROOT / "02_Visual_Engine"

# ──────────────────────────────────────────────────────────────
# PyVista config for Streamlit (off-screen rendering)
# ──────────────────────────────────────────────────────────────
# pv.start_xvfb() is only needed on headless Linux — skip on Windows
pv.global_theme.background = "white"
pv.global_theme.anti_aliasing = "msaa"

# ──────────────────────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────────────────────
if "maxilla_mm" not in st.session_state:
    st.session_state.maxilla_mm = 0.0
if "mandible_mm" not in st.session_state:
    st.session_state.mandible_mm = 0.0
if "skull_mesh" not in st.session_state:
    st.session_state.skull_mesh = None        # combined / single mesh
if "maxilla_mesh" not in st.session_state:
    st.session_state.maxilla_mesh = None      # separate upper jaw
if "mandible_mesh" not in st.session_state:
    st.session_state.mandible_mesh = None     # separate lower jaw
if "cut_result" not in st.session_state:
    st.session_state.cut_result = None
if "cutter" not in st.session_state:
    st.session_state.cutter = None
if "last_cut_signature" not in st.session_state:
    st.session_state.last_cut_signature = None
if "preview_signature" not in st.session_state:
    st.session_state.preview_signature = None
if "preview_planes" not in st.session_state:
    st.session_state.preview_planes = None
if "moved_signature" not in st.session_state:
    st.session_state.moved_signature = None
if "moved_segments" not in st.session_state:
    st.session_state.moved_segments = None


def reset_clinical_state() -> None:
    """Reset loaded meshes and derived cutting/movement state."""
    st.session_state.maxilla_mesh = None
    st.session_state.mandible_mesh = None
    st.session_state.cut_result = None
    st.session_state.cutter = None
    st.session_state.last_cut_signature = None
    st.session_state.preview_signature = None
    st.session_state.preview_planes = None
    st.session_state.moved_signature = None
    st.session_state.moved_segments = None


def run_external_command(
    cmd: list[str],
    spinner_text: str,
    success_text: str,
    failure_text: str,
) -> None:
    """Run a subprocess and render concise stdout/stderr diagnostics."""
    st.info(f"Running: `{' '.join(cmd)}`")
    with st.spinner(spinner_text):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

    if result.returncode == 0:
        st.success(success_text)
        if result.stdout:
            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    else:
        st.error(failure_text)
        payload = result.stderr or result.stdout
        if payload:
            st.code(payload[-2000:] if len(payload) > 2000 else payload)

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="4D OMFS Surgical Planning",
    page_icon="🦴",
    layout="wide",
)

st.title("🦴 4D OMFS Surgical Planning & Prediction Platform")

tab_plan, tab_predict = st.tabs(["📐 Surgical Planning", "🎬 Visual Prediction"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — CLINICAL ENGINE
# ══════════════════════════════════════════════════════════════
with tab_plan:
    st.header("Step 1 — Load Skull Mesh")
    st.markdown(
        "Upload a **DICOM folder** for automatic segmentation, "
        "or upload a pre-existing **.stl / .ply** skull mesh."
    )

    col_upload, col_demo = st.columns(2)

    with col_upload:
        uploaded_mesh_file = st.file_uploader(
            "Upload skull mesh (.stl / .ply)",
            type=["stl", "ply"],
            key="mesh_upload",
        )
        if uploaded_mesh_file is not None:
            with tempfile.NamedTemporaryFile(
                suffix=f".{uploaded_mesh_file.name.split('.')[-1]}",
                delete=False,
            ) as tmp:
                tmp.write(uploaded_mesh_file.getvalue())
                tmp_path = tmp.name
            st.session_state.skull_mesh = pv.read(tmp_path)
            reset_clinical_state()
            os.unlink(tmp_path)
            st.success(
                f"Mesh loaded: {st.session_state.skull_mesh.n_points:,} points"
            )

    with col_demo:
        if st.button("🧪 Load Demo Skull (Sphere)", key="demo_skull"):
            demo = pv.Sphere(radius=80, center=(0, 0, 0), theta_resolution=60,
                             phi_resolution=60)
            st.session_state.skull_mesh = demo
            reset_clinical_state()
            st.success("Demo sphere loaded.")

    # ── DICOM Upload ────────────────────────────────────────
    st.divider()
    st.subheader("Or: Load from DICOM CT/CBCT Scan")
    st.markdown(
        "Upload your DICOM files (`.dcm`). The app will extract bone "
        "using Hounsfield Unit thresholding — **no AI weights required**."
    )

    uploaded_dicoms = st.file_uploader(
        "Upload DICOM files",
        type=["dcm", "ima"],
        accept_multiple_files=True,
        key="dicom_upload",
    )

    col_hu, col_smooth = st.columns(2)
    with col_hu:
        hu_threshold = st.slider(
            "Bone HU Threshold",
            min_value=100.0, max_value=1500.0,
            value=300.0, step=50.0,
            key="hu_threshold",
            help="300 = cancellous bone, 700 = cortical bone. Lower → more geometry.",
        )
    with col_smooth:
        smooth_iters = st.slider(
            "Smoothing Iterations",
            min_value=0, max_value=100,
            value=30, step=5,
            key="smooth_iters",
            help="More iterations = smoother surface.",
        )

    if uploaded_dicoms and st.button("🦷 Extract Bone from DICOM", key="run_dicom"):
        # Save uploaded files to a temp folder
        dicom_tmp = os.path.join(tempfile.mkdtemp(prefix="dicom_"), "slices")
        os.makedirs(dicom_tmp, exist_ok=True)
        for dcm_file in uploaded_dicoms:
            save_path = os.path.join(dicom_tmp, dcm_file.name)
            with open(save_path, "wb") as f:
                f.write(dcm_file.getvalue())

        with st.spinner(f"Processing {len(uploaded_dicoms)} DICOM slices …"):
            try:
                from dicom_loader import dicom_to_bone_mesh
                mesh = dicom_to_bone_mesh(
                    dicom_tmp,
                    hu_threshold=hu_threshold,
                    smooth_iterations=smooth_iters,
                )
                st.session_state.skull_mesh = mesh
                reset_clinical_state()
                st.success(
                    f"Bone mesh extracted: "
                    f"{mesh.n_points:,} vertices, "
                    f"{mesh.n_faces_strict:,} triangles"
                )
            except Exception as exc:
                st.error(f"DICOM processing failed: {exc}")

        # Clean up temp files
        import shutil
        shutil.rmtree(os.path.dirname(dicom_tmp), ignore_errors=True)

    # ── ToothFairy3 / NIfTI Loading ──────────────────────────
    st.divider()
    st.subheader("Or: Load from ToothFairy3 / NIfTI File")
    st.markdown(
        "Load a `.nii.gz` file directly. Use a **label file** from ToothFairy3 "
        "(pre-segmented jawbones) or a raw **CBCT image** for HU thresholding."
    )

    nifti_mode = st.radio(
        "NIfTI file type",
        ["Label file (pre-segmented — recommended)", "Raw CBCT image"],
        key="nifti_mode",
        horizontal=True,
    )

    nifti_path = st.text_input(
        "Path to .nii.gz file",
        value="",
        key="nifti_path",
        placeholder=r"C:\Users\lacha\Desktop\ToothFairy3\labelsTr\ToothFairy3F_001.nii.gz",
    )

    if nifti_mode == "Label file (pre-segmented — recommended)":
        st.markdown(
            "**ToothFairy3 labels:** Select which structures to include:"
        )
        col_lab1, col_lab2, col_lab3 = st.columns(3)
        with col_lab1:
            inc_lower = st.checkbox("Lower Jawbone (mandible)", value=True, key="inc_lower")
        with col_lab2:
            inc_upper = st.checkbox("Upper Jawbone (maxilla)", value=True, key="inc_upper")
        with col_lab3:
            inc_teeth = st.checkbox("Teeth (all)", value=True, key="inc_teeth")

        upper_label_ids = []
        lower_label_ids = []
        if inc_lower:
            lower_label_ids.append(1)
        if inc_upper:
            upper_label_ids.append(2)
        if inc_teeth:
            from dicom_loader import LOWER_TEETH_LABELS, UPPER_TEETH_LABELS
            upper_label_ids.extend(UPPER_TEETH_LABELS)
            lower_label_ids.extend(LOWER_TEETH_LABELS)

    if nifti_path and st.button("🦴 Load NIfTI Bone Mesh", key="run_nifti"):
        if not os.path.isfile(nifti_path):
            st.error(f"File not found: `{nifti_path}`")
        else:
            with st.spinner("Extracting bone mesh from NIfTI …"):
                try:
                    if nifti_mode == "Label file (pre-segmented — recommended)":
                        from dicom_loader import nifti_label_to_separate_meshes
                        if not upper_label_ids and not lower_label_ids:
                            raise ValueError(
                                "No labels selected. Enable at least one jaw or teeth set."
                            )
                        meshes = nifti_label_to_separate_meshes(
                            nifti_path,
                            include_upper_labels=upper_label_ids,
                            include_lower_labels=lower_label_ids,
                            smooth_iterations=30,
                            decimate_fraction=0.5,
                        )
                        reset_clinical_state()
                        st.session_state.maxilla_mesh = meshes["maxilla_mesh"]
                        st.session_state.mandible_mesh = meshes["mandible_mesh"]
                        st.session_state.skull_mesh = meshes["combined_mesh"]
                        n_max = meshes["maxilla_mesh"].n_points
                        n_man = meshes["mandible_mesh"].n_points
                        st.success(
                            f"Loaded separately — "
                            f"Maxilla: {n_max:,} pts · "
                            f"Mandible: {n_man:,} pts"
                        )
                    else:
                        from dicom_loader import nifti_image_to_bone_mesh
                        mesh = nifti_image_to_bone_mesh(
                            nifti_path,
                            hu_threshold=300.0,
                            smooth_iterations=30,
                            decimate_fraction=0.5,
                        )
                        reset_clinical_state()
                        st.session_state.skull_mesh = mesh
                        st.success(
                            f"Bone mesh loaded: "
                            f"{mesh.n_points:,} vertices, "
                            f"{mesh.n_faces_strict:,} triangles"
                        )
                except Exception as exc:
                    st.error(f"NIfTI processing failed: {exc}")

    # ── Guard: need a mesh to continue ───────────────────────
    if st.session_state.skull_mesh is None:
        st.info("⬆️ Upload or generate a skull mesh to continue.")

    if st.session_state.skull_mesh is not None:
        from surgical_sim import SurgicalCutter

        st.header("Step 2 — Define Cut Planes & Perform Osteotomies")

        has_separate = (
            st.session_state.maxilla_mesh is not None
            and st.session_state.mandible_mesh is not None
        )
        if has_separate:
            st.success("✅ Separate maxilla & mandible loaded — cuts will be anatomically correct.")
        else:
            st.warning("⚠️ Single mesh mode — for best results, load from ToothFairy3 labels.")

        bounds = st.session_state.skull_mesh.bounds
        x_min, x_max = float(bounds[0]), float(bounds[1])
        y_min, y_max = float(bounds[2]), float(bounds[3])
        z_min, z_max = float(bounds[4]), float(bounds[5])
        x_mid = (x_min + x_max) / 2.0
        z_mid = (z_min + z_max) / 2.0

        st.info(
            f"📏 Mesh bounds — "
            f"X: [{x_min:.1f}, {x_max:.1f}]  ·  "
            f"Y: [{y_min:.1f}, {y_max:.1f}]  ·  "
            f"Z: [{z_min:.1f}, {z_max:.1f}]"
        )

        col_sliders, col_preview = st.columns([1, 2])

        with col_sliders:
            st.subheader("3 Cut Planes")
            st.markdown("**🔴 Le Fort I** — horizontal cut through maxilla")
            lefort_z = st.slider("Le Fort I height (Z)", float(z_min), float(z_max),
                                  float(z_mid + (z_max - z_mid) * 0.3), 0.5, key="lefort_z")
            lefort_flip = st.checkbox("Flip Le Fort mobile side", False, key="lefort_flip",
                                       help="Use this if the wrong maxillary side is being freed.")

            st.markdown("**🔵 BSSO** — sagittal cuts through mandibular rami")
            bsso_l_x = st.slider("BSSO Left (X)", float(x_min), float(x_mid),
                                  float(x_min + (x_mid - x_min) * 0.3), 0.5, key="bsso_l_x")
            bsso_r_x = st.slider("BSSO Right (X)", float(x_mid), float(x_max),
                                  float(x_mid + (x_max - x_mid) * 0.7), 0.5, key="bsso_r_x")

            with st.expander("🔧 Plane Angle Controls"):
                st.caption("Tilt planes from their default orientation (degrees).")
                lefort_pitch = st.slider("Le Fort Pitch", -45.0, 45.0, 0.0, 1.0, key="lefort_pitch")
                lefort_yaw = st.slider("Le Fort Yaw", -45.0, 45.0, 0.0, 1.0, key="lefort_yaw")
                bsso_l_pitch = st.slider("BSSO-L Pitch", -45.0, 45.0, 0.0, 1.0, key="bsso_l_pitch")
                bsso_l_yaw = st.slider("BSSO-L Yaw", -45.0, 45.0, 0.0, 1.0, key="bsso_l_yaw")
                bsso_r_pitch = st.slider("BSSO-R Pitch", -45.0, 45.0, 0.0, 1.0, key="bsso_r_pitch")
                bsso_r_yaw = st.slider("BSSO-R Yaw", -45.0, 45.0, 0.0, 1.0, key="bsso_r_yaw")

            perform_cut = st.button("✂️ Perform Osteotomies", key="perform_cut", type="primary")

        _cut_args = (lefort_z, bsso_l_x, bsso_r_x,
                     lefort_pitch, lefort_yaw, bsso_l_pitch, bsso_l_yaw, bsso_r_pitch, bsso_r_yaw)

        if has_separate:
            cutter = SurgicalCutter(st.session_state.maxilla_mesh, st.session_state.mandible_mesh)
        else:
            cutter = SurgicalCutter(st.session_state.skull_mesh)

        planes = cutter.preview_planes(*_cut_args)

        with col_preview:
            st.subheader("Cut Plane Preview")
            st.caption("🖱️ Left-drag to rotate · Right-drag to pan · Scroll to zoom")
            plotter = pv.Plotter(window_size=(700, 500))
            if has_separate:
                plotter.add_mesh(planes["maxilla"], color="lightyellow", opacity=0.6, label="Maxilla")
                if planes["mandible"] is not None:
                    plotter.add_mesh(planes["mandible"], color="lightcyan", opacity=0.6, label="Mandible")
            else:
                plotter.add_mesh(planes["combined"], color="ivory", opacity=0.6)
            plotter.add_mesh(planes["lefort"], color="red", opacity=0.3, label="Le Fort I")
            plotter.add_mesh(planes["bsso_l"], color="blue", opacity=0.3, label="BSSO Left")
            plotter.add_mesh(planes["bsso_r"], color="blue", opacity=0.3, label="BSSO Right")
            plotter.add_legend()
            plotter.camera_position = "xz"
            plotter.background_color = "white"
            _preview_key = f"preview_3d_{hash((_cut_args, lefort_flip, has_separate))}"
            stpyvista(plotter, key=_preview_key)

        # --- Perform cut ---
        if perform_cut:
            with st.spinner("Cutting bone …"):
                result = cutter.perform_cut(*_cut_args, lefort_flip=lefort_flip)
                st.session_state.cut_result = result
                st.session_state.cutter = cutter
                n_max = result["mobile_maxilla"].n_points if result["mobile_maxilla"] is not None else 0
                n_dist = result["distal_mandible"].n_points if result["distal_mandible"] is not None else 0
                st.success(f"Osteotomies complete! Maxilla: {n_max:,} · Mandible: {n_dist:,}")
                if n_max == 0 or n_dist == 0:
                    st.warning("⚠️ A mobile segment is empty — adjust the plane positions.")
        elif st.session_state.cut_result is not None:
            cutter.perform_cut(*_cut_args, lefort_flip=lefort_flip)
            st.session_state.cutter = cutter

        # ── Step 3 ────────────────────────────────────────────
        if st.session_state.cut_result is None:
            st.info("⬆️ Adjust planes and click **Perform Osteotomies** to continue.")

        if st.session_state.cut_result is not None:
            st.header("Step 3 — Move Segments")

            col_move_sliders, col_move_vis = st.columns([1, 2])

            with col_move_sliders:
                st.subheader("Advancement (mm)")
                st.session_state.maxilla_mm = st.slider(
                    "Maxilla Advancement (Le Fort I)",
                    min_value=-15.0, max_value=15.0,
                    value=st.session_state.maxilla_mm,
                    step=0.5, key="slider_maxilla",
                )
                st.session_state.mandible_mm = st.slider(
                    "Distal Mandible Advancement (BSSO)",
                    min_value=-15.0, max_value=15.0,
                    value=st.session_state.mandible_mm,
                    step=0.5, key="slider_mandible",
                )
                move_axis = st.selectbox(
                    "Advancement direction",
                    options=["+Y (anterior)", "-Y (posterior)", "+X (left)",
                             "-X (right)", "+Z (superior)", "-Z (inferior)"],
                    index=0, key="move_axis",
                )
                axis_vectors = {
                    "+Y (anterior)": (0.0, 1.0, 0.0),
                    "-Y (posterior)": (0.0, -1.0, 0.0),
                    "+X (left)": (1.0, 0.0, 0.0),
                    "-X (right)": (-1.0, 0.0, 0.0),
                    "+Z (superior)": (0.0, 0.0, 1.0),
                    "-Z (inferior)": (0.0, 0.0, -1.0),
                }
                advancement_direction = axis_vectors[move_axis]
                st.metric("Maxilla", f"{st.session_state.maxilla_mm:+.1f} mm")
                st.metric("Distal Mandible", f"{st.session_state.mandible_mm:+.1f} mm")

            with col_move_vis:
                st.subheader("Post-Osteotomy Preview")
                st.caption("🖱️ Left-drag to rotate · Right-drag to pan · Scroll to zoom")
                moved = st.session_state.cutter.move_segments(
                    maxilla_mm=st.session_state.maxilla_mm,
                    mandible_mm=st.session_state.mandible_mm,
                    advancement_direction=advancement_direction,
                )
                plotter2 = pv.Plotter(window_size=(700, 500))
                for seg_key, seg_color, seg_label in [
                    ("upper_skull", "ivory", "Upper Skull (fixed)"),
                    ("proximal_rami", "lightgrey", "Proximal Rami (fixed)"),
                ]:
                    seg = moved.get(seg_key)
                    if seg is not None and seg.n_points > 0:
                        plotter2.add_mesh(seg, color=seg_color, opacity=0.5, label=seg_label)
                seg_max = moved.get("mobile_maxilla")
                if seg_max is not None and seg_max.n_points > 0:
                    plotter2.add_mesh(seg_max, color="salmon", opacity=0.9, label="Maxilla (mobile)")
                seg_mand = moved.get("distal_mandible")
                if seg_mand is not None and seg_mand.n_points > 0:
                    plotter2.add_mesh(seg_mand, color="cornflowerblue", opacity=0.9,
                                      label="Distal Mandible (mobile)")
                plotter2.add_legend()
                plotter2.camera_position = "xz"
                plotter2.background_color = "white"
                _move_key = f"moved_3d_{st.session_state.maxilla_mm}_{st.session_state.mandible_mm}_{move_axis}"
                stpyvista(plotter2, key=_move_key)


# ══════════════════════════════════════════════════════════════
# TAB 2 — VISUAL ENGINE
# ══════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Visual Prediction Engine")
    st.markdown(
        "This tab reads the **Maxilla** and **Mandible** advancement values "
        "from the Planning tab and renders a predicted post-surgical video."
    )

    # ── Show current surgical plan values ────────────────────
    st.subheader("Current Surgical Plan")
    col_v1, col_v2 = st.columns(2)
    col_v1.metric("Maxilla (mm)", f"{st.session_state.maxilla_mm:+.1f}")
    col_v2.metric("Mandible (mm)", f"{st.session_state.mandible_mm:+.1f}")

    st.divider()

    # ── Step 1: Video Input ─────────────────────────────────
    st.subheader("1. Upload Patient Video")
    uploaded_video = st.file_uploader(
        "Upload patient video (.mp4)",
        type=["mp4", "avi", "mov"],
        key="video_upload",
    )
    data_dir = str(VISUAL_DIR / "data")
    input_video_path = os.path.join(data_dir, "input_video.mp4")
    if uploaded_video is not None:
        os.makedirs(data_dir, exist_ok=True)
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.getvalue())
        st.success(f"Video saved to `{input_video_path}`")
        st.video(uploaded_video)

    st.divider()

    # ── Step 2: Preprocess Video ─────────────────────────────
    st.subheader("2. Preprocess Video (Extract Frames + FLAME Params)")
    st.markdown(
        "Extracts frames from the video and creates the dataset structure "
        "needed for GaussianAvatars training."
    )
    col_prep1, col_prep2 = st.columns(2)
    with col_prep1:
        max_frames = st.number_input(
            "Max frames to extract",
            min_value=10, max_value=2000,
            value=300, step=50,
            key="max_frames",
            help="More frames = better quality but longer training. 300 is a good start.",
        )
    with col_prep2:
        target_size = st.selectbox(
            "Frame resolution",
            options=[256, 512, 768, 1024],
            index=1,
            key="target_size",
        )

    transforms_path = os.path.join(data_dir, "transforms_train.json")
    dataset_ready = os.path.exists(transforms_path)

    if dataset_ready:
        st.success("Dataset already preprocessed. Re-run to update.")

    if st.button("📹 Preprocess Video", key="preprocess", type="secondary"):
        if not os.path.exists(input_video_path):
            st.error("Upload a video first (Step 1).")
        else:
            preprocess_cmd = [
                VISUAL_PYTHON_PATH,
                str(VISUAL_DIR / "preprocess_video.py"),
                "--video", input_video_path,
                "--output_dir", data_dir,
                "--max_frames", str(max_frames),
                "--target_size", str(target_size),
            ]
            st.info(f"Running: `{' '.join(preprocess_cmd)}`")
            with st.spinner(f"Extracting up to {max_frames} frames …"):
                result = subprocess.run(
                    preprocess_cmd,
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                )
            if result.returncode == 0:
                st.success("Preprocessing complete!")
                st.code(result.stdout[-2000:] if len(result.stdout) > 2000
                        else result.stdout)
                dataset_ready = True
            else:
                st.error("Preprocessing failed.")
                st.code(result.stderr[-2000:] if len(result.stderr) > 2000
                        else result.stderr)

    st.divider()

    # ── Step 3: Train Model ──────────────────────────────────
    st.subheader("3. Train Gaussian Avatar Model")
    model_output_dir = str(VISUAL_DIR / "output" / "model")
    iterations = st.number_input(
        "Training iterations",
        min_value=1000, max_value=600000,
        value=30000, step=5000,
        key="train_iterations",
        help="30K = quick test (~30 min), 600K = full quality (~8 hrs on RTX 4070 Ti).",
    )

    if st.button("🧠 Train Model", key="train_model", type="secondary"):
        if not dataset_ready:
            st.error("Preprocess the video first (Step 2).")
        else:
            train_cmd = [
                VISUAL_PYTHON_PATH,
                str(VISUAL_DIR / "train_ghost.py"),
                "--data_dir", data_dir,
                "--output_dir", model_output_dir,
                "--iterations", str(iterations),
            ]
            st.info(f"Running: `{' '.join(train_cmd)}`")
            with st.spinner(f"Training model for {iterations:,} iterations … (this will take a while)"):
                result = subprocess.run(
                    train_cmd,
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                )
            if result.returncode == 0:
                st.success("Training complete!")
                st.code(result.stdout[-2000:] if len(result.stdout) > 2000
                        else result.stdout)
            else:
                st.error("Training failed.")
                st.code(result.stderr[-2000:] if len(result.stderr) > 2000
                        else result.stderr)

    st.divider()

    # ── Step 4: Generate Prediction ──────────────────────────
    st.subheader("4. Generate Prediction from Surgical Plan")

    sensitivity = st.slider(
        "Sensitivity Multiplier",
        min_value=0.1, max_value=3.0,
        value=1.0, step=0.1,
        key="sensitivity",
        help="Controls how strongly the mm values map to facial change. "
             "1.0 = direct mapping. Increase if effect is too subtle.",
    )

    output_video_path = str(PROJECT_ROOT / "final_prediction.mp4")

    if st.button(
        "🎬 Generate Prediction from Surgical Plan",
        key="generate_pred",
        type="primary",
    ):
        maxilla_val = st.session_state.maxilla_mm
        mandible_val = st.session_state.mandible_mm

        if maxilla_val == 0.0 and mandible_val == 0.0:
            st.warning(
                "Both advancement values are 0.0 mm. "
                "Go to the Planning tab and set the movement sliders first."
            )
        elif not dataset_ready:
            st.error("Preprocess the video first (Step 2).")
        else:
            render_cmd = [
                VISUAL_PYTHON_PATH,
                str(VISUAL_DIR / "render_surgery.py"),
                "--lefort_mm", str(maxilla_val),
                "--bsso_mm", str(mandible_val),
                "--sensitivity", str(sensitivity),
                "--model_path", model_output_dir,
                "--data_dir", data_dir,
                "--output", output_video_path,
            ]
            st.info(f"Running: `{' '.join(render_cmd)}`")
            with st.spinner("Rendering prediction video …"):
                result = subprocess.run(
                    render_cmd,
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                )
            if result.returncode == 0:
                st.success("Prediction rendered successfully!")
                st.code(result.stdout[-2000:] if len(result.stdout) > 2000
                        else result.stdout)
            else:
                st.error("Rendering failed.")
                st.code(result.stderr[-2000:] if len(result.stderr) > 2000
                        else result.stderr)

    # ── Step 5: Side-by-side Results ─────────────────────────
    st.divider()
    st.subheader("5. Results — Before vs After")

    col_pre, col_post = st.columns(2)

    with col_pre:
        st.markdown("**Pre-Op Video**")
        if os.path.exists(input_video_path):
            st.video(input_video_path)
        else:
            st.info("No pre-op video uploaded yet.")

    with col_post:
        st.markdown("**Post-Op Prediction**")
        if os.path.exists(output_video_path):
            st.video(output_video_path)
        else:
            st.info("No prediction generated yet.")
