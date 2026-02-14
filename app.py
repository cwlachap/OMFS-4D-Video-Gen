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
# Dark theme colors
DARK_BG = "#0e1117"
DARK_SECONDARY = "#1a1f2e"
DARK_ACCENT = "#262c3a"

pv.global_theme.background = DARK_BG
pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.font.color = "white"

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
# Rotation state (pitch, yaw, roll in degrees)
if "maxilla_rotation" not in st.session_state:
    st.session_state.maxilla_rotation = (0.0, 0.0, 0.0)
if "mandible_rotation" not in st.session_state:
    st.session_state.mandible_rotation = (0.0, 0.0, 0.0)
# Movement history for undo/redo
if "movement_history" not in st.session_state:
    st.session_state.movement_history = []
if "history_index" not in st.session_state:
    st.session_state.history_index = -1


def get_current_movement_state() -> dict:
    """Capture the current movement parameters as a state dict."""
    return {
        "maxilla_mm": st.session_state.maxilla_mm,
        "mandible_mm": st.session_state.mandible_mm,
        "maxilla_rotation": st.session_state.maxilla_rotation,
        "mandible_rotation": st.session_state.mandible_rotation,
    }


def restore_movement_state(state: dict) -> None:
    """Restore movement parameters from a state dict."""
    st.session_state.maxilla_mm = state["maxilla_mm"]
    st.session_state.mandible_mm = state["mandible_mm"]
    st.session_state.maxilla_rotation = state["maxilla_rotation"]
    st.session_state.mandible_rotation = state["mandible_rotation"]


def save_movement_to_history() -> None:
    """Save current movement state to history (for undo/redo)."""
    current_state = get_current_movement_state()
    
    # If we're not at the end of history, truncate future states
    if st.session_state.history_index < len(st.session_state.movement_history) - 1:
        st.session_state.movement_history = st.session_state.movement_history[:st.session_state.history_index + 1]
    
    # Don't save duplicate states
    if st.session_state.movement_history:
        last_state = st.session_state.movement_history[-1]
        if last_state == current_state:
            return
    
    st.session_state.movement_history.append(current_state)
    st.session_state.history_index = len(st.session_state.movement_history) - 1
    
    # Limit history to 50 states
    if len(st.session_state.movement_history) > 50:
        st.session_state.movement_history = st.session_state.movement_history[-50:]
        st.session_state.history_index = len(st.session_state.movement_history) - 1


def undo_movement() -> bool:
    """Undo to previous movement state. Returns True if successful."""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        restore_movement_state(st.session_state.movement_history[st.session_state.history_index])
        return True
    return False


def redo_movement() -> bool:
    """Redo to next movement state. Returns True if successful."""
    if st.session_state.history_index < len(st.session_state.movement_history) - 1:
        st.session_state.history_index += 1
        restore_movement_state(st.session_state.movement_history[st.session_state.history_index])
        return True
    return False


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


def get_visual_engine_env() -> dict:
    """Create environment dict for Visual Engine subprocesses.
    
    Ensures CUDA, FFmpeg, venv paths, and nvdiffrast cache are properly set.
    This is critical for VHAP to run at full GPU speed.
    """
    env = os.environ.copy()
    
    # Ensure venv's Scripts directory is at the front of PATH
    venv_scripts = str(PROJECT_ROOT / "venv" / "Scripts")
    venv_lib = str(PROJECT_ROOT / "venv" / "Lib" / "site-packages")
    
    # Add FFmpeg to PATH (common install location on Windows)
    ffmpeg_paths = [
        r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin",
        r"C:\ffmpeg\bin",
    ]
    
    # Build new PATH with venv and ffmpeg at the front
    path_parts = [venv_scripts]
    for ffp in ffmpeg_paths:
        if os.path.isdir(ffp):
            path_parts.append(ffp)
    path_parts.append(env.get("PATH", ""))
    
    env["PATH"] = os.pathsep.join(path_parts)
    
    # Set PYTHONPATH to ensure venv packages are found
    env["PYTHONPATH"] = venv_lib
    
    # Ensure CUDA is visible (if not already set)
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # CRITICAL: Set nvdiffrast cache directory explicitly
    # Without this, Streamlit subprocesses may not find the CUDA kernel cache,
    # causing nvdiffrast to recompile kernels slowly or fall back to CPU
    nvdiffrast_cache = PROJECT_ROOT / ".cache" / "nvdiffrast"
    nvdiffrast_cache.mkdir(parents=True, exist_ok=True)
    env["NVDIFFRAST_CACHE_DIR"] = str(nvdiffrast_cache)
    
    # Also ensure HOME/USERPROFILE are set consistently
    # (nvdiffrast uses these as fallback for cache location)
    if "USERPROFILE" not in env:
        env["USERPROFILE"] = str(Path.home())
    if "HOME" not in env:
        env["HOME"] = str(Path.home())
    
    # Ensure temp directories are accessible
    temp_dir = PROJECT_ROOT / ".cache" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temp_dir)
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    
    return env


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
            env=get_visual_engine_env(),
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
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Dark Mode Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f3f4f6 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3748;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #60a5fa !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: transparent;
        border: 2px solid #3b82f6;
        color: #3b82f6;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1a1f2e;
        border-radius: 8px;
        color: #e5e7eb;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Dividers */
    hr {
        border-color: #2d3748;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #9ca3af;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1a1f2e;
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1a1f2e !important;
    }
    
    /* Selectbox */
    [data-baseweb="select"] {
        background-color: #1a1f2e;
    }
    
    /* Number input */
    [data-testid="stNumberInput"] input {
        background-color: #1a1f2e;
        color: #f3f4f6;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar — Surgical Plan Summary
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bone.png", width=60)
    st.title("Surgical Plan")
    st.caption("Real-time summary of your surgical plan")
    
    st.divider()
    
    # Mesh status
    st.subheader("1. Mesh Status")
    if st.session_state.skull_mesh is not None:
        n_points = st.session_state.skull_mesh.n_points
        st.success(f"Loaded: {n_points:,} vertices")
        if st.session_state.maxilla_mesh is not None:
            st.caption(f"Maxilla: {st.session_state.maxilla_mesh.n_points:,} pts")
        if st.session_state.mandible_mesh is not None:
            st.caption(f"Mandible: {st.session_state.mandible_mesh.n_points:,} pts")
    else:
        st.warning("No mesh loaded")
    
    st.divider()
    
    # Osteotomy status
    st.subheader("2. Osteotomies")
    if st.session_state.cut_result is not None:
        st.success("Cuts performed")
        mobile_max = st.session_state.cut_result.get("mobile_maxilla")
        distal_mand = st.session_state.cut_result.get("distal_mandible")
        if mobile_max is not None:
            st.caption(f"Mobile maxilla: {mobile_max.n_points:,} pts")
        if distal_mand is not None:
            st.caption(f"Distal mandible: {distal_mand.n_points:,} pts")
    else:
        st.info("Pending")
    
    st.divider()
    
    # Movement summary
    st.subheader("3. Planned Movements")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        maxilla_color = "#f97316" if st.session_state.maxilla_mm != 0 else "#6b7280"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: {DARK_SECONDARY}; border-radius: 8px; border-left: 4px solid {maxilla_color};">
            <div style="color: #9ca3af; font-size: 0.8rem;">Maxilla</div>
            <div style="color: {maxilla_color}; font-size: 1.5rem; font-weight: bold;">{st.session_state.maxilla_mm:+.1f}</div>
            <div style="color: #6b7280; font-size: 0.7rem;">mm</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        mandible_color = "#3b82f6" if st.session_state.mandible_mm != 0 else "#6b7280"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: {DARK_SECONDARY}; border-radius: 8px; border-left: 4px solid {mandible_color};">
            <div style="color: #9ca3af; font-size: 0.8rem;">Mandible</div>
            <div style="color: {mandible_color}; font-size: 1.5rem; font-weight: bold;">{st.session_state.mandible_mm:+.1f}</div>
            <div style="color: #6b7280; font-size: 0.7rem;">mm</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show rotation if set
    if st.session_state.get("maxilla_rotation") or st.session_state.get("mandible_rotation"):
        st.caption("Rotations applied")
    
    st.divider()
    
    # Visual Engine status
    st.subheader("4. Visual Engine")
    data_dir = str(VISUAL_DIR / "data")
    transforms_path = os.path.join(data_dir, "transforms_train.json")
    model_path = str(VISUAL_DIR / "output" / "model")
    
    # Check preprocessing status
    if os.path.exists(transforms_path):
        st.success("VHAP: Complete")
    else:
        st.info("VHAP: Not run")
    
    # Check model status
    if os.path.exists(model_path) and os.listdir(model_path) if os.path.exists(model_path) else False:
        st.success("Model: Trained")
    else:
        st.info("Model: Not trained")
    
    st.divider()
    
    # Quick actions
    st.subheader("Quick Actions")
    if st.button("Reset All", type="secondary", use_container_width=True):
        reset_clinical_state()
        st.session_state.maxilla_mm = 0.0
        st.session_state.mandible_mm = 0.0
        st.rerun()

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
                plotter.add_mesh(planes["maxilla"], color="#fbbf24", opacity=0.6, label="Maxilla")
                if planes["mandible"] is not None:
                    plotter.add_mesh(planes["mandible"], color="#22d3d1", opacity=0.6, label="Mandible")
            else:
                plotter.add_mesh(planes["combined"], color="#9ca3af", opacity=0.6)
            plotter.add_mesh(planes["lefort"], color="red", opacity=0.3, label="Le Fort I")
            plotter.add_mesh(planes["bsso_l"], color="blue", opacity=0.3, label="BSSO Left")
            plotter.add_mesh(planes["bsso_r"], color="blue", opacity=0.3, label="BSSO Right")
            plotter.add_legend()
            plotter.camera_position = "xz"
            plotter.background_color = DARK_BG
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
                
                # Advanced rotation controls
                with st.expander("🔄 Advanced: Rotation Controls"):
                    st.caption("Rotate segments around their center (degrees)")
                    
                    st.markdown("**Maxilla Rotation**")
                    col_mr1, col_mr2, col_mr3 = st.columns(3)
                    with col_mr1:
                        max_pitch = st.slider("Pitch (X)", -15.0, 15.0, 
                                              st.session_state.maxilla_rotation[0], 0.5, 
                                              key="max_pitch", help="Tilt forward/backward")
                    with col_mr2:
                        max_yaw = st.slider("Yaw (Z)", -15.0, 15.0, 
                                            st.session_state.maxilla_rotation[1], 0.5, 
                                            key="max_yaw", help="Turn left/right")
                    with col_mr3:
                        max_roll = st.slider("Roll (Y)", -15.0, 15.0, 
                                             st.session_state.maxilla_rotation[2], 0.5, 
                                             key="max_roll", help="Tilt sideways")
                    st.session_state.maxilla_rotation = (max_pitch, max_yaw, max_roll)
                    
                    st.markdown("**Mandible Rotation**")
                    col_md1, col_md2, col_md3 = st.columns(3)
                    with col_md1:
                        mand_pitch = st.slider("Pitch (X)", -15.0, 15.0, 
                                               st.session_state.mandible_rotation[0], 0.5, 
                                               key="mand_pitch", help="Tilt forward/backward")
                    with col_md2:
                        mand_yaw = st.slider("Yaw (Z)", -15.0, 15.0, 
                                             st.session_state.mandible_rotation[1], 0.5, 
                                             key="mand_yaw", help="Turn left/right")
                    with col_md3:
                        mand_roll = st.slider("Roll (Y)", -15.0, 15.0, 
                                              st.session_state.mandible_rotation[2], 0.5, 
                                              key="mand_roll", help="Tilt sideways")
                    st.session_state.mandible_rotation = (mand_pitch, mand_yaw, mand_roll)
                    
                    if st.button("Reset Rotations", key="reset_rot"):
                        st.session_state.maxilla_rotation = (0.0, 0.0, 0.0)
                        st.session_state.mandible_rotation = (0.0, 0.0, 0.0)
                        st.rerun()
                
                st.metric("Maxilla", f"{st.session_state.maxilla_mm:+.1f} mm")
                st.metric("Distal Mandible", f"{st.session_state.mandible_mm:+.1f} mm")
                
                # Undo/Redo controls
                st.divider()
                col_undo, col_redo, col_save = st.columns(3)
                
                with col_undo:
                    can_undo = st.session_state.history_index > 0
                    if st.button("↩️ Undo", disabled=not can_undo, key="undo_btn", use_container_width=True):
                        if undo_movement():
                            st.rerun()
                
                with col_redo:
                    can_redo = st.session_state.history_index < len(st.session_state.movement_history) - 1
                    if st.button("↪️ Redo", disabled=not can_redo, key="redo_btn", use_container_width=True):
                        if redo_movement():
                            st.rerun()
                
                with col_save:
                    if st.button("💾 Save State", key="save_state_btn", use_container_width=True):
                        save_movement_to_history()
                        st.success("State saved!")
                
                # Show history info
                if st.session_state.movement_history:
                    st.caption(f"History: {st.session_state.history_index + 1} / {len(st.session_state.movement_history)} states")

            with col_move_vis:
                st.subheader("Post-Osteotomy Preview")
                st.caption("🖱️ Left-drag to rotate · Right-drag to pan · Scroll to zoom")
                moved = st.session_state.cutter.move_segments(
                    maxilla_mm=st.session_state.maxilla_mm,
                    mandible_mm=st.session_state.mandible_mm,
                    advancement_direction=advancement_direction,
                    maxilla_rotation=st.session_state.maxilla_rotation,
                    mandible_rotation=st.session_state.mandible_rotation,
                )
                plotter2 = pv.Plotter(window_size=(700, 500))
                for seg_key, seg_color, seg_label in [
                    ("upper_skull", "#6b7280", "Upper Skull (fixed)"),
                    ("proximal_rami", "#4b5563", "Proximal Rami (fixed)"),
                ]:
                    seg = moved.get(seg_key)
                    if seg is not None and seg.n_points > 0:
                        plotter2.add_mesh(seg, color=seg_color, opacity=0.5, label=seg_label)
                seg_max = moved.get("mobile_maxilla")
                if seg_max is not None and seg_max.n_points > 0:
                    plotter2.add_mesh(seg_max, color="#f97316", opacity=0.9, label="Maxilla (mobile)")
                seg_mand = moved.get("distal_mandible")
                if seg_mand is not None and seg_mand.n_points > 0:
                    plotter2.add_mesh(seg_mand, color="#3b82f6", opacity=0.9,
                                      label="Distal Mandible (mobile)")
                plotter2.add_legend()
                plotter2.camera_position = "xz"
                plotter2.background_color = DARK_BG
                _move_key = f"moved_3d_{st.session_state.maxilla_mm}_{st.session_state.mandible_mm}_{move_axis}"
                stpyvista(plotter2, key=_move_key)

            # ── STL Export ─────────────────────────────────────────
            st.divider()
            st.subheader("Export Modified Mesh")
            st.markdown(
                "Download the modified skull mesh for 3D printing, "
                "surgical guide fabrication, or further analysis."
            )
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                export_format = st.selectbox(
                    "Export format",
                    options=["STL (Binary)", "STL (ASCII)", "PLY", "OBJ"],
                    key="export_format",
                )
            
            with col_export2:
                export_segments = st.multiselect(
                    "Include segments",
                    options=["Upper Skull", "Mobile Maxilla", "Distal Mandible", "Proximal Rami"],
                    default=["Upper Skull", "Mobile Maxilla", "Distal Mandible", "Proximal Rami"],
                    key="export_segments",
                )
            
            if st.button("📥 Generate Download", type="primary", key="export_btn"):
                # Collect selected segments
                segments_to_merge = []
                segment_map = {
                    "Upper Skull": moved.get("upper_skull"),
                    "Mobile Maxilla": moved.get("mobile_maxilla"),
                    "Distal Mandible": moved.get("distal_mandible"),
                    "Proximal Rami": moved.get("proximal_rami"),
                }
                
                for seg_name in export_segments:
                    seg = segment_map.get(seg_name)
                    if seg is not None and seg.n_points > 0:
                        segments_to_merge.append(seg)
                
                if not segments_to_merge:
                    st.error("No valid segments selected.")
                else:
                    # Merge all selected segments
                    combined_mesh = segments_to_merge[0]
                    for seg in segments_to_merge[1:]:
                        combined_mesh = combined_mesh.merge(seg)
                    
                    # Determine file extension and format
                    format_map = {
                        "STL (Binary)": ("stl", False),
                        "STL (ASCII)": ("stl", True),
                        "PLY": ("ply", None),
                        "OBJ": ("obj", None),
                    }
                    ext, ascii_flag = format_map[export_format]
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        suffix=f".{ext}", delete=False
                    ) as tmp:
                        if ext == "stl":
                            combined_mesh.save(tmp.name, binary=not ascii_flag)
                        else:
                            combined_mesh.save(tmp.name)
                        tmp_path = tmp.name
                    
                    # Read file for download
                    with open(tmp_path, "rb") as f:
                        mesh_data = f.read()
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    # Offer download
                    filename = f"surgical_plan_maxilla{st.session_state.maxilla_mm:+.1f}mm_mandible{st.session_state.mandible_mm:+.1f}mm.{ext}"
                    st.download_button(
                        label=f"⬇️ Download {export_format}",
                        data=mesh_data,
                        file_name=filename,
                        mime="application/octet-stream",
                        type="primary",
                    )
                    st.success(f"Mesh ready: {combined_mesh.n_points:,} vertices, {combined_mesh.n_faces_strict:,} triangles")

            # ── Measurement Tools ─────────────────────────────────────
            st.divider()
            st.subheader("Measurement Tools")
            st.markdown(
                "Measure distances and angles on the mesh. "
                "Enter coordinates manually or use the mesh bounds as reference."
            )
            
            # Initialize measurements in session state
            if "measurements_list" not in st.session_state:
                st.session_state.measurements_list = []
            
            # Show mesh bounds for reference
            combined_for_bounds = st.session_state.skull_mesh
            if combined_for_bounds is not None:
                bounds = combined_for_bounds.bounds
                with st.expander("📏 Mesh bounds (for reference)"):
                    col_b1, col_b2, col_b3 = st.columns(3)
                    col_b1.metric("X range", f"{bounds[0]:.1f} to {bounds[1]:.1f}")
                    col_b2.metric("Y range", f"{bounds[2]:.1f} to {bounds[3]:.1f}")
                    col_b3.metric("Z range", f"{bounds[4]:.1f} to {bounds[5]:.1f}")
            
            measurement_type = st.radio(
                "Measurement type",
                options=["Distance (2 points)", "Angle (3 points)"],
                horizontal=True,
                key="measurement_type",
            )
            
            if measurement_type == "Distance (2 points)":
                st.markdown("**Point A**")
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    ax = st.number_input("X", value=0.0, key="dist_ax", format="%.2f")
                with col_a2:
                    ay = st.number_input("Y", value=0.0, key="dist_ay", format="%.2f")
                with col_a3:
                    az = st.number_input("Z", value=0.0, key="dist_az", format="%.2f")
                
                st.markdown("**Point B**")
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    bx = st.number_input("X", value=10.0, key="dist_bx", format="%.2f")
                with col_b2:
                    by = st.number_input("Y", value=0.0, key="dist_by", format="%.2f")
                with col_b3:
                    bz = st.number_input("Z", value=0.0, key="dist_bz", format="%.2f")
                
                # Calculate distance
                point_a = np.array([ax, ay, az])
                point_b = np.array([bx, by, bz])
                distance = np.linalg.norm(point_b - point_a)
                
                col_result, col_save = st.columns([2, 1])
                with col_result:
                    st.metric("Distance", f"{distance:.2f} mm")
                with col_save:
                    if st.button("Save measurement", key="save_dist"):
                        st.session_state.measurements_list.append({
                            "type": "Distance",
                            "points": [point_a.tolist(), point_b.tolist()],
                            "value": f"{distance:.2f} mm",
                        })
                        st.success("Measurement saved!")
            
            else:  # Angle (3 points)
                st.markdown("**Point A** (first arm endpoint)")
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    ax = st.number_input("X", value=-10.0, key="ang_ax", format="%.2f")
                with col_a2:
                    ay = st.number_input("Y", value=0.0, key="ang_ay", format="%.2f")
                with col_a3:
                    az = st.number_input("Z", value=0.0, key="ang_az", format="%.2f")
                
                st.markdown("**Point B** (vertex / apex)")
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    bx = st.number_input("X", value=0.0, key="ang_bx", format="%.2f")
                with col_b2:
                    by = st.number_input("Y", value=0.0, key="ang_by", format="%.2f")
                with col_b3:
                    bz = st.number_input("Z", value=0.0, key="ang_bz", format="%.2f")
                
                st.markdown("**Point C** (second arm endpoint)")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    cx = st.number_input("X", value=10.0, key="ang_cx", format="%.2f")
                with col_c2:
                    cy = st.number_input("Y", value=0.0, key="ang_cy", format="%.2f")
                with col_c3:
                    cz = st.number_input("Z", value=0.0, key="ang_cz", format="%.2f")
                
                # Calculate angle
                point_a = np.array([ax, ay, az])
                point_b = np.array([bx, by, bz])  # vertex
                point_c = np.array([cx, cy, cz])
                
                vec_ba = point_a - point_b
                vec_bc = point_c - point_b
                
                # Avoid division by zero
                norm_ba = np.linalg.norm(vec_ba)
                norm_bc = np.linalg.norm(vec_bc)
                
                if norm_ba > 1e-10 and norm_bc > 1e-10:
                    cos_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                else:
                    angle_deg = 0.0
                
                col_result, col_save = st.columns([2, 1])
                with col_result:
                    st.metric("Angle", f"{angle_deg:.1f}°")
                with col_save:
                    if st.button("Save measurement", key="save_ang"):
                        st.session_state.measurements_list.append({
                            "type": "Angle",
                            "points": [point_a.tolist(), point_b.tolist(), point_c.tolist()],
                            "value": f"{angle_deg:.1f}°",
                        })
                        st.success("Measurement saved!")
            
            # Show saved measurements
            if st.session_state.measurements_list:
                st.markdown("**Saved Measurements**")
                for i, m in enumerate(st.session_state.measurements_list):
                    col_m1, col_m2 = st.columns([3, 1])
                    with col_m1:
                        st.text(f"{i+1}. {m['type']}: {m['value']}")
                    with col_m2:
                        if st.button("🗑️", key=f"del_m_{i}"):
                            st.session_state.measurements_list.pop(i)
                            st.rerun()
                
                if st.button("Clear all measurements", key="clear_measurements"):
                    st.session_state.measurements_list = []
                    st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2 — VISUAL ENGINE (VHAP-powered)
# ══════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Visual Prediction Engine")
    st.markdown(
        "This tab reads the **Maxilla** and **Mandible** advancement values "
        "from the Planning tab and renders a predicted post-surgical video.\n\n"
        "**Powered by [VHAP](https://github.com/ShenhanQian/VHAP)** — "
        "Versatile Head Alignment with Adaptive Appearance Priors for accurate FLAME tracking."
    )

    # ── Show current surgical plan values ────────────────────
    st.subheader("Current Surgical Plan")
    col_v1, col_v2 = st.columns(2)
    col_v1.metric("Maxilla (mm)", f"{st.session_state.maxilla_mm:+.1f}")
    col_v2.metric("Mandible (mm)", f"{st.session_state.mandible_mm:+.1f}")

    st.divider()

    # ── Step 1: Video Input ─────────────────────────────────
    st.subheader("1. Upload Patient Video")
    st.markdown(
        "Upload a front-facing video of the patient. Best results with:\n"
        "- Good lighting, neutral background\n"
        "- Face clearly visible throughout\n"
        "- 10-30 seconds of footage"
    )
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

    # ── Step 2: Preprocess Video with VHAP ─────────────────────
    st.subheader("2. Preprocess Video with VHAP")
    st.markdown(
        "**VHAP Pipeline** (may take 10-30 minutes depending on video length):\n"
        "1. **Frame extraction** — Extract frames from video\n"
        "2. **Background matting** — Segment foreground from background\n"
        "3. **Landmark detection** — Detect facial landmarks\n"
        "4. **FLAME tracking** — Fit 3D face model with photometric optimization\n"
        "5. **Export** — Convert to GaussianAvatars format"
    )
    
    col_prep1, col_prep2 = st.columns(2)
    with col_prep1:
        target_size = st.selectbox(
            "Frame resolution",
            options=[256, 512, 768, 1024],
            index=1,
            key="target_size",
            help="Lower = faster processing, higher = better quality.",
        )
    with col_prep2:
        use_static_offset = st.checkbox(
            "Use static offset",
            value=True,
            key="use_static_offset",
            help="Enable for better facial geometry alignment. Disable if tracking fails.",
        )

    transforms_path = os.path.join(data_dir, "transforms_train.json")
    flame_param_path = os.path.join(data_dir, "flame_param.npz")
    dataset_ready = os.path.exists(transforms_path) and os.path.exists(flame_param_path)

    if dataset_ready:
        st.success("✅ Dataset already preprocessed with VHAP. Re-run to update.")
        # Show some stats
        try:
            params = np.load(flame_param_path)
            n_frames = params["expr"].shape[0]
            st.info(f"📊 Dataset: {n_frames} frames with FLAME parameters")
        except Exception:
            pass

    if st.button("🔬 Run VHAP Preprocessing", key="preprocess", type="primary"):
        if not os.path.exists(input_video_path):
            st.error("Upload a video first (Step 1).")
        else:
            preprocess_cmd = [
                VISUAL_PYTHON_PATH,
                str(VISUAL_DIR / "preprocess_video.py"),
                "--video", input_video_path,
                "--output_dir", data_dir,
                "--target_size", str(target_size),
            ]
            if not use_static_offset:
                preprocess_cmd.append("--no-static-offset")
            
            st.info(f"Running VHAP pipeline... This may take 10-30 minutes.")
            
            # Progress tracking for VHAP stages
            progress_bar = st.progress(0, text="Starting VHAP preprocessing...")
            stage_status = st.empty()
            
            # Create a placeholder for streaming output
            with st.expander("View detailed output", expanded=False):
                output_placeholder = st.empty()
            
            # Get proper environment with CUDA, FFmpeg, and venv paths
            visual_env = get_visual_engine_env()
            
            # VHAP stage detection patterns and progress values
            vhap_stages = [
                ("Extracting frames", 5, "Extracting video frames..."),
                ("BackgroundMattingV2", 15, "Background matting..."),
                ("face_alignment", 25, "Detecting facial landmarks..."),
                ("Start sequential tracking", 35, "Starting FLAME tracking..."),
                ("train-lmk_init_rigid", 40, "Landmark init (rigid)..."),
                ("train-lmk_init_all", 45, "Landmark init (all)..."),
                ("train-rgb_init_texture", 50, "RGB texture optimization..."),
                ("train-rgb_init_all", 60, "RGB full optimization..."),
                ("train-rgb_init_offset", 65, "RGB offset optimization..."),
                ("train-rgb_sequential_tracking", 70, "Sequential tracking..."),
                ("Start global optimization", 80, "Global optimization..."),
                ("EPOCH", 85, "Global optimization epochs..."),
                ("Exporting", 95, "Exporting to GaussianAvatars format..."),
            ]
            current_progress = 0
            
            # Run with streaming output
            process = subprocess.Popen(
                preprocess_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                env=visual_env,
                bufsize=1,
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                
                # Update progress based on output patterns
                for pattern, progress_val, status_text in vhap_stages:
                    if pattern in line and progress_val > current_progress:
                        current_progress = progress_val
                        progress_bar.progress(current_progress, text=status_text)
                        stage_status.markdown(f"**Current stage:** {status_text}")
                        break
                
                # Keep last 50 lines for display
                display_text = ''.join(output_lines[-50:])
                output_placeholder.code(display_text, language="")
            
            process.wait()
                
            if process.returncode == 0:
                progress_bar.progress(100, text="VHAP preprocessing complete!")
                st.success("✅ VHAP preprocessing complete!")
                dataset_ready = True
                st.balloons()
            else:
                progress_bar.progress(current_progress, text="VHAP preprocessing failed")
                st.error("❌ VHAP preprocessing failed. Check the output above for errors.")
                full_output = ''.join(output_lines)
                with st.expander("Full output"):
                    st.code(full_output[-5000:] if len(full_output) > 5000 else full_output)

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
            st.info(f"Training Gaussian Avatar model for {iterations:,} iterations...")
            
            # Progress tracking
            progress_bar = st.progress(0, text="Starting training...")
            iter_status = st.empty()
            
            # Create placeholder for streaming output
            with st.expander("View training output", expanded=False):
                output_placeholder = st.empty()
            
            # Get proper environment with CUDA and venv paths
            visual_env = get_visual_engine_env()
            
            # Run with streaming output for better feedback
            process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                env=visual_env,
                bufsize=1,
            )
            
            output_lines = []
            import re
            iter_pattern = re.compile(r'iteration\s+(\d+)')
            
            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                
                # Parse iteration progress
                match = iter_pattern.search(line.lower())
                if match:
                    current_iter = int(match.group(1))
                    progress_pct = min(int((current_iter / iterations) * 100), 99)
                    progress_bar.progress(progress_pct, text=f"Training: {current_iter:,} / {iterations:,} iterations")
                    iter_status.markdown(f"**Progress:** {current_iter:,} / {iterations:,} iterations ({progress_pct}%)")
                
                # Keep last 30 lines for display
                display_text = ''.join(output_lines[-30:])
                output_placeholder.code(display_text, language="")
            
            process.wait()
            
            if process.returncode == 0:
                progress_bar.progress(100, text="Training complete!")
                st.success("Training complete!")
                full_output = ''.join(output_lines)
                with st.expander("Final output"):
                    st.code(full_output[-2000:] if len(full_output) > 2000 else full_output)
            else:
                st.error("Training failed.")
                full_output = ''.join(output_lines)
                st.code(full_output[-2000:] if len(full_output) > 2000 else full_output)

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
            
            # Get proper environment with CUDA and venv paths
            visual_env = get_visual_engine_env()
            
            with st.spinner("Rendering prediction video …"):
                result = subprocess.run(
                    render_cmd,
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                    env=visual_env,
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
