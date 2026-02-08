"""
run_segmentation.py — nnU-Net wrapper for CBCT / CT skull segmentation.

This module wraps nnU-Net v2 inference to produce a skull surface mesh
from DICOM input.  The user must supply trained weights in
01_Clinical_Engine/weights/.

Usage (called from app.py):
    from run_segmentation import segment_dicom
    mesh = segment_dicom(dicom_folder="01_Clinical_Engine/temp_data")
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv


WEIGHTS_DIR = Path(__file__).parent / "weights"
TEMP_DIR = Path(__file__).parent / "temp_data"


def segment_dicom(dicom_folder: str) -> pv.PolyData:
    """Run nnU-Net inference on a DICOM folder and return a surface mesh.

    Parameters
    ----------
    dicom_folder : str
        Path to folder containing DICOM files.

    Returns
    -------
    pv.PolyData
        Triangulated skull surface mesh.
    """
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise ImportError(
            "nnunetv2 is not installed. "
            "Install it in the omfs_clinic environment."
        ) from exc

    # --- Verify weights exist ---
    if not WEIGHTS_DIR.exists() or not any(WEIGHTS_DIR.iterdir()):
        raise FileNotFoundError(
            f"No model weights found in {WEIGHTS_DIR}. "
            "Please place your trained nnU-Net weights there."
        )

    # --- Run nnU-Net prediction ---
    output_dir = tempfile.mkdtemp(prefix="nnunet_pred_")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device_type="cuda",
    )
    predictor.initialize_from_trained_model_folder(
        str(WEIGHTS_DIR),
        use_folds="all",
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        list_of_lists_of_strings=[[str(p) for p in Path(dicom_folder).glob("*")]],
        output_filenames_truncated=[os.path.join(output_dir, "skull_seg")],
        save_probabilities=False,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    # --- Convert segmentation mask to surface mesh ---
    seg_path = os.path.join(output_dir, "skull_seg.nii.gz")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(
            f"Segmentation output not found at {seg_path}."
        )

    import nibabel as nib
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata().astype(np.uint8)

    grid = pv.ImageData(dimensions=seg_data.shape)
    grid.point_data["labels"] = seg_data.flatten(order="F")
    surface = grid.contour(isosurfaces=[0.5], scalars="labels")

    return surface
