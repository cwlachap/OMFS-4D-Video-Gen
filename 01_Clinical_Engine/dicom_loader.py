"""
dicom_loader.py — Load DICOM / NIfTI scans and extract bone surface meshes.

Supports:
  • DICOM folders (.dcm) — HU thresholding + marching cubes.
  • NIfTI files (.nii.gz) — direct loading (images or segmentation labels).
  • ToothFairy3 dataset — uses pre-segmented labels for maxilla/mandible.

No AI weights required.

Usage:
    from dicom_loader import dicom_to_bone_mesh, nifti_label_to_bone_mesh
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pyvista as pv
from skimage import measure


def load_dicom_volume(dicom_path: str) -> tuple[np.ndarray, tuple]:
    """Load a DICOM series from a folder and return the 3D volume in HU.

    Parameters
    ----------
    dicom_path : str
        Path to folder containing .dcm files, or a single .dcm file.

    Returns
    -------
    volume : np.ndarray
        3D array of Hounsfield Unit values (Z, Y, X).
    spacing : tuple
        Voxel spacing in mm (z_spacing, y_spacing, x_spacing).
    """
    path = Path(dicom_path)

    if path.is_file():
        # Single file — treat parent dir as the series folder
        path = path.parent

    # Collect all DICOM files
    dcm_files = []
    for f in sorted(path.iterdir()):
        if f.is_file() and f.suffix.lower() in (".dcm", ".ima", ""):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if hasattr(ds, "ImagePositionPatient"):
                    dcm_files.append(str(f))
            except Exception:
                continue

    if not dcm_files:
        raise FileNotFoundError(
            f"No valid DICOM files found in: {path}\n"
            "Ensure the folder contains .dcm files with image data."
        )

    # Read all slices
    slices = [pydicom.dcmread(f) for f in dcm_files]

    # Sort by ImagePositionPatient Z coordinate
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # Extract pixel spacing
    pixel_spacing = slices[0].PixelSpacing
    x_spacing = float(pixel_spacing[1])
    y_spacing = float(pixel_spacing[0])

    # Z spacing from slice positions
    if len(slices) > 1:
        z_spacing = abs(
            float(slices[1].ImagePositionPatient[2])
            - float(slices[0].ImagePositionPatient[2])
        )
    else:
        z_spacing = float(getattr(slices[0], "SliceThickness", 1.0))

    # Build 3D volume
    rows = slices[0].Rows
    cols = slices[0].Columns
    volume = np.zeros((len(slices), rows, cols), dtype=np.float32)

    for i, s in enumerate(slices):
        pixel_array = s.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        volume[i] = pixel_array * slope + intercept

    spacing = (z_spacing, y_spacing, x_spacing)
    return volume, spacing


def dicom_to_bone_mesh(
    dicom_path: str,
    hu_threshold: float = 300.0,
    smooth_iterations: int = 30,
    decimate_fraction: float = 0.5,
) -> pv.PolyData:
    """Convert a DICOM folder to a bone surface mesh.

    Pipeline:
        1. Load DICOM → 3D HU volume.
        2. Threshold at hu_threshold (default 300 HU for bone).
        3. Marching cubes → triangulated surface.
        4. Smooth and decimate for cleaner geometry.

    Parameters
    ----------
    dicom_path : str
        Path to DICOM folder.
    hu_threshold : float
        Hounsfield Unit threshold for bone (300 = cancellous, 700 = cortical).
    smooth_iterations : int
        Number of Laplacian smoothing iterations.
    decimate_fraction : float
        Target fraction of triangles to keep (0.5 = keep 50%).

    Returns
    -------
    pv.PolyData
        Triangulated bone surface mesh.
    """
    volume, spacing = load_dicom_volume(dicom_path)

    # Marching cubes on the thresholded volume
    verts, faces, normals, _ = measure.marching_cubes(
        volume,
        level=hu_threshold,
        spacing=spacing,
    )

    # Convert to PyVista PolyData
    # faces need to be in VTK format: [n_pts, v0, v1, v2, ...]
    n_faces = len(faces)
    vtk_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        faces.astype(np.int64),
    ]).ravel()

    mesh = pv.PolyData(verts, vtk_faces)

    # Clean up
    mesh = mesh.clean()

    # Smooth
    if smooth_iterations > 0:
        mesh = mesh.smooth(n_iter=smooth_iterations)

    # Decimate to reduce triangle count
    if 0.0 < decimate_fraction < 1.0:
        target_reduction = 1.0 - decimate_fraction
        mesh = mesh.decimate(target_reduction)

    # Centre the mesh at the origin for easier manipulation
    mesh.translate(-np.array(mesh.center), inplace=True)

    return mesh


# ══════════════════════════════════════════════════════════════
# NIfTI / ToothFairy3 Loaders
# ══════════════════════════════════════════════════════════════

# ToothFairy3 label mapping (from dataset.json)
TOOTHFAIRY_LABELS = {
    "Lower Jawbone": 1,
    "Upper Jawbone": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    "Left Maxillary Sinus": 5,
    "Right Maxillary Sinus": 6,
}

# Tooth label ranges from dataset.json (FDI numbering)
UPPER_TEETH_LABELS = [
    11, 12, 13, 14, 15, 16, 17, 18,   # upper right
    21, 22, 23, 24, 25, 26, 27, 28,   # upper left
]
LOWER_TEETH_LABELS = [
    31, 32, 33, 34, 35, 36, 37, 38,   # lower left
    41, 42, 43, 44, 45, 46, 47, 48,   # lower right
]
ALL_TEETH_LABELS = UPPER_TEETH_LABELS + LOWER_TEETH_LABELS


def nifti_to_volume(nifti_path: str) -> tuple[np.ndarray, tuple]:
    """Load a NIfTI file and return the 3D volume + voxel spacing.

    Parameters
    ----------
    nifti_path : str
        Path to a .nii or .nii.gz file.

    Returns
    -------
    volume  : np.ndarray (Z, Y, X) or (X, Y, Z) depending on orientation.
    spacing : tuple of voxel sizes in mm.
    """
    img = nib.load(nifti_path)
    volume = np.asarray(img.dataobj, dtype=np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return volume, spacing


def _volume_mask_to_mesh(
    mask: np.ndarray,
    spacing: tuple,
    smooth_iterations: int = 30,
    decimate_fraction: float = 0.5,
) -> pv.PolyData:
    """Convert a binary mask volume to a surface mesh via marching cubes."""
    if mask.sum() == 0:
        return pv.PolyData()

    verts, faces, normals, _ = measure.marching_cubes(
        mask, level=0.5, spacing=spacing,
    )
    n_faces = len(faces)
    vtk_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        faces.astype(np.int64),
    ]).ravel()

    mesh = pv.PolyData(verts, vtk_faces)
    mesh = mesh.clean()
    if smooth_iterations > 0:
        mesh = mesh.smooth(n_iter=smooth_iterations)
    if 0.0 < decimate_fraction < 1.0:
        mesh = mesh.decimate(1.0 - decimate_fraction)
    return mesh


def nifti_label_to_separate_meshes(
    label_path: str,
    include_upper_labels: list[int] | None = None,
    include_lower_labels: list[int] | None = None,
    smooth_iterations: int = 30,
    decimate_fraction: float = 0.5,
) -> dict:
    """Extract SEPARATE maxilla and mandible meshes from a ToothFairy3 label file.

    Returns
    -------
    dict with keys: maxilla_mesh, mandible_mesh, combined_mesh.
    All meshes are centred at the combined mesh's centre.
    """
    if include_upper_labels is None:
        include_upper_labels = [2] + UPPER_TEETH_LABELS
    if include_lower_labels is None:
        include_lower_labels = [1] + LOWER_TEETH_LABELS

    volume, spacing = nifti_to_volume(label_path)
    vol_int = volume.astype(int)

    upper_mask = np.isin(vol_int, include_upper_labels).astype(np.float32)
    lower_mask = np.isin(vol_int, include_lower_labels).astype(np.float32)

    maxilla_mesh = _volume_mask_to_mesh(upper_mask, spacing,
                                         smooth_iterations, decimate_fraction)
    mandible_mesh = _volume_mask_to_mesh(lower_mask, spacing,
                                          smooth_iterations, decimate_fraction)

    # Combine for preview and centre both at the same origin
    if maxilla_mesh.n_points and mandible_mesh.n_points:
        combined = maxilla_mesh.merge(mandible_mesh)
    elif maxilla_mesh.n_points:
        combined = maxilla_mesh
    else:
        combined = mandible_mesh

    origin = np.array(combined.center)
    maxilla_mesh.translate(-origin, inplace=True)
    mandible_mesh.translate(-origin, inplace=True)
    combined.translate(-origin, inplace=True)

    return {
        "maxilla_mesh": maxilla_mesh,
        "mandible_mesh": mandible_mesh,
        "combined_mesh": combined,
    }


def nifti_label_to_bone_mesh(
    label_path: str,
    include_labels: list[int] | None = None,
    smooth_iterations: int = 30,
    decimate_fraction: float = 0.5,
) -> pv.PolyData:
    """Extract a bone mesh from a NIfTI segmentation label file.

    This is ideal for the ToothFairy3 dataset where labels are
    already provided (1 = Lower Jaw, 2 = Upper Jaw, etc.).

    Parameters
    ----------
    label_path : str
        Path to label .nii.gz file.
    include_labels : list[int] or None
        Which label IDs to include in the mesh.
        Default: [1, 2] (Lower + Upper Jawbone).
    smooth_iterations : int
        Laplacian smoothing iterations.
    decimate_fraction : float
        Fraction of triangles to keep.

    Returns
    -------
    pv.PolyData
        Bone surface mesh.
    """
    if include_labels is None:
        include_labels = [1, 2]  # Lower + Upper Jawbone

    volume, spacing = nifti_to_volume(label_path)

    # Create a binary mask of the selected labels
    mask = np.isin(volume.astype(int), include_labels).astype(np.float32)

    if mask.sum() == 0:
        raise ValueError(
            f"No voxels found for labels {include_labels} in {label_path}."
        )

    # Marching cubes on the binary mask
    verts, faces, normals, _ = measure.marching_cubes(
        mask,
        level=0.5,
        spacing=spacing,
    )

    n_faces = len(faces)
    vtk_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        faces.astype(np.int64),
    ]).ravel()

    mesh = pv.PolyData(verts, vtk_faces)
    mesh = mesh.clean()

    if smooth_iterations > 0:
        mesh = mesh.smooth(n_iter=smooth_iterations)

    if 0.0 < decimate_fraction < 1.0:
        target_reduction = 1.0 - decimate_fraction
        mesh = mesh.decimate(target_reduction)

    mesh.translate(-np.array(mesh.center), inplace=True)

    return mesh


def nifti_image_to_bone_mesh(
    image_path: str,
    hu_threshold: float = 300.0,
    smooth_iterations: int = 30,
    decimate_fraction: float = 0.5,
) -> pv.PolyData:
    """Extract a bone mesh from a NIfTI CBCT image using HU thresholding.

    Parameters
    ----------
    image_path : str
        Path to image .nii.gz file.
    hu_threshold : float
        Hounsfield Unit threshold for bone.
    smooth_iterations : int
        Laplacian smoothing iterations.
    decimate_fraction : float
        Fraction of triangles to keep.

    Returns
    -------
    pv.PolyData
        Bone surface mesh.
    """
    volume, spacing = nifti_to_volume(image_path)

    verts, faces, normals, _ = measure.marching_cubes(
        volume,
        level=hu_threshold,
        spacing=spacing,
    )

    n_faces = len(faces)
    vtk_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        faces.astype(np.int64),
    ]).ravel()

    mesh = pv.PolyData(verts, vtk_faces)
    mesh = mesh.clean()

    if smooth_iterations > 0:
        mesh = mesh.smooth(n_iter=smooth_iterations)

    if 0.0 < decimate_fraction < 1.0:
        target_reduction = 1.0 - decimate_fraction
        mesh = mesh.decimate(target_reduction)

    mesh.translate(-np.array(mesh.center), inplace=True)

    return mesh
