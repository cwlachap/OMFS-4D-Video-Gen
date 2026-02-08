"""
surgical_sim.py — Interactive Bone Cutting & Movement for OMFS Planning.

Coordinate convention (medical / NIfTI):
    X = Left–Right
    Y = Anterior–Posterior (front–back)
    Z = Superior–Inferior (up–down)

3 Planes:
    1. Le Fort I  — horizontal (normal Z) — cuts only the MAXILLA mesh
    2. BSSO Left  — sagittal   (normal X) — cuts only the MANDIBLE mesh
    3. BSSO Right — sagittal   (normal X) — cuts only the MANDIBLE mesh

4 Segments after cut:
    • Upper skull    (fixed)  — maxilla above Le Fort
    • Mobile maxilla (mobile) — maxilla below Le Fort
    • Distal mandible (mobile) — mandible between BSSO planes
    • Proximal rami  (fixed)  — mandible outside BSSO planes
"""

import numpy as np
import pyvista as pv


def _angle_to_normal(base_normal: tuple, pitch_deg: float, yaw_deg: float) -> tuple:
    """Rotate a base normal by pitch (around X) and yaw (around Z) in degrees."""
    n = np.array(base_normal, dtype=float)

    pitch = np.radians(pitch_deg)
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])

    yaw = np.radians(yaw_deg)
    rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    rotated = rz @ rx @ n
    length = np.linalg.norm(rotated)
    if length < 1e-12:
        return tuple(base_normal)
    return tuple(rotated / length)


def _normalise_direction(direction: tuple[float, float, float]) -> np.ndarray:
    """Return a unit direction vector; fail fast for near-zero vectors."""
    vec = np.array(direction, dtype=float)
    length = np.linalg.norm(vec)
    if length < 1e-12:
        raise ValueError("advancement_direction must be a non-zero vector.")
    return vec / length


class SurgicalCutter:
    """Cuts separate maxilla and mandible meshes with 3 planes.

    If only a single combined mesh is provided (no separate bones),
    it will operate on that single mesh with a best-effort approach.
    """

    def __init__(
        self,
        maxilla_mesh: pv.PolyData,
        mandible_mesh: pv.PolyData | None = None,
    ):
        """
        Parameters
        ----------
        maxilla_mesh  : Upper jawbone mesh (or combined skull if no mandible).
        mandible_mesh : Lower jawbone mesh (None if using a single mesh).
        """
        self.maxilla = maxilla_mesh
        self.mandible = mandible_mesh

        # If no separate mandible, we work with maxilla as the full skull
        self.has_separate = mandible_mesh is not None and mandible_mesh.n_points > 0

        # Results after cutting
        self.upper_skull: pv.PolyData | None = None
        self.mobile_maxilla: pv.PolyData | None = None
        self.distal_mandible: pv.PolyData | None = None
        self.proximal_rami: pv.PolyData | None = None

    def get_combined_mesh(self) -> pv.PolyData:
        """Return both jaws as one mesh for preview."""
        if self.has_separate:
            return self.maxilla.merge(self.mandible)
        return self.maxilla

    # ── Preview ──────────────────────────────────────────────
    def preview_planes(
        self,
        lefort_z: float,
        bsso_l_x: float,
        bsso_r_x: float,
        lefort_pitch: float = 0.0,
        lefort_yaw: float = 0.0,
        bsso_l_pitch: float = 0.0,
        bsso_l_yaw: float = 0.0,
        bsso_r_pitch: float = 0.0,
        bsso_r_yaw: float = 0.0,
    ) -> dict:
        """Return meshes and 3 visualisation planes."""
        combined = self.get_combined_mesh()
        bounds = combined.bounds
        sizes = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
        plane_size = max(sizes) * 1.2
        center = combined.center

        lefort_n = _angle_to_normal((0, 0, 1), lefort_pitch, lefort_yaw)
        bsso_l_n = _angle_to_normal((1, 0, 0), bsso_l_pitch, bsso_l_yaw)
        bsso_r_n = _angle_to_normal((1, 0, 0), bsso_r_pitch, bsso_r_yaw)

        lefort_plane = pv.Plane(
            center=(center[0], center[1], lefort_z),
            direction=lefort_n, i_size=plane_size, j_size=plane_size,
        )
        bsso_l_plane = pv.Plane(
            center=(bsso_l_x, center[1], center[2]),
            direction=bsso_l_n, i_size=plane_size, j_size=plane_size,
        )
        bsso_r_plane = pv.Plane(
            center=(bsso_r_x, center[1], center[2]),
            direction=bsso_r_n, i_size=plane_size, j_size=plane_size,
        )

        return {
            "maxilla": self.maxilla,
            "mandible": self.mandible,
            "combined": combined,
            "lefort": lefort_plane,
            "bsso_l": bsso_l_plane,
            "bsso_r": bsso_r_plane,
        }

    # ── Cut ──────────────────────────────────────────────────
    def perform_cut(
        self,
        lefort_z: float,
        bsso_l_x: float,
        bsso_r_x: float,
        lefort_pitch: float = 0.0,
        lefort_yaw: float = 0.0,
        bsso_l_pitch: float = 0.0,
        bsso_l_yaw: float = 0.0,
        bsso_r_pitch: float = 0.0,
        bsso_r_yaw: float = 0.0,
        lefort_flip: bool = False,
    ) -> dict:
        """Cut the maxilla and mandible into 4 segments.

        Le Fort I cuts the MAXILLA only:
            above → upper skull (fixed)
            below → mobile maxilla

        BSSO cuts the MANDIBLE only:
            between L and R → distal segment (mobile)
            outside → proximal rami (fixed)
        """
        combined = self.get_combined_mesh()
        center = combined.center

        lefort_n = _angle_to_normal((0, 0, 1), lefort_pitch, lefort_yaw)
        bsso_l_n = _angle_to_normal((1, 0, 0), bsso_l_pitch, bsso_l_yaw)
        bsso_r_n = _angle_to_normal((1, 0, 0), bsso_r_pitch, bsso_r_yaw)

        lefort_origin = (center[0], center[1], lefort_z)
        bsso_l_origin = (bsso_l_x, center[1], center[2])
        bsso_r_origin = (bsso_r_x, center[1], center[2])

        if self.has_separate:
            # ── Separate meshes: cut each with its own plane(s) ──

            # Le Fort I → cuts MAXILLA only
            # The tooth-bearing segment (mobile) is below the cut.
            # NIfTI Z can point either way — invert=False keeps the
            # side in the normal direction; swap if your data has Z-down.
            mobile_invert = bool(lefort_flip)
            mobile_maxilla = self.maxilla.clip(
                normal=lefort_n, origin=lefort_origin, invert=mobile_invert
            )
            upper_skull = self.maxilla.clip(
                normal=lefort_n, origin=lefort_origin, invert=not mobile_invert
            )

            # BSSO → cuts MANDIBLE only
            mid = self.mandible.clip(
                normal=bsso_l_n, origin=bsso_l_origin, invert=False
            )
            distal_mandible = mid.clip(
                normal=bsso_r_n, origin=bsso_r_origin, invert=True
            )

            left_ramus = self.mandible.clip(
                normal=bsso_l_n, origin=bsso_l_origin, invert=True
            )
            right_ramus = self.mandible.clip(
                normal=bsso_r_n, origin=bsso_r_origin, invert=False
            )

            # Proximal rami = left + right
            proximal_rami = pv.PolyData()
            if left_ramus.n_points and right_ramus.n_points:
                proximal_rami = left_ramus.merge(right_ramus)
            elif left_ramus.n_points:
                proximal_rami = left_ramus
            elif right_ramus.n_points:
                proximal_rami = right_ramus

        else:
            # ── Single mesh fallback ──
            # Le Fort
            mobile_invert = bool(lefort_flip)
            upper_skull = self.maxilla.clip(
                normal=lefort_n, origin=lefort_origin, invert=not mobile_invert
            )
            below = self.maxilla.clip(
                normal=lefort_n, origin=lefort_origin, invert=mobile_invert
            )
            # Treat everything below Le Fort as mobile maxilla
            mobile_maxilla = below

            # BSSO on the full mesh
            mid = self.maxilla.clip(
                normal=bsso_l_n, origin=bsso_l_origin, invert=False
            )
            distal_mandible = mid.clip(
                normal=bsso_r_n, origin=bsso_r_origin, invert=True
            )
            left_ramus = self.maxilla.clip(
                normal=bsso_l_n, origin=bsso_l_origin, invert=True
            )
            right_ramus = self.maxilla.clip(
                normal=bsso_r_n, origin=bsso_r_origin, invert=False
            )
            proximal_rami = pv.PolyData()
            if left_ramus.n_points and right_ramus.n_points:
                proximal_rami = left_ramus.merge(right_ramus)
            elif left_ramus.n_points:
                proximal_rami = left_ramus
            elif right_ramus.n_points:
                proximal_rami = right_ramus

        self.upper_skull = upper_skull
        self.mobile_maxilla = mobile_maxilla
        self.distal_mandible = distal_mandible
        self.proximal_rami = proximal_rami

        return {
            "upper_skull": upper_skull,
            "mobile_maxilla": mobile_maxilla,
            "distal_mandible": distal_mandible,
            "proximal_rami": proximal_rami,
        }

    # ── Move ─────────────────────────────────────────────────
    def move_segments(
        self,
        maxilla_mm: float = 0.0,
        mandible_mm: float = 0.0,
        advancement_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> dict:
        """Translate mobile segments along a configured advancement direction.

        maxilla_mm  moves ONLY the mobile maxilla.
        mandible_mm moves ONLY the distal mandible.
        Upper skull and proximal rami stay fixed.
        """
        if self.mobile_maxilla is None or self.distal_mandible is None:
            raise RuntimeError("Call perform_cut() before move_segments().")

        adv_dir = _normalise_direction(advancement_direction)
        moved_maxilla = self.mobile_maxilla.copy()
        moved_mandible = self.distal_mandible.copy()

        moved_maxilla.translate(tuple(adv_dir * maxilla_mm), inplace=True)
        moved_mandible.translate(tuple(adv_dir * mandible_mm), inplace=True)

        return {
            "upper_skull": self.upper_skull,
            "mobile_maxilla": moved_maxilla,
            "distal_mandible": moved_mandible,
            "proximal_rami": self.proximal_rami,
        }
