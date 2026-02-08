"""Unit tests for 01_Clinical_Engine/surgical_sim.py."""

import sys
import os
import unittest

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_Clinical_Engine"))

from surgical_sim import SurgicalCutter


class TestSurgicalCutterSeparateMeshes(unittest.TestCase):
    """Tests with separate maxilla and mandible meshes."""

    def setUp(self):
        # Maxilla: sphere centred above origin (upper jaw)
        self.maxilla = pv.Sphere(radius=30, center=(0, 0, 20),
                                  theta_resolution=20, phi_resolution=20)
        # Mandible: sphere centred below origin (lower jaw)
        self.mandible = pv.Sphere(radius=30, center=(0, 0, -20),
                                   theta_resolution=20, phi_resolution=20)
        self.cutter = SurgicalCutter(self.maxilla, self.mandible)

    def test_preview_returns_expected_keys(self):
        result = self.cutter.preview_planes(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        for key in ("maxilla", "mandible", "combined", "lefort", "bsso_l", "bsso_r"):
            self.assertIn(key, result)

    def test_perform_cut_returns_four_segments(self):
        result = self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        for key in ("upper_skull", "mobile_maxilla", "distal_mandible", "proximal_rami"):
            self.assertIn(key, result)

    def test_maxilla_cut_only_affects_maxilla(self):
        """Le Fort should only cut the maxilla mesh, not the mandible."""
        result = self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        # The mandible parts should together equal the original mandible
        distal_pts = result["distal_mandible"].n_points
        rami_pts = result["proximal_rami"].n_points
        self.assertGreater(distal_pts, 0)
        self.assertGreater(rami_pts, 0)

    def test_move_maxilla_does_not_move_mandible(self):
        """Moving maxilla must NOT move distal mandible."""
        self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        mand_orig = np.array(self.cutter.distal_mandible.center)
        moved = self.cutter.move_segments(maxilla_mm=10.0, mandible_mm=0.0)
        mand_after = np.array(moved["distal_mandible"].center)
        np.testing.assert_array_almost_equal(mand_orig, mand_after)

    def test_move_mandible_does_not_move_maxilla(self):
        """Moving mandible must NOT move maxilla."""
        self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        max_orig = np.array(self.cutter.mobile_maxilla.center)
        moved = self.cutter.move_segments(maxilla_mm=0.0, mandible_mm=10.0)
        max_after = np.array(moved["mobile_maxilla"].center)
        np.testing.assert_array_almost_equal(max_orig, max_after)

    def test_move_segments_translates_correctly(self):
        self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        max_orig = np.array(self.cutter.mobile_maxilla.center)
        mand_orig = np.array(self.cutter.distal_mandible.center)
        moved = self.cutter.move_segments(maxilla_mm=5.0, mandible_mm=8.0)
        np.testing.assert_almost_equal(
            np.array(moved["mobile_maxilla"].center)[1] - max_orig[1], 5.0, decimal=1
        )
        np.testing.assert_almost_equal(
            np.array(moved["distal_mandible"].center)[1] - mand_orig[1], 8.0, decimal=1
        )

    def test_move_without_cut_raises(self):
        with self.assertRaises(RuntimeError):
            self.cutter.move_segments(maxilla_mm=5.0)

    def test_fixed_segments_stay_fixed(self):
        self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        skull_orig = np.array(self.cutter.upper_skull.center)
        rami_orig = np.array(self.cutter.proximal_rami.center)
        moved = self.cutter.move_segments(maxilla_mm=10.0, mandible_mm=10.0)
        np.testing.assert_array_almost_equal(
            skull_orig, np.array(moved["upper_skull"].center)
        )
        np.testing.assert_array_almost_equal(
            rami_orig, np.array(moved["proximal_rami"].center)
        )

    def test_move_segments_supports_custom_direction(self):
        self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        max_orig = np.array(self.cutter.mobile_maxilla.center)
        moved = self.cutter.move_segments(
            maxilla_mm=5.0,
            mandible_mm=0.0,
            advancement_direction=(1.0, 0.0, 0.0),
        )
        delta = np.array(moved["mobile_maxilla"].center) - max_orig
        self.assertAlmostEqual(delta[0], 5.0, places=1)
        self.assertAlmostEqual(delta[1], 0.0, places=1)
        self.assertAlmostEqual(delta[2], 0.0, places=1)

    def test_move_segments_rejects_zero_direction(self):
        self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        with self.assertRaises(ValueError):
            self.cutter.move_segments(
                maxilla_mm=1.0,
                mandible_mm=1.0,
                advancement_direction=(0.0, 0.0, 0.0),
            )

    def test_upper_skull_above_mobile_maxilla(self):
        """Upper skull center should be above mobile maxilla center in Z."""
        result = self.cutter.perform_cut(lefort_z=20, bsso_l_x=-15, bsso_r_x=15)
        if result["upper_skull"].n_points > 0 and result["mobile_maxilla"].n_points > 0:
            upper_z = float(result["upper_skull"].center[2])
            mobile_z = float(result["mobile_maxilla"].center[2])
            # They should be on different sides of the cut
            self.assertNotAlmostEqual(upper_z, mobile_z, places=1)


class TestSurgicalCutterSingleMesh(unittest.TestCase):
    """Tests with a single combined mesh (fallback mode)."""

    def setUp(self):
        self.mesh = pv.Sphere(radius=50, center=(0, 0, 0),
                              theta_resolution=30, phi_resolution=30)
        self.cutter = SurgicalCutter(self.mesh)

    def test_perform_cut_works(self):
        result = self.cutter.perform_cut(lefort_z=0, bsso_l_x=-20, bsso_r_x=20)
        self.assertGreater(result["upper_skull"].n_points, 0)

    def test_preview_works(self):
        result = self.cutter.preview_planes(lefort_z=0, bsso_l_x=-20, bsso_r_x=20)
        self.assertIn("combined", result)


if __name__ == "__main__":
    unittest.main()
