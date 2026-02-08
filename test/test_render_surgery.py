"""Unit tests for 02_Visual_Engine/render_surgery.py (offset computation)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_Visual_Engine"))

from render_surgery import compute_offset, modify_flame_params, SCALE_FACTOR
import numpy as np
import tempfile


class TestComputeOffset(unittest.TestCase):
    """Tests for the compute_offset pure function."""

    def test_zero_mm_returns_zero(self):
        self.assertEqual(compute_offset(0.0, 1.0), 0.0)

    def test_positive_mm(self):
        result = compute_offset(5.0, 1.0)
        expected = 5.0 * 1.0 * SCALE_FACTOR
        self.assertAlmostEqual(result, expected)

    def test_negative_mm(self):
        result = compute_offset(-3.0, 1.0)
        expected = -3.0 * 1.0 * SCALE_FACTOR
        self.assertAlmostEqual(result, expected)

    def test_sensitivity_scaling(self):
        result = compute_offset(5.0, 2.5)
        expected = 5.0 * 2.5 * SCALE_FACTOR
        self.assertAlmostEqual(result, expected)

    def test_zero_sensitivity(self):
        self.assertEqual(compute_offset(10.0, 0.0), 0.0)


class TestModifyFlameParams(unittest.TestCase):
    """Tests for modify_flame_params."""

    def setUp(self):
        """Create a temp FLAME param file."""
        self.temp_dir = tempfile.mkdtemp()
        self.source_path = os.path.join(self.temp_dir, "source.npz")
        self.output_path = os.path.join(self.temp_dir, "modified.npz")
        np.savez(self.source_path,
                 jaw_pose=np.zeros((10, 3), dtype=np.float32),
                 translation=np.zeros((10, 3), dtype=np.float32),
                 expr=np.zeros((10, 100), dtype=np.float32),
                 shape=np.zeros(300, dtype=np.float32))

    def test_lefort_modifies_translation_y(self):
        modify_flame_params(self.source_path, self.output_path, 0.005, 0.0)
        data = np.load(self.output_path)
        self.assertAlmostEqual(float(data["translation"][0, 1]), 0.005, places=5)

    def test_bsso_modifies_jaw_pose_x(self):
        modify_flame_params(self.source_path, self.output_path, 0.0, 0.003)
        data = np.load(self.output_path)
        self.assertAlmostEqual(float(data["jaw_pose"][0, 0]), 0.003, places=5)

    def test_does_not_mutate_source(self):
        modify_flame_params(self.source_path, self.output_path, 0.01, 0.02)
        source = np.load(self.source_path)
        self.assertAlmostEqual(float(source["translation"][0, 1]), 0.0)
        self.assertAlmostEqual(float(source["jaw_pose"][0, 0]), 0.0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
