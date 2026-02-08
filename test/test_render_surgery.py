"""Unit tests for 02_Visual_Engine/render_surgery.py (offset computation)."""

import sys
import os
import unittest

# Ensure the Visual Engine module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_Visual_Engine"))

from render_surgery import compute_offset, apply_surgical_offsets, SCALE_FACTOR
import numpy as np


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


class TestApplySurgicalOffsets(unittest.TestCase):
    """Tests for apply_surgical_offsets."""

    def _make_params(self):
        return {
            "jaw_pose": np.array([0.0, 0.0, 0.0]),
            "translation": np.array([0.0, 0.0, 0.0]),
            "expression": np.zeros(10),
            "shape": np.zeros(10),
        }

    def test_lefort_offset_modifies_translation_y(self):
        params = self._make_params()
        modified = apply_surgical_offsets(params, lefort_offset=0.005, bsso_offset=0.0)
        self.assertAlmostEqual(modified["translation"][1], 0.005)
        # Original should be unchanged
        self.assertAlmostEqual(params["translation"][1], 0.0)

    def test_bsso_offset_modifies_jaw_pose_x(self):
        params = self._make_params()
        modified = apply_surgical_offsets(params, lefort_offset=0.0, bsso_offset=0.003)
        self.assertAlmostEqual(modified["jaw_pose"][0], 0.003)
        self.assertAlmostEqual(params["jaw_pose"][0], 0.0)

    def test_combined_offsets(self):
        params = self._make_params()
        modified = apply_surgical_offsets(params, lefort_offset=0.01, bsso_offset=0.02)
        self.assertAlmostEqual(modified["translation"][1], 0.01)
        self.assertAlmostEqual(modified["jaw_pose"][0], 0.02)

    def test_does_not_mutate_input(self):
        params = self._make_params()
        original_jaw = params["jaw_pose"].copy()
        original_trans = params["translation"].copy()
        apply_surgical_offsets(params, lefort_offset=0.1, bsso_offset=0.2)
        np.testing.assert_array_equal(params["jaw_pose"], original_jaw)
        np.testing.assert_array_equal(params["translation"], original_trans)


if __name__ == "__main__":
    unittest.main()
