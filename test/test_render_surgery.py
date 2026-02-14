"""Unit tests for 02_Visual_Engine/render_surgery.py (offset computation)."""

import sys
import os
import unittest
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_Visual_Engine"))

from render_surgery import (
    compute_offset,
    modify_flame_params,
    choose_rig_mode,
    export_deterministic_frames,
    SCALE_FACTOR,
)
import numpy as np
import tempfile
from PIL import Image


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

    def test_hybrid_deformation_map_axes_and_scale(self):
        deform = {
            "translation_axis": 2,
            "jaw_axis": 1,
            "lefort_scale": 2.0,
            "bsso_scale": 0.5,
        }
        modify_flame_params(self.source_path, self.output_path, 0.01, 0.02, deformation_map=deform)
        data = np.load(self.output_path)
        self.assertAlmostEqual(float(data["translation"][0, 2]), 0.02, places=5)
        self.assertAlmostEqual(float(data["jaw_pose"][0, 1]), 0.01, places=5)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestRigModeFallback(unittest.TestCase):
    def test_hybrid_falls_back_without_asset(self):
        mode, reason = choose_rig_mode("hybrid_full_head", "")
        self.assertEqual(mode, "flame_only")
        self.assertIn("missing", reason)

    def test_hybrid_kept_when_asset_exists(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "asset.npz"
            np.savez(p, version=np.array([1]))
            mode, _ = choose_rig_mode("hybrid_full_head", str(p))
            self.assertEqual(mode, "hybrid_full_head")


class TestDeterministicFrameExport(unittest.TestCase):
    def test_export_with_explicit_indices(self):
        with tempfile.TemporaryDirectory() as d:
            frames_dir = Path(d) / "renders"
            out_dir = Path(d) / "out"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(6):
                img = Image.fromarray(np.full((8, 8, 3), i * 20, dtype=np.uint8))
                img.save(frames_dir / f"{i:05d}.png")
            idx_file = Path(d) / "idx.json"
            idx_file.write_text(json.dumps({"indices": [0, 3, 5]}), encoding="utf-8")
            export_deterministic_frames(str(frames_dir), str(out_dir), str(idx_file))
            manifest = json.loads((out_dir / "deterministic_indices_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_indices"], [0, 3, 5])
            self.assertTrue((out_dir / "idx_00000.png").exists())
            self.assertTrue((out_dir / "idx_00003.png").exists())
            self.assertTrue((out_dir / "idx_00005.png").exists())


if __name__ == "__main__":
    unittest.main()
