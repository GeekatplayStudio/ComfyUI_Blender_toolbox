# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - Test Suite for Image Robustness & Automation
"""
Tests to verify that all image inputs across the system are self-adjusting,
self-analyzing, robust to any tensor shape, channel count, alpha channel,
resolution, or format, and that builder nodes self-analyze reference images.
"""

import base64
import io
import os
import sys
import unittest
import numpy as np
import torch
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conftest  # noqa: F401,E402

from nodes.ai_builder_nodes import (
    _normalize_image_tensor,
    _tensor_to_b64_list,
    _load_image_batch,
    GapAIReferenceAnalyzer,
    GapAIScenePlanner,
    GapAISceneBuilder,
    GapAIStepBuilder,
)
from nodes.ollama_nodes import tensor_to_base64
from nodes.geekatplay_toolbox import GapSmartResizer
from nodes.pbr_nodes import SimplePBRGenerator, SimpleHeightmapNormalizer, ColorToHeightmap


class TestImageRobustness(unittest.TestCase):

    def test_normalize_image_tensor_formats(self):
        # 1. None
        self.assertIsNone(_normalize_image_tensor(None))

        # 2. 2D mask / grayscale
        t2d = torch.rand(128, 128)
        norm = _normalize_image_tensor(t2d)
        self.assertEqual(norm.shape, (1, 128, 128, 3))
        self.assertTrue(0.0 <= norm.min() <= norm.max() <= 1.0)

        # 3. 3D HWC
        t3d_hwc = torch.rand(128, 128, 3)
        norm = _normalize_image_tensor(t3d_hwc)
        self.assertEqual(norm.shape, (1, 128, 128, 3))

        # 4. 3D CHW
        t3d_chw = torch.rand(3, 128, 128)
        norm = _normalize_image_tensor(t3d_chw)
        self.assertEqual(norm.shape, (1, 128, 128, 3))

        # 5. 3D BHW (batch of masks)
        t3d_bhw = torch.rand(2, 128, 128)
        norm = _normalize_image_tensor(t3d_bhw)
        self.assertEqual(norm.shape, (2, 128, 128, 3))

        # 6. 4D BCHW
        t4d_bchw = torch.rand(2, 3, 128, 128)
        norm = _normalize_image_tensor(t4d_bchw)
        self.assertEqual(norm.shape, (2, 128, 128, 3))

        # 7. 4D RGBA with alpha compositing
        t4d_rgba = torch.rand(2, 128, 128, 4)
        norm = _normalize_image_tensor(t4d_rgba)
        self.assertEqual(norm.shape, (2, 128, 128, 3))
        self.assertTrue(0.0 <= norm.min() <= norm.max() <= 1.0)

        # 8. uint8 [0, 255]
        t_uint8 = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
        norm = _normalize_image_tensor(t_uint8)
        self.assertEqual(norm.dtype, torch.float32)
        self.assertTrue(0.0 <= norm.min() <= norm.max() <= 1.0)

        # 9. float [0.0, 255.0]
        t_float255 = torch.rand(1, 64, 64, 3) * 255.0
        norm = _normalize_image_tensor(t_float255)
        self.assertTrue(0.0 <= norm.min() <= norm.max() <= 1.0)

        # 10. NaNs and Infs
        t_nan = torch.tensor([[[[float("nan"), float("inf"), float("-inf")]]]])
        norm = _normalize_image_tensor(t_nan)
        self.assertFalse(torch.isnan(norm).any())
        self.assertFalse(torch.isinf(norm).any())

        # 11. PIL Image
        pil_img = Image.new("RGBA", (100, 80), (255, 128, 0, 200))
        norm = _normalize_image_tensor(pil_img)
        self.assertEqual(norm.shape, (1, 80, 100, 3))

        # 12. NumPy array
        np_arr = np.random.rand(64, 64, 3).astype(np.float32)
        norm = _normalize_image_tensor(np_arr)
        self.assertEqual(norm.shape, (1, 64, 64, 3))

        # 13. List of differing sizes
        t_list = [torch.rand(1, 64, 64, 3), torch.rand(1, 128, 128, 3)]
        norm = _normalize_image_tensor(t_list)
        self.assertEqual(norm.shape, (2, 64, 64, 3))

    def test_tensor_to_b64_list(self):
        # Empty / None
        self.assertEqual(_tensor_to_b64_list(None), [])

        # Valid tensor to base64 PNG
        t = torch.rand(2, 64, 64, 3)
        b64s = _tensor_to_b64_list(t)
        self.assertEqual(len(b64s), 2)
        for b64 in b64s:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            self.assertEqual(img.format, "PNG")
            self.assertEqual(img.size, (64, 64))

        # Downscaling works
        large = torch.rand(1, 2048, 1500, 3)
        b64s = _tensor_to_b64_list(large, max_side=512)
        raw = base64.b64decode(b64s[0])
        img = Image.open(io.BytesIO(raw))
        self.assertTrue(max(img.size) <= 512)

    def test_load_image_batch_auto_resizes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "view1.png")
            p2 = os.path.join(tmp, "view2.png")
            Image.new("RGB", (640, 480), (255, 0, 0)).save(p1)
            Image.new("RGB", (800, 600), (0, 255, 0)).save(p2)  # different size!

            batch = _load_image_batch([p1, p2])
            # Must contain both views, auto-resized to match view1's (480, 640)
            self.assertEqual(batch.shape, (2, 480, 640, 3))

    def test_ollama_nodes_tensor_to_base64_robustness(self):
        # Test 2D mask, float255, RGBA with ollama's helper
        t2d = torch.rand(64, 64)
        b64_2d = tensor_to_base64(t2d)
        self.assertTrue(len(b64_2d) > 0)

        t_rgba = torch.rand(1, 64, 64, 4)
        b64_rgba = tensor_to_base64(t_rgba)
        self.assertTrue(len(b64_rgba) > 0)

    def test_aspect_padding_robustness(self):
        iap = GapSmartResizer()
        t2d = torch.rand(64, 64)
        out, w, h = iap.resize(t2d, "Custom", "1:1", "Scale (Lanczos)", 128, 128)
        self.assertEqual(out.ndim, 4)
        self.assertEqual(out.shape[-1], 3)
        self.assertEqual((w, h), (128, 128))

    def test_pbr_nodes_robustness(self):
        pbr = SimplePBRGenerator()
        t2d = torch.rand(64, 64)
        rough, metal, norm = pbr.generate(t2d, 0.5, 1.0)
        self.assertEqual(rough.shape, (1, 64, 64, 3))
        self.assertEqual(metal.shape, (1, 64, 64, 3))
        self.assertEqual(norm.shape, (1, 64, 64, 3))

        normalizer = SimpleHeightmapNormalizer()
        res = normalizer.normalize(t2d)
        self.assertEqual(res[0].shape, (1, 64, 64, 3))

        conv = ColorToHeightmap()
        hmap = conv.convert(t2d, False, True, 1.0)
        self.assertEqual(hmap[0].shape, (1, 64, 64, 3))

    def test_combined_preview_grid_and_headless_preview(self):
        import tempfile
        from nodes.ai_builder_nodes import (
            _load_image_batch,
            GapAIHeadlessPreview,
            _create_combined_preview_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "view_01_three_quarter.png")
            p2 = os.path.join(tmp, "view_02_front.png")
            p3 = os.path.join(tmp, "view_03_right.png")
            p4 = os.path.join(tmp, "view_04_top.png")
            for p in (p1, p2, p3, p4):
                Image.new("RGB", (320, 240), (60, 80, 100)).save(p)

            # 1. _load_image_batch with combine=False returns standard batch
            batch = _load_image_batch([p1, p2, p3, p4], combine=False)
            self.assertEqual(batch.shape, (4, 240, 320, 3))

            # 2. _load_image_batch with combine=True returns single composite image (B=1)
            combined = _load_image_batch([p1, p2, p3, p4], combine=True)
            self.assertEqual(combined.shape[0], 1)
            # 2x2 grid with 2px dividers: w = 2*320 + 2 = 642, h = 2*240 + 2 = 482
            self.assertEqual(combined.shape[1], 482)
            self.assertEqual(combined.shape[2], 642)

            # 3. GapAIHeadlessPreview node processes the 4-frame batch into single UI preview
            prev_node = GapAIHeadlessPreview()
            out = prev_node.preview(layout="auto", show_labels=True, images=batch)
            self.assertIn("ui", out)
            self.assertIn("images", out["ui"])
            # Crucial: exactly ONE preview image to avoid ComfyUI "loading 1/4... 2/4..." lag
            self.assertEqual(len(out["ui"]["images"]), 1)
            combined_out, info_out = out["result"]
            self.assertEqual(combined_out.shape[0], 1)
            self.assertIn("Combined 4 views", info_out)


if __name__ == "__main__":
    unittest.main()

