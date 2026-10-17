#!/usr/bin/env python3
"""对齐 ROI 的 NCC/ZNCC/直方图/SSIM，以及只平移的 locate。不是金字塔 PATTERN 匹配器。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import synth
import viz_util as viz


class TestSimilarityViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("similarity")

    def test_compare_identical_and_noise(self):
        patch = synth.make_asymmetric_patch()
        same = cvter.compare_similarity(patch, patch)
        self.assertTrue(same.found)
        self.assertGreater(same.zncc, 0.99)
        self.assertGreater(same.ncc, 0.99)
        self.assertGreater(same.ssim, 0.98)
        self.assertGreater(same.histCorr, 0.99)
        self.assertLess(same.histBhattacharyya, 0.05)

        noisy = patch.copy()
        rng = np.random.RandomState(0)
        noise = rng.randint(-25, 26, size=patch.shape, dtype=np.int16)
        noisy = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        diff = cvter.compare_similarity(patch, noisy)
        self.assertTrue(diff.found)
        self.assertLess(diff.ssim, same.ssim)
        self.assertGreater(diff.zncc, 0.5)

        view = viz.to_bgr(patch)
        viz.put_text(view, f"ZNCC={same.zncc:.3f} SSIM={same.ssim:.3f}", (4, 16), viz.GREEN, 0.4)
        noisy_view = viz.to_bgr(noisy)
        viz.put_text(noisy_view, f"ZNCC={diff.zncc:.3f} SSIM={diff.ssim:.3f}", (4, 16), viz.ORANGE, 0.4)
        viz.assert_written(self, viz.save_panels("similarity", "01_compare", [
            ("identical", view),
            ("noisy", noisy_view),
        ]))

    def test_locate_translation(self):
        patch = synth.make_asymmetric_patch()
        canvas = np.full((220, 280), 40, np.uint8)
        x0, y0 = 70, 48
        synth.paste(canvas, patch, x0, y0)
        hit = cvter.locate_zncc(canvas, patch, 0.7)
        self.assertTrue(hit.found)
        self.assertGreater(hit.score, 0.9)
        self.assertLess(abs(hit.topLeft.x - x0), 1.5)
        self.assertLess(abs(hit.topLeft.y - y0), 1.5)
        ncc = cvter.locate_ncc(canvas, patch, 0.7)
        self.assertTrue(ncc.found)
        view = viz.to_bgr(canvas)
        cv2.rectangle(view, (int(hit.topLeft.x), int(hit.topLeft.y)),
                      (int(hit.topLeft.x + hit.width), int(hit.topLeft.y + hit.height)), viz.GREEN, 2)
        viz.draw_cross(view, hit.center.x, hit.center.y)
        viz.put_text(view, f"ZNCC={hit.score:.3f} NCC={ncc.score:.3f}", (8, 18), viz.GREEN)
        viz.assert_written(self, viz.save_panels("similarity", "02_locate", [
            ("template", patch),
            ("PASS translate", view),
        ]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
