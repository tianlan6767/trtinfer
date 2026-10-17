#!/usr/bin/env python3
"""预处理 / Blob 可视化。断言与 tests/test_blob.py 对齐，并写出叠加图。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cvter
import viz_util as viz


class TestBlobViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        viz.prepare("blob")

    def test_preprocess_and_blobs(self):
        img = np.zeros((180, 220), np.uint8)
        cv2.circle(img, (55, 60), 22, 255, -1)
        cv2.rectangle(img, (130, 30), (143, 99), 255, -1)
        img[150:152, 100:102] = 255

        roi = cvter.crop_roi(img, 40, 30, 40, 50)
        self.assertEqual(roi.shape, (50, 40))
        blur = cvter.gaussian_blur(img, 3, 0.6)
        th = cvter.ThresholdParam()
        th.type = cvter.ThresholdType.Fixed
        th.value = 80
        binary = cvter.threshold(blur, th)
        morph = cvter.MorphParam()
        morph.op = cvter.MorphOp.Open
        morph.ksize = 3
        opened = cvter.morphology(binary, morph)

        filt = cvter.BlobFilter()
        filt.minArea = 30
        blobs = cvter.find_blobs(opened, filt)
        self.assertEqual(len(blobs), 2)
        circ = max(blobs, key=lambda b: b.circularity)
        self.assertGreater(circ.circularity, 0.7)
        self.assertLess(abs(circ.center.x - 55) + abs(circ.center.y - 60), 4.0)

        overlay = viz.to_bgr(img)
        for b in blobs:
            viz.draw_blob(overlay, b, viz.GREEN if b.circularity > 0.7 else viz.ORANGE)
        path = viz.save_panels("blob", "01_preprocess_blobs", [
            ("input", img),
            ("blur", blur),
            ("threshold", binary),
            ("open + blobs", overlay),
        ])
        viz.assert_written(self, path)

        filt.minCircularity = 0.7
        only_round = cvter.find_blobs(opened, filt)
        self.assertEqual(len(only_round), 1)
        round_view = viz.to_bgr(img)
        viz.draw_blob(round_view, only_round[0])
        viz.assert_written(self, viz.save_panels("blob", "02_circularity", [("circ>0.7", round_view)]))

        bars = cvter.BlobFilter()
        bars.minArea = 30
        bars.minAspect = 3.0
        only = cvter.find_blobs(opened, bars)
        self.assertEqual(len(only), 1)
        self.assertGreater(only[0].aspect, 3.0)
        bar_view = viz.to_bgr(img)
        viz.draw_blob(bar_view, only[0], viz.ORANGE)
        viz.assert_written(self, viz.save_panels("blob", "03_aspect", [("aspect>3", bar_view)]))

        roi_blobs = [b for b in cvter.find_blobs(opened, cvter.BlobFilter(), 100, 0, 120, 160) if b.area >= 30]
        self.assertEqual(len(roi_blobs), 1)
        self.assertGreater(roi_blobs[0].center.x, 100)
        roi_view = viz.to_bgr(img)
        viz.draw_search_rect(roi_view, 100, 0, 220, 160, viz.YELLOW)
        viz.draw_blob(roi_view, roi_blobs[0])
        viz.assert_written(self, viz.save_panels("blob", "04_roi", [("roi x>100", roi_view)]))

    def test_contour_stats(self):
        square = np.zeros((80, 90), np.uint8)
        cv2.rectangle(square, (10, 15), (49, 54), 255, -1)
        stats = cvter.contour_stats(square)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats[0].area, 1400)
        self.assertLessEqual(stats[0].area, 1600)
        pixels = cvter.find_blobs(square)
        self.assertEqual(len(pixels), 1)
        self.assertEqual(pixels[0].area, 1600)
        contours = cvter.extract_contours(square)
        self.assertGreaterEqual(len(contours), 1)
        view = viz.to_bgr(square)
        viz.draw_poly(view, contours[0].points, viz.GREEN, 1)
        viz.draw_blob(view, pixels[0])
        viz.assert_written(self, viz.save_panels("blob", "05_contour", [("contour + blob", view)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
