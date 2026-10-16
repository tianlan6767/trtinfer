#!/usr/bin/env python3
"""预处理与 Blob。需要先 make 出 workspace/cvter.so。

不依赖 GPU 的同内容测试：make test-blob
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

try:
    import cvter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"无法 import cvter（需要先 make）: {exc}") from exc


class TestBlob(unittest.TestCase):
    def test_crop_threshold_and_blobs(self):
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
        binary = cvter.morphology(binary, morph)

        filt = cvter.BlobFilter()
        filt.minArea = 30
        blobs = cvter.find_blobs(binary, filt)
        self.assertEqual(len(blobs), 2)
        circ = max(blobs, key=lambda b: b.circularity)
        self.assertGreater(circ.circularity, 0.7)
        self.assertLess(abs(circ.center.x - 55) + abs(circ.center.y - 60), 4.0)

        filt.minCircularity = 0.7
        self.assertEqual(len(cvter.find_blobs(binary, filt)), 1)

        bars = cvter.BlobFilter()
        bars.minArea = 30
        bars.minAspect = 3.0
        only = cvter.find_blobs(binary, bars)
        self.assertEqual(len(only), 1)
        self.assertGreater(only[0].aspect, 3.0)

        roi_blobs = cvter.find_blobs(binary, cvter.BlobFilter(), 100, 0, 120, 160)
        roi_blobs = [b for b in roi_blobs if b.area >= 30]
        self.assertEqual(len(roi_blobs), 1)
        self.assertGreater(roi_blobs[0].center.x, 100)

        square = np.zeros((80, 90), np.uint8)
        cv2.rectangle(square, (10, 15), (49, 54), 255, -1)
        stats = cvter.contour_stats(square)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats[0].area, 1400)
        self.assertLessEqual(stats[0].area, 1600)
        pixels = cvter.find_blobs(square)
        self.assertEqual(len(pixels), 1)
        self.assertEqual(pixels[0].area, 1600)


if __name__ == "__main__":
    unittest.main(verbosity=2)
