#!/usr/bin/env python3
"""DeepLabV3 TensorRT 语义分割冒烟测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"
sys.path.insert(0, str(WORKSPACE))

try:
    import cvter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"无法 import cvter: {exc}") from exc

ENGINE = WORKSPACE / "deepv3" / "logs_tgwy_v3" / "best_epoch_weights.trtmodel"
IMAGE = WORKSPACE / "deepv3" / "datasets" / "tgwy" / "VOC2007" / "JPEGImages" / "101_PMST_反射率图_crop.jpg"
NAMES = ["_background_", "胶路", "基准面"]


@unittest.skipUnless(ENGINE.is_file() and IMAGE.is_file(), "缺少 DeepLab engine 或测试图")
class TestDeeplabV3(unittest.TestCase):
    def test_find_foreground(self):
        img = cv2.imdecode(np.fromfile(str(IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        model = cvter.TrtInfer(
            str(ENGINE),
            cvter.ModelType.DEEPLABV3,
            NAMES,
            0,
            0.2,
            0.0,
            1,
            False,
            0,
            0,
            0.0,
            0.0,
        )
        self.assertTrue(model.valid)
        batch = model.forwards([img])
        self.assertEqual(len(batch), 1)
        hits = batch[0]
        self.assertGreater(len(hits), 0)
        names = {h.box.class_name for h in hits}
        self.assertTrue(names & {"胶路", "基准面"})
        for h in hits:
            self.assertGreater(h.seg.size, 0)
            self.assertGreater(h.box.score, 0.2)
            self.assertGreater(h.box.right - h.box.left, 2)
            self.assertGreater(h.box.bottom - h.box.top, 2)

    def test_class_map(self):
        img = cv2.imdecode(np.fromfile(str(IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        model = cvter.TrtInfer(
            str(ENGINE),
            cvter.ModelType.DEEPLABV3,
            NAMES,
            0,
            0.2,
            0.0,
            1,
            False,
            0,
            0,
            0.0,
            0.0,
            cvter.SegOutput.CLASS_MAP,
        )
        self.assertTrue(model.valid)
        hits = model.forwards([img])[0]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].box.class_name, "class_map")
        cls = hits[0].seg
        self.assertEqual(cls.shape, img.shape[:2])
        self.assertEqual(cls.dtype, np.uint8)
        ids = set(np.unique(cls).tolist())
        self.assertTrue(ids & {1, 2})


@unittest.skipUnless(ENGINE.is_file() and IMAGE.is_file(), "缺少 DeepLab engine 或测试图")
class TestDeeplabV3Sahi(unittest.TestCase):
    def test_sahi_covers_image(self):
        img = cv2.imdecode(np.fromfile(str(IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        model = cvter.TrtInfer(
            str(ENGINE),
            cvter.ModelType.DEEPLABV3SAHI,
            NAMES,
            0,
            0.2,
            0.0,
            1,
            False,
            960,
            960,
            0.1,
            0.1,
        )
        self.assertTrue(model.valid)
        hits = model.forwards([img])[0]
        self.assertGreater(len(hits), 0)
        names = {h.box.class_name for h in hits}
        self.assertTrue(names & {"胶路", "基准面"})
        max_right = max(h.box.right for h in hits)
        max_bottom = max(h.box.bottom for h in hits)
        self.assertGreater(max_right, img.shape[1] * 0.5)
        self.assertGreater(max_bottom, img.shape[0] * 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
