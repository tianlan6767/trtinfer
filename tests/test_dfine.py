#!/usr/bin/env python3
"""DFINE / DFINESAHI Python 绑定冒烟测试。

枚举导出不需要引擎。推理用例在找不到引擎时跳过，不把大文件放进仓库。

查找顺序：
  1. 环境变量 CVTER_DFINE_ENGINE
  2. <repo>/models/engine/dfine_l_obj2coco.engine
  3. <repo>/workspace/models/engine/dfine_l_obj2coco.engine
  4. <repo>/workspace/dfine_l_obj2coco.engine

可选：
  CVTER_DFINE_IMAGE  测试图；缺省用一张合成 BGR 图
  CVTER_DFINE_NAMES  逗号分隔类别名；缺省为 COCO 80 类
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"
sys.path.insert(0, str(WORKSPACE))

try:
    import cvter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"无法 import cvter: {exc}") from exc

# 与 src/examples/dfine/dfine.cpp 一致，避免真实 COCO 引擎 class_id 越界。
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

ENGINE_CANDIDATES = [
    ROOT / "models" / "engine" / "dfine_l_obj2coco.engine",
    WORKSPACE / "models" / "engine" / "dfine_l_obj2coco.engine",
    WORKSPACE / "dfine_l_obj2coco.engine",
]


def class_names() -> list[str]:
    raw = os.environ.get("CVTER_DFINE_NAMES", "").strip()
    if not raw:
        return list(COCO_NAMES)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise unittest.SkipTest("CVTER_DFINE_NAMES 为空")
    return names


def resolve_engine() -> tuple[Path | None, str]:
    env = os.environ.get("CVTER_DFINE_ENGINE", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path, ""
        return None, f"CVTER_DFINE_ENGINE 指向的文件不存在: {path}"
    for path in ENGINE_CANDIDATES:
        if path.is_file():
            return path, ""
    return None, (
        "未找到 DFINE engine。设置 CVTER_DFINE_ENGINE，"
        "或放到 models/engine/dfine_l_obj2coco.engine"
    )


ENGINE, ENGINE_SKIP = resolve_engine()


def load_image():
    import numpy as np

    raw = os.environ.get("CVTER_DFINE_IMAGE", "").strip()
    if not raw:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :] = (32, 64, 96)
        image[40:120, 50:180] = (180, 40, 40)
        return image
    path = Path(raw)
    if not path.is_file():
        raise unittest.SkipTest(f"CVTER_DFINE_IMAGE 指向的文件不存在: {path}")
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise unittest.SkipTest(f"无法读取 CVTER_DFINE_IMAGE: {path}")
    return image


def make_model(model_type, **kwargs):
    params = dict(
        gpu_id=0,
        confidence_threshold=0.5,
        nms_threshold=0.45,
        max_batch_size=4,
        auto_slice=False,
        slice_width=0,
        slice_height=0,
        slice_horizontal_ratio=0.0,
        slice_vertical_ratio=0.0,
    )
    params.update(kwargs)
    return cvter.TrtInfer(
        str(ENGINE),
        model_type,
        class_names(),
        gpu_id=params["gpu_id"],
        confidence_threshold=params["confidence_threshold"],
        nms_threshold=params["nms_threshold"],
        max_batch_size=params["max_batch_size"],
        auto_slice=params["auto_slice"],
        slice_width=params["slice_width"],
        slice_height=params["slice_height"],
        slice_horizontal_ratio=params["slice_horizontal_ratio"],
        slice_vertical_ratio=params["slice_vertical_ratio"],
    )


def assert_detection_batch(test: unittest.TestCase, batch, image) -> None:
    import numpy as np

    # 合成图上允许零检出。有框时只检查 h.box 字段，不要求框落在画面内。
    test.assertEqual(image.ndim, 3)
    test.assertEqual(image.shape[2], 3)
    test.assertIsInstance(batch, list)
    test.assertEqual(len(batch), 1)
    hits = batch[0]
    test.assertIsInstance(hits, list)
    for hit in hits:
        box = hit.box
        test.assertGreaterEqual(box.score, 0.5)
        test.assertIsInstance(box.class_name, str)
        test.assertGreaterEqual(box.class_id, 0)
        test.assertLess(box.class_id, len(class_names()))
        test.assertGreaterEqual(box.right, box.left)
        test.assertGreaterEqual(box.bottom, box.top)
        test.assertTrue(np.isfinite([box.left, box.top, box.right, box.bottom, box.score]).all())


class TestDfineModelType(unittest.TestCase):
    def test_enum_exported(self):
        dfine = cvter.ModelType.DFINE
        sahi = cvter.ModelType.DFINESAHI
        self.assertEqual(int(dfine), 10)
        self.assertEqual(int(sahi), 11)
        self.assertNotEqual(dfine, sahi)
        self.assertNotEqual(dfine, cvter.ModelType.YOLO11)
        self.assertNotEqual(sahi, cvter.ModelType.YOLO11SAHI)
        self.assertNotEqual(dfine, cvter.ModelType.DEEPLABV3)
        self.assertIn("DFINE", repr(dfine))
        self.assertIn("DFINESAHI", repr(sahi))


@unittest.skipUnless(ENGINE is not None, ENGINE_SKIP)
class TestDfineInfer(unittest.TestCase):
    def test_forwards_boxes(self):
        image = load_image()
        model = make_model(cvter.ModelType.DFINE)
        self.assertTrue(model.valid)
        batch = model.forwards([image])
        assert_detection_batch(self, batch, image)

    def test_sahi_forwards_boxes(self):
        image = load_image()
        # 切片大于图，只有一块，不要求大 batch。
        model = make_model(
            cvter.ModelType.DFINESAHI,
            slice_width=640,
            slice_height=640,
            slice_horizontal_ratio=0.2,
            slice_vertical_ratio=0.2,
        )
        self.assertTrue(model.valid)
        batch = model.forwards([image])
        assert_detection_batch(self, batch, image)


if __name__ == "__main__":
    unittest.main(verbosity=2)
