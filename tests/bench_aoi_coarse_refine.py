#!/usr/bin/env python3
"""可选的大图粗到精计时。没有图就跳过，不写死机器路径。

用法:
  CVTER_AOI_SCENE=scene.jpg CVTER_AOI_TEMPL=templ.jpg \\
    [CVTER_AOI_MASK=mask.png] PYTHONPATH=objs/pytrad \\
    python3 tests/bench_aoi_coarse_refine.py

  python3 tests/bench_aoi_coarse_refine.py --scene scene.jpg --templ templ.jpg [--mask mask.png]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def resolve_paths(argv: list[str]) -> tuple[Path, Path, Path | None] | None:
    parser = argparse.ArgumentParser(description="Time ORB coarse + ROI Matcher on an AOI pair.")
    parser.add_argument("--scene", default=os.environ.get("CVTER_AOI_SCENE", ""))
    parser.add_argument("--templ", default=os.environ.get("CVTER_AOI_TEMPL", ""))
    parser.add_argument("--mask", default=os.environ.get("CVTER_AOI_MASK", ""))
    args = parser.parse_args(argv)
    if not args.scene or not args.templ:
        return None
    scene = Path(args.scene)
    templ = Path(args.templ)
    mask = Path(args.mask) if args.mask else None
    if not scene.is_file() or not templ.is_file():
        print(f"skip: image not found ({scene} / {templ})")
        return None
    if mask is not None and not mask.is_file():
        print(f"skip: mask not found ({mask})")
        return None
    return scene, templ, mask


def load_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"failed to read {path}")
    return image


def run_case(cvter, name: str, scene, templ, mask, param) -> None:
    t0 = time.perf_counter()
    result = cvter.match_coarse_to_fine(scene, templ, mask, param)
    elapsed = time.perf_counter() - t0
    roi = result.roi
    print(
        f"{name:24} {elapsed:7.3f}s found={result.found} "
        f"inliers={result.coarse.inliers} "
        f"roi=({roi.x},{roi.y},{roi.width}x{roi.height}) {result.message}"
    )
    for hit in result.hits[:1]:
        print(
            f"{'':24} score={hit.Score:.4f} "
            f"LT=({hit.LeftTop.x:.1f},{hit.LeftTop.y:.1f}) "
            f"C=({hit.Center.x:.1f},{hit.Center.y:.1f}) ang={hit.Angle:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    paths = resolve_paths(sys.argv[1:] if argv is None else argv)
    if paths is None:
        print("skip: set CVTER_AOI_SCENE and CVTER_AOI_TEMPL, or pass --scene and --templ")
        return 0
    scene_path, templ_path, mask_path = paths
    try:
        import cvter
    except ImportError as exc:
        print(f"cannot import cvter ({exc}). Set PYTHONPATH to objs/pytrad or workspace.")
        return 1

    scene = load_gray(scene_path)
    templ = load_gray(templ_path)
    if mask_path is not None:
        mask = (load_gray(mask_path) > 0).astype(np.uint8)
    else:
        mask = np.zeros((0, 0), np.uint8)
    print(f"scene={scene.shape[1]}x{scene.shape[0]} templ={templ.shape[1]}x{templ.shape[0]}")

    pattern = cvter.CoarseRefineParam()
    pattern.detector = cvter.FeatureDetector.ORB
    pattern.matcherType = cvter.MatcherType.PATTERN
    pattern.angle = 0
    pattern.margin = 120
    pattern.scoreThreshold = 0.5
    pattern.maxCount = 1
    pattern.minArea = 1024
    pattern.iouThreshold = 0.3
    pattern.meanBorder = True
    pattern.stopLayer = 1

    shape = cvter.CoarseRefineParam()
    shape.detector = cvter.FeatureDetector.ORB
    shape.matcherType = cvter.MatcherType.SHAPE
    shape.angle = 5
    shape.margin = 120
    shape.scoreThreshold = 0.4
    shape.maxCount = 1
    shape.minArea = 1024
    shape.iouThreshold = 0.3
    shape.edgeMinMag = 12
    shape.maxEdgePoints = 400
    shape.usePolarity = True
    shape.greediness = 0.8
    shape.stopLayer = 1

    run_case(cvter, "ORB+PATTERN angle=0", scene, templ, mask, pattern)
    run_case(cvter, "ORB+SHAPE angle=5", scene, templ, mask, shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
