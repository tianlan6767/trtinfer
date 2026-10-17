"""传统视觉测试的结果图落盘。只 imwrite，不 imshow。"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "out_viz"

GREEN = (40, 190, 40)
RED = (40, 40, 220)
YELLOW = (0, 220, 220)
CYAN = (220, 210, 40)
MAGENTA = (210, 40, 210)
WHITE = (240, 240, 240)
ORANGE = (30, 150, 255)


def prepare(group: str, clear: bool = True) -> Path:
    """创建 tests/out_viz/<group>。clear 时删掉该组已有图片。"""
    folder = OUT_DIR / group
    folder.mkdir(parents=True, exist_ok=True)
    if clear:
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
    return folder


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def as_points(obj) -> np.ndarray:
    if obj is None:
        return np.zeros((0, 2), np.float64)
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return np.zeros((0, 2), np.float64)
        return np.asarray(obj, np.float64).reshape(-1, 2)
    pts = []
    for p in obj:
        if hasattr(p, "x"):
            pts.append((float(p.x), float(p.y)))
        else:
            pts.append((float(p[0]), float(p[1])))
    if not pts:
        return np.zeros((0, 2), np.float64)
    return np.asarray(pts, np.float64)


def put_text(img: np.ndarray, text: str, org, color=WHITE, scale=0.45, thickness=1) -> None:
    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_cross(img, x, y, color=GREEN, size=10, thickness=1) -> None:
    cv2.drawMarker(img, (int(round(x)), int(round(y))), color, cv2.MARKER_CROSS, size, thickness, cv2.LINE_AA)


def draw_poly(img, points, color=GREEN, thickness=2, closed=True) -> None:
    pts = as_points(points)
    if len(pts) < 2:
        return
    chain = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [chain], closed, color, thickness, cv2.LINE_AA)


def draw_points(img, points, color=CYAN, radius=2) -> None:
    for x, y in as_points(points):
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)


def draw_search_rect(img, x0, y0, x1, y1, color=YELLOW) -> None:
    cv2.rectangle(img, (int(round(min(x0, x1))), int(round(min(y0, y1)))),
                  (int(round(max(x0, x1))), int(round(max(y0, y1)))), color, 1, cv2.LINE_AA)


def draw_oriented_search(img, cx, cy, phi_deg, length1, length2, color=YELLOW) -> None:
    phi = math.radians(phi_deg)
    ux, uy = math.cos(phi), math.sin(phi)
    sx, sy = math.sin(phi), -math.cos(phi)
    corners = []
    for s1, s2 in ((-length1, -length2), (length1, -length2), (length1, length2), (-length1, length2)):
        corners.append((cx + ux * s1 + sx * s2, cy + uy * s1 + sy * s2))
    draw_poly(img, corners, color, 1, True)


def draw_annulus(img, cx, cy, radius, search, start_deg=0, end_deg=360, color=YELLOW) -> None:
    axes_out = (int(round(radius + search)), int(round(radius + search)))
    axes_in = (max(1, int(round(radius - search))), max(1, int(round(radius - search))))
    center = (int(round(cx)), int(round(cy)))
    cv2.ellipse(img, center, axes_out, 0, start_deg, end_deg, color, 1, cv2.LINE_AA)
    cv2.ellipse(img, center, axes_in, 0, start_deg, end_deg, color, 1, cv2.LINE_AA)


def draw_blob(img, blob, color=GREEN) -> None:
    x, y, w, h = int(blob.left), int(blob.top), int(blob.width), int(blob.height)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    draw_cross(img, blob.center.x, blob.center.y, color, 12)
    rad = math.radians(blob.angle)
    x2 = blob.center.x + math.cos(rad) * max(w, h) * 0.45
    y2 = blob.center.y + math.sin(rad) * max(w, h) * 0.45
    cv2.line(img, (int(round(blob.center.x)), int(round(blob.center.y))),
             (int(round(x2)), int(round(y2))), color, 1, cv2.LINE_AA)
    put_text(img, f"a={blob.area:.0f} c={blob.circularity:.2f} asp={blob.aspect:.2f}",
             (x, max(12, y - 4)), color, 0.38)


def draw_match(img, result, color=GREEN) -> None:
    draw_poly(img, [result.LeftTop, result.RightTop, result.RightBottom, result.LeftBottom], color, 2)
    draw_cross(img, result.Center.x, result.Center.y, color, 14, 2)
    put_text(
        img,
        f"score={result.Score:.2f} ang={result.Angle:.1f} sc={result.Scale:.2f}",
        (int(result.Center.x) + 6, max(14, int(result.Center.y) - 8)),
        color,
        0.45,
    )


def draw_line_result(img, result, color=GREEN, label="") -> None:
    draw_points(img, getattr(result, "points", []), CYAN, 2)
    draw_points(img, getattr(result, "inliers", []), MAGENTA, 3)
    if result.found:
        cv2.line(img, (int(round(result.p1.x)), int(round(result.p1.y))),
                 (int(round(result.p2.x)), int(round(result.p2.y))), color, 2, cv2.LINE_AA)
        draw_cross(img, result.center.x, result.center.y, color, 10)
        text = f"{label} ang={result.angle:.1f} rms={result.rms:.3f} n={result.numInliers}/{result.numPoints}"
    else:
        text = f"{label} MISS"
        color = RED
    put_text(img, text.strip(), (8, 18 if not label else 18), color, 0.42)


def draw_circle_result(img, result, color=GREEN) -> None:
    draw_points(img, getattr(result, "points", []), CYAN, 2)
    draw_points(img, getattr(result, "inliers", []), MAGENTA, 3)
    if result.found:
        cv2.circle(img, (int(round(result.center.x)), int(round(result.center.y))),
                   max(1, int(round(result.radius))), color, 2, cv2.LINE_AA)
        draw_cross(img, result.center.x, result.center.y, color, 12)
        put_text(img, f"r={result.radius:.2f} rms={result.rms:.3f} n={result.numInliers}",
                 (8, 18), color, 0.45)
    else:
        put_text(img, "CIRCLE MISS", (8, 18), RED, 0.5)


def draw_rect_result(img, result) -> None:
    for side, color in ((result.top, CYAN), (result.right, CYAN), (result.bottom, CYAN), (result.left, CYAN)):
        if side is None:
            continue
        draw_points(img, getattr(side, "points", []), color, 2)
        if getattr(side, "found", False):
            cv2.line(img, (int(round(side.p1.x)), int(round(side.p1.y))),
                     (int(round(side.p2.x)), int(round(side.p2.y))), ORANGE, 1, cv2.LINE_AA)
    if result.found:
        draw_poly(img, [result.topLeft, result.topRight, result.bottomRight, result.bottomLeft], GREEN, 2)
        draw_cross(img, result.center.x, result.center.y, GREEN, 12)
        put_text(img, f"w={result.width:.1f} h={result.height:.1f} ang={result.angle:.2f} rms={result.rms:.3f}",
                 (8, 18), GREEN, 0.45)
    else:
        put_text(img, "RECT MISS", (8, 18), RED, 0.5)


def zoom_inset(img, x, y, scale=8, half=12) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(round(x)) - half
    y0 = int(round(y)) - half
    patch = np.full((half * 2, half * 2, 3), 20, np.uint8)
    for yy in range(half * 2):
        for xx in range(half * 2):
            sx, sy = x0 + xx, y0 + yy
            if 0 <= sx < w and 0 <= sy < h:
                patch[yy, xx] = img[sy, sx]
    big = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    fx = (x - x0) * scale
    fy = (y - y0) * scale
    cv2.drawMarker(big, (int(round(fx)), int(round(fy))), (0, 0, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
    return big


def save_panels(group: str, name: str, panels) -> Path:
    """panels: [(title, gray_or_bgr), ...]。横向拼接后写入 PNG。"""
    tiles = []
    height = 0
    for title, image in panels:
        bgr = to_bgr(np.asarray(image))
        bar = np.full((22, bgr.shape[1], 3), 32, np.uint8)
        put_text(bar, title, (6, 16), WHITE, 0.42)
        tile = np.vstack([bar, bgr])
        tiles.append(tile)
        height = max(height, tile.shape[0])
    padded = []
    for tile in tiles:
        if tile.shape[0] < height:
            pad = np.full((height - tile.shape[0], tile.shape[1], 3), 20, np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    gap = np.full((height, 6, 3), 20, np.uint8)
    parts = []
    for i, tile in enumerate(padded):
        if i:
            parts.append(gap)
        parts.append(tile)
    canvas = np.hstack(parts) if parts else np.zeros((8, 8, 3), np.uint8)
    path = OUT_DIR / group / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"imwrite failed: {path}")
    return path


def assert_written(test, path: Path) -> None:
    test.assertTrue(path.is_file(), path)
    test.assertGreater(path.stat().st_size, 64, path)


def fractional_max(points) -> float:
    pts = as_points(points)
    if len(pts) == 0:
        return 0.0
    frac = np.abs(pts - np.round(pts))
    return float(frac.max())
