import cv2
import os
import math
import numpy as np
from pathlib import Path
import os.path as osp

def slice_image(
    img,
    slice_width,
    slice_height,
    slice_horizontal_ratio=0.0,
    slice_vertical_ratio=0.0,
    drop_last=True
):
    H, W = img.shape[:2]

    step_x = int(slice_width * (1 - slice_horizontal_ratio))
    step_y = int(slice_height * (1 - slice_vertical_ratio))

    slices = []

    for y in range(0, H - slice_height + 1, step_y):
        for x in range(0, W - slice_width + 1, step_x):
            crop = img[y:y+slice_height, x:x+slice_width]
            slices.append((x, y, crop))

    # 👉 处理右边/下边剩余区域（可选）
    if not drop_last:
        # 最右列
        for y in range(0, H - slice_height + 1, step_y):
            x = W - slice_width
            crop = img[y:y+slice_height, x:x+slice_width]
            slices.append((x, y, crop))

        # 最下行
        for x in range(0, W - slice_width + 1, step_x):
            y = H - slice_height
            crop = img[y:y+slice_height, x:x+slice_width]
            slices.append((x, y, crop))

        # 右下角
        x = W - slice_width
        y = H - slice_height
        crop = img[y:y+slice_height, x:x+slice_width]
        slices.append((x, y, crop))

    return slices

if __name__ == "__main__":
    imp = r"/home/ps/workspace/trt/cvter/workspace/散热片-aoi/src/TnM_X9601_EVT_967A1_AOI_20260416_DYHHS9B0A2M0001HJX+EF61P_20260416210601_3_1_1_Color.jpg"
    dst = r"/home/ps/workspace/trt/cvter/workspace/散热片-aoi/src/slice2560"
    Path(dst).mkdir(parents=True, exist_ok=True)
    img = cv2.imdecode(np.fromfile(imp, dtype=np.uint8), cv2.IMREAD_COLOR)

    slices = slice_image(
        img,
        slice_width=1280,
        slice_height=1280,
        slice_horizontal_ratio=0.0125,
        slice_vertical_ratio=0.0125,
        drop_last=False
    )
    for idx, (x, y, crop) in enumerate(slices):
        cv2.imencode(".jpg", crop)[1].tofile(osp.join(dst, f"slice_{idx}.jpg"))
    print("切图数量:", len(slices))