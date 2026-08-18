import numpy as np
import cv2
from collections import Counter

# -------- 参数设置 --------
num_holes = 13
theory_left_x = 785
theory_left_y = 1600  # 可以取平均 y
hole_distance = 250
tolerance = 30  # 孔间微偏差容差
offset_range = 100  # 图片整体偏移容差

img_w, img_h = 4096, 3072
img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

# -------- 实际检测到的孔中心坐标 --------
detected_holes = [
    (782.5, 1597.5), (1022.5, 1597.5), (1262.5, 1597.5), (1497.5, 1597.5),
    (1737.5, 1597.5), (1977.5, 1597.5), (2217.5, 1597.5), (2452.5, 1597.5),
    (2692.5, 1597.5), (2932.5, 1597.5), (3172.5, 1602.5), (3407.5, 1602.5)
]

# -------- Offset 投票法 --------
dx_candidates = []

for x_det, y_det in detected_holes:
    for i in range(num_holes):
        dx = x_det - (i * hole_distance + theory_left_x)
        if abs(dx) <= offset_range:
            dx_candidates.append(dx)

dx_best = Counter(dx_candidates).most_common(1)[0][0] if dx_candidates else 0

# -------- 对齐后的理论孔位置 --------
aligned_theory_positions = [(theory_left_x + dx_best + i*hole_distance, theory_left_y) for i in range(num_holes)]

# -------- 判断缺失孔并分类 --------
missing_indices = []
missing_positions = {"left": [], "middle": [], "right": []}

for i, (tx, ty) in enumerate(aligned_theory_positions):
    found = False
    for dx, dy in detected_holes:
        if abs(tx - dx) <= tolerance and abs(ty - dy) <= tolerance:
            found = True
            break
    if not found:
        missing_indices.append(i)
        # 分类
        if tx < img_w / 3:
            missing_positions["left"].append(i)
        elif tx < img_w * 2 / 3:
            missing_positions["middle"].append(i)
        else:
            missing_positions["right"].append(i)

# -------- 输出结果 --------
print("最佳偏移 dx_best:", dx_best)
print("缺失孔索引（从0开始）：", missing_indices)
print("缺失孔位置分类：", missing_positions)

# -------- 可视化 --------
for i, (tx, ty) in enumerate(aligned_theory_positions):
    color = (0, 255, 0) if i not in missing_indices else (0, 0, 255)  # 红色表示缺失
    cv2.circle(img, (int(tx), int(ty)), 20, color, 2)

for dx, dy in detected_holes:
    cv2.circle(img, (int(dx), int(dy)), 10, (255, 255, 0), -1)  # 蓝色表示检测到孔

# -------- 使用 namedWindow 可缩放显示 --------
cv2.namedWindow("Holes Check", cv2.WINDOW_NORMAL)
scale = 0.5  # 缩放比例
img_small = cv2.resize(img, (int(img_w*scale), int(img_h*scale)))
cv2.imshow("Holes Check", img_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
