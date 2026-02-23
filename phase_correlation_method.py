import cv2
import numpy as np

def estimate_shift_phase(img1, img2):
    # 转灰度
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 下采样（可选，提升速度）
    scale = 0.3  # 或更低
    g1 = cv2.resize(g1, (0, 0), fx=scale, fy=scale)
    g2 = cv2.resize(g2, (0, 0), fx=scale, fy=scale)

    # 创建 Hanning 窗口减少边缘干扰
    win = cv2.createHanningWindow(g1.shape[::-1], cv2.CV_32F)

    # 转为 float32
    f1 = np.float32(g1) * win
    f2 = np.float32(g2) * win

    # 相位相关法估计平移
    (dx, dy), resp = cv2.phaseCorrelate(f1, f2)
    dx /= scale
    dy /= scale

    return dx, dy, resp