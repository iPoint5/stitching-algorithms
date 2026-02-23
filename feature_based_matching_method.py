import cv2
import numpy as np
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
import random
import sys
from stitching import AffineStitcher,Stitcher
import math


class Features_get(Stitcher):
    def __init__(self, img_list, mask_list, confidence=0.3):
        # 先初始化父类（会设置各种 affine 默认值）
        super().__init__()
        
        self.img_list = img_list
        self.mask_list = mask_list
        self.confidence = confidence

        # 这里覆盖默认的 confidence 阈值
        self.subsetter = Subsetter(
            self.confidence,
            Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE
        )

        # 特征检测器
        self.detector = FeatureDetector(
            # FeatureDetector.DEFAULT_DETECTOR,
            'sift',
            nfeatures=500
        )

        # 特征匹配器
        self.matcher = FeatureMatcher(
            FeatureMatcher.DEFAULT_MATCHER,
            FeatureMatcher.DEFAULT_RANGE_WIDTH,
            try_use_gpu=False,
            match_conf=None
        )

        # 保持和父类一致的分辨率设置
        self.medium_megapix = Images.Resolution.MEDIUM.value
        self.low_megapix   = Images.Resolution.LOW.value
        self.final_megapix = Images.Resolution.FINAL.value

    def process_features(self):
        # 初始化 Images 对象
        self.images = Images.of(
            self.img_list,
            self.medium_megapix,
            self.low_megapix,
            self.final_megapix
        )

        # 降到中分辨率，提取特征、匹配并筛选
        imgs     = self.resize_medium_resolution()
        features = self.find_features(imgs, self.mask_list)
        matches  = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)

        # 还原坐标
        scale = self.images._scalers["MEDIUM"].scale
        for feat in features:
            for kp in feat.keypoints:
                kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

        return features, matches


def calculate_points_dx(features, matches):
    """
    从匹配的特征点中提取每一对的横坐标差(dx = pt2_x - pt1_x)
    并返回一个 dx 列表。
    
    参数:
    - features: 图像特征列表(cv2.detail_ImageFeatures)
    - matches: 图像之间的匹配结果列表(cv2.detail_MatchesInfo)

    返回:
    - dx_list: 所有特征点对的横向偏移值（float）
    """
    dx_list = []

    for m in matches:
         # 可选过滤置信度较低的点对

        i = m.src_img_idx
        j = m.dst_img_idx

        for match in m.matches:
            pt1 = features[i].keypoints[match.queryIdx].pt  # (x1, y1)
            pt2 = features[j].keypoints[match.trainIdx].pt  # (x2, y2)
            dx = pt2[0] - pt1[0]  # 仅计算 x 方向偏移
            dx_list.append(dx)

    return dx_list

def calculate_points_dy(features, matches):
    """
    从匹配的特征点中提取每一对的横坐标差(dx = pt2_x - pt1_x)
    并返回一个 dx 列表。
    
    参数:
    - features: 图像特征列表(cv2.detail_ImageFeatures)
    - matches: 图像之间的匹配结果列表(cv2.detail_MatchesInfo)

    返回:
    - dx_list: 所有特征点对的横向偏移值（float）
    """
    dy_list = []

    for m in matches:
         # 可选过滤置信度较低的点对

        i = m.src_img_idx
        j = m.dst_img_idx

        for match in m.matches:
            pt1 = features[i].keypoints[match.queryIdx].pt  # (x1, y1)
            pt2 = features[j].keypoints[match.trainIdx].pt  # (x2, y2)
            dy = pt2[1] - pt1[1]  # 仅计算 x 方向偏移
            dy_list.append(dy)

    return dy_list
def ransac_dx(disps_x, iters=100, tol=2.0):
    """
    Estimate horizontal translation (dx) using RANSAC on 1D displacements.

    Parameters:
    - disps_x: array-like of float, the dx values for each matched point pair
    - iters: int, number of RANSAC iterations
    - tol: float, inlier threshold in pixels

    Returns:
    - final_dx: float, estimated horizontal translation
    - inliers_mask: numpy.ndarray of bool, mask of inlier displacements
    """
    disps_x = np.array(disps_x)
    best_inliers = None
    best_count = 0
    best_model = 0.0

    n = len(disps_x)
    for _ in range(iters):
        i1, i2 = random.sample(range(n), 2)
        candidate = (disps_x[i1] + disps_x[i2]) / 2.0
        inliers = np.abs(disps_x - candidate) < tol
        count = np.sum(inliers)
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = candidate

    # Compute final dx from inliers
    if best_inliers is None or best_count == 0:
        final_dx = np.median(disps_x)
        inliers_mask = np.ones(n, dtype=bool)
    else:
        final_dx = np.median(disps_x[best_inliers])
        inliers_mask = best_inliers

    return final_dx, inliers_mask

def histogram_vote_dt(disps_x, bin_size=1.0):
    """
    Estimate horizontal translation (dx) using histogram voting (mode).

    Parameters:
    - disps_x: array-like of float, the dx values for each matched point pair
    - bin_size: float, size of histogram bin in pixels

    Returns:
    - mode_dx: float, estimated horizontal translation (peak bin center)
    """
    disps_x = np.array(disps_x)
    # Discretize to bins
    bins = np.floor(disps_x / bin_size).astype(int)
    unique_bins, counts = np.unique(bins, return_counts=True)
    peak_bin = unique_bins[np.argmax(counts)]
    mode_dx = peak_bin * bin_size
    return mode_dx


    