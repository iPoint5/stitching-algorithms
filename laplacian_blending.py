import cv2
import numpy as np
import sys


def preprocess(img1, img2, overlap_w, flag_half, need_mask=False):
    # 先判断这个图片是否符合要求
    '''
    img1, img2: 输入图像
    overlap_w: width of overlapped area, a positive integer
    flag_half: if True, 两张图像融合成原来的一张大小，否则就是两图像宽减去重叠
    '''
    if img1.shape[0] != img2.shape[0]:
        print ("error: image dimension error")
        sys.exit()
    if overlap_w > img1.shape[1] or overlap_w > img2.shape[1]:
        print ("error: overlapped area too large")
        sys.exit()

    w1 = img1.shape[1]
    w2 = img2.shape[1]

    if flag_half:
        shape = np.array(img1.shape) # 建立了一个空的数组
        shape[1] = w1 // 2 + w2 // 2

        subA = np.zeros(shape)
        subA[:, :w1 // 2 + overlap_w // 2] = img1[:, :w1 // 2 + overlap_w // 2]
        subB = np.zeros(shape)
        subB[:, w1 // 2 - overlap_w // 2:] = img2[:,
                                                w2 - (w2 // 2 + overlap_w // 2):]
        if need_mask:
            mask = np.zeros(shape)
            mask[:, :w1 // 2] = 1
            return subA, subB, mask
    else:
        shape = np.array(img1.shape)
        shape[1] = w1 + w2 - overlap_w

        subA = np.zeros(shape)
        subA[:, :w1] = img1
        subB = np.zeros(shape)
        subB[:, w1 - overlap_w:] = img2
        if need_mask:
            mask = np.zeros(shape)
            mask[:, :w1 - overlap_w // 2] = 1
            return subA, subB, mask

    return subA, subB, None

def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP

#修改以适应图像尺寸为奇数的情况
def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        up = cv2.pyrUp(next_img)

        # 裁剪 up 到 img 的大小
        if up.shape != img.shape:
            up = up[:img.shape[0], :img.shape[1]]

        LP.append(img - up)
        img = next_img
    LP.append(img)
    return LP



def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended

#防止上采样出现问题
def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, lev_img.shape[1::-1])
        # 对齐尺寸，防止 broadcasting 错误
        if img.shape != lev_img.shape:
            h = min(img.shape[0], lev_img.shape[0])
            w = min(img.shape[1], lev_img.shape[1])
            img = img[:h, :w]
            lev_img = lev_img[:h, :w]
        img += lev_img
    return img



def multi_band_blending(img1, img2, mask, overlap_w, leveln=None, flag_half=False, need_mask=False):
    if overlap_w < 0:
        print ("error: overlap_w should be a positive integer")
        sys.exit()

    if need_mask:  # no input mask
        subA, subB, mask = preprocess(img1, img2, overlap_w, flag_half, True)
    else:  # have input mask
        subA, subB, _ = preprocess(img1, img2, overlap_w, flag_half)

    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1],
                                          img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print ("warning: inappropriate number of leveln")
        leveln = max_leveln

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result.astype(np.uint8)