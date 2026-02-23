from feature_based_matching_method import *
from phase_correlation_method import *
from laplacian_blending import *
import time
import os

def simple_append_side(stitched, img2, dx, side=100):
    """
    将 img2 的 [side-dx, side) 区间拼接到 stitched 右边，
    边界两列进行加权平均处理，保留两列融合区。
    """
    dx = int(round(dx))
    h_s, w_s = stitched.shape[:2]
    h2, w2 = img2.shape[:2]
    assert h_s == h2, "两张图像高度必须一致"
    assert 0 <= side <= w2, f"side({side}) 必须在 [0, {w2}] 范围内"
    assert dx <= side, f"dx({dx}) 不能大于 side({side})"

    if dx > 1:
        # stitched 最右侧一列
        stitched_right = stitched[:, -1:].astype(np.float32)
        # img2 区间的第一列（也就是 side-dx 列）
        img2_boundary = img2[:, side-dx:side-dx+1].astype(np.float32)

        # 加权生成两个融合列
        blend_col1 = (0.7 * stitched_right + 0.3 * img2_boundary).astype(np.uint8)
        blend_col2 = (0.3 * stitched_right + 0.7 * img2_boundary).astype(np.uint8)

        # 要拼接的纯图像部分：side-dx+1 到 side-1 这 dx-1 列
        patch = img2[:, side-dx+1:side]

        # 拼接：去掉 stitched 最后一列，插入 blend_col1、blend_col2，再加上 patch
        new_stitched = np.concatenate(
            [stitched[:, :-1], blend_col1, blend_col2, patch],
            axis=1
        )

    elif dx > 0:
        # 当 dx == 1 时，只能拼接 side-1 这一列，不做融合
        patch = img2[:, side-dx:side]
        new_stitched = np.concatenate([stitched, patch], axis=1)

    else:
        # dx <= 0，不拼接
        new_stitched = stitched.copy()

    return new_stitched


def pipeline(img_list, mask_list=[], confidence=0.3,output_path=""):
    confidence = 0

    start_time = time.time()
    N = 100   # 每隔 N 帧记录一次
    initial_width = img_list[0].shape[1]//2
    current_width = initial_width
    max_width = initial_width

    img = cv2.imread(img_list[0])
    side=100        #拼接位置，决定了是图像亮的地方拼接还是暗的地方拼接，中间亮两侧暗
    stitched = img[:, :side].copy()

    position_dict={}   # 记录米数的对应位置

    max_base=os.path.basename(img_list[0])
    max_base = int(os.path.splitext(max_base)[0]) 

    min_base=os.path.basename(img_list[-1])
    min_base = int(os.path.splitext(min_base)[0]) 

    distance_info = {max_base-v+min_base: k for k, v in distance_info.items()}
    # if textLog is None:
    #     # 普通终端模式
    #     iterator = tqdm.tqdm(image_files, total=len(image_files), desc="图像拼接")
    # else:
    #     # UI 模式
    #     logger = QTextEditLogger(textLog)
    #     iterator = tqdm.tqdm(image_files, total=len(image_files), desc="图像拼接", file=logger)

    for i in range(len(img_list) - 1):
        img1 = cv2.imread(img_list[i])
        img2 = cv2.imread(img_list[i + 1])
        img_list = [img1, img2]

    # 特征提取与匹配
        try:
            dx,dy,_=estimate_shift_phase(img1,img2)
            dx=int(round(dx))
            # 防止局部的拉扯震动
            if current_width-dx>max_width:
                max_width=current_width-dx
                current_width=max_width
                stitched=simple_append_side(stitched,img2,-dx,side=side)
            else:
                current_width -= dx
                # continue
        except:
            try:
                
                extractor = Features_get(confidence=confidence, img_list=img_list, mask_list=[])
                features, matches = extractor.process_features()

                dx_list = calculate_points_dx(features, matches)
                final_dx = histogram_vote_dt(dx_list)
                final_dx=int(round(final_dx))

                # 防止局部拉扯振动
                if current_width-dx>max_width:
                    stitched = simple_append_side(stitched, img2, final_dx)
                    current_width -= final_dx
                    max_width=current_width
                    print(f"特征点{i}")
                else:
                    current_width-=final_dx
                    # continue
            except:
                overlap=100
                leveln=7
                img2_hardstitched=img2[:,side//4:]
                stitched=multi_band_blending(stitched, img2_hardstitched, None, overlap, leveln=leveln, flag_half=False, need_mask=True)#就改上述两个参数就好了，bool值不要轻易动
                stitched=stitched[:,:stitched.shape[1]-(img2.shape[1]-side)//4]
                current_width += int((img2.shape[1]*3)//4)
                max_width=current_width
                print(f"硬拼消边：{i}-{i+1}")
        if i%N==0:
            cv2.imwrite(os.path.join(output_path, f"result_{i}.png"), stitched)
    end_time = time.time()
    print(f"[INFO] 拼接完成，总耗时: {end_time - start_time} 秒")
    print(position_dict)

    # 保存图像
    final_output_path = os.path.join(output_path, "final_result_annotated_vertical.png")
    cv2.imwrite(final_output_path, stitched)
    print(f"[INFO] 竖向图像保存至 {final_output_path}，并完成左侧标注")
    return stitched ,position_dict
