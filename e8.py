import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import argparse
import os

# 支持中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_image(img_input):
    if isinstance(img_input, str):
        img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_input}")
        return img
    elif isinstance(img_input, np.ndarray):
        if len(img_input.shape) == 3:
            return cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        return img_input
    raise TypeError("输入必须为图片路径或 numpy 数组")


def load_and_binarize(img_input):
    img = load_image(img_input)

    # 大津法自动寻找双峰间的最佳阈值
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 统计图像四周边缘像素的平均灰度，判断背景色
    edge_mean = (np.mean(thresh[0, :]) + np.mean(thresh[-1, :]) + np.mean(thresh[:, 0]) + np.mean(thresh[:, -1])) / 4

    if edge_mean > 127:
        # 白底黑物 -> 自动反色，使背景变黑(0)，目标变白(255)
        return 255 - thresh
    return thresh

def _erode(img_bin, kernel):
    bin_01 = (img_bin > 127).astype(np.uint8)
    kernel_sum = np.sum(kernel)
    conv = convolve2d(bin_01, kernel, mode='same', boundary='fill', fillvalue=0)
    return (conv >= kernel_sum).astype(np.uint8) * 255


def _dilate(img_bin, kernel):
    bin_01 = (img_bin > 127).astype(np.uint8)
    conv = convolve2d(bin_01, kernel, mode='same', boundary='fill', fillvalue=0)
    return (conv > 0).astype(np.uint8) * 255

def manual_erode(img_input, kernel):
    img_bin = load_and_binarize(img_input)
    return _erode(img_bin, kernel)


def manual_dilate(img_input, kernel):
    img_bin = load_and_binarize(img_input)
    return _dilate(img_bin, kernel)


def manual_open(img_input, kernel):
    img_bin = load_and_binarize(img_input)
    # 先腐蚀后膨胀
    return _dilate(_erode(img_bin, kernel), kernel)


def manual_close(img_input, kernel):
    img_bin = load_and_binarize(img_input)
    # 先膨胀后腐蚀
    return _erode(_dilate(img_bin, kernel), kernel)


def manual_hit_or_miss(img_input, B1, B2):
    # 仅在入口处执行一次自适应二值化，确保 A 为黑底白球
    A = load_and_binarize(img_input)
    # Ac 为白底黑球
    Ac = 255 - A

    # 调用纯底层数学算子 _erode，彻底避免二次二值化反色冲突！
    hit = _erode(A, B1)
    miss = _erode(Ac, B2)

    res = np.bitwise_and(hit, miss)
    return res


def manual_skeletonize(img_input, kernel):
    A = load_and_binarize(img_input)
    skeleton = np.zeros_like(A)
    current = A.copy()

    while cv2.countNonZero(current) > 0:
        opened = manual_open(current, kernel)
        # 用底层 _erode 对二值图做逻辑减法，防止直接减法溢出
        diff = cv2.subtract(current, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        current = _erode(current, kernel)

    return skeleton


def experiment_1_basic(img_input=None):
    if img_input is not None:
        img_bin = load_and_binarize(img_input)
    else:
        # 自动生成学术测试图
        img = np.zeros((150, 150), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
        cv2.circle(img, (100, 100), 25, 255, -1)
        cv2.circle(img, (100, 100), 5, 0, -1)
        img[20:25, 100:105] = 255
        img[120:125, 40:45] = 255
        cv2.line(img, (70, 50), (85, 85), 255, 3)
        img_bin = load_and_binarize(img)

    kernel = np.ones((5, 5), np.uint8)

    eroded = _erode(img_bin, kernel)
    dilated = _dilate(img_bin, kernel)
    opened = _dilate(_erode(img_bin, kernel), kernel)
    closed = _erode(_dilate(img_bin, kernel), kernel)

    plt.figure(figsize=(15, 5))
    plt.suptitle("实验八（一）", fontsize=16, fontweight='bold')

    titles = ["1. 原始二值图", "2. 腐蚀", "3. 膨胀", "4. 开运算", "5. 闭运算"]
    imgs = [img_bin, eroded, dilated, opened, closed]

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()


def experiment_2_advanced(img_hm_input=None, img_skel_input=None):
    # ---------- Hit-or-Miss (自动遍历 R 寻找目标球体) ----------
    if img_hm_input is not None:
        img_hm_gray = load_image(img_hm_input)
    else:
        img_hm_gray = np.zeros((150, 150), dtype=np.uint8)
        cv2.rectangle(img_hm_gray, (20, 20), (40, 40), 255, -1)
        cv2.circle(img_hm_gray, (120, 35), 16, 255, -1)
        cv2.rectangle(img_hm_gray, (20, 90), (45, 120), 255, -1)
        cv2.circle(img_hm_gray, (100, 100), 6, 255, -1)  # 默认测试图半径为 6

    # ==================== 🛠️ 全自动遍历寻球算法 ====================
    found_R = None
    hit_miss = None
    B1_best = None
    B2_best = None

    print("\\n开始全自动搜寻契合目标球体尺寸的结构元半径...")

    # 自动从半径 3 像素遍历到 25 像素
    for r_test in range(3, 26):
        # 1. 构造当前半径的 B1 和 B2
        B1_temp = np.zeros((2 * r_test + 9, 2 * r_test + 9), dtype=np.uint8)
        center = (r_test + 4, r_test + 4)
        cv2.circle(B1_temp, center, r_test, 1, -1)

        B2_temp = np.ones((2 * r_test + 9, 2 * r_test + 9), dtype=np.uint8)
        cv2.circle(B2_temp, center, r_test + 2, 0, -1)

        # 2. 尝试击中击不中
        res_temp = manual_hit_or_miss(img_hm_gray, B1_temp, B2_temp)
        active_points = cv2.countNonZero(res_temp)

        if active_points > 0:
            found_R = r_test
            hit_miss = res_temp
            B1_best = B1_temp
            B2_best = B2_temp
            print(f"🎉 自动检测成功！最匹配的目标球体半径为 R = {found_R} 像素！")
            print(f"该半径下，精确定位到目标球心的像素点数: {active_points}\\n")
            break

    # 容错：如果实在没找到，采用 R=8 兜底展示
    if found_R is None:
        print("❌ 遍历 R=3~25 未找到任何匹配球体。已开启 R=8 默认分析。")
        found_R = 8
        B1_best = np.zeros((2 * found_R + 9, 2 * found_R + 9), dtype=np.uint8)
        cv2.circle(B1_best, (found_R + 4, found_R + 4), found_R, 1, -1)
        B2_best = np.ones((2 * found_R + 9, 2 * found_R + 9), dtype=np.uint8)
        cv2.circle(B2_best, (found_R + 4, found_R + 4), found_R + 2, 0, -1)
        hit_miss = manual_hit_or_miss(img_hm_gray, B1_best, B2_best)

    # 膨胀重建与彩色高亮
    detected_circle = _dilate(hit_miss, B1_best)
    img_hm_binary = load_and_binarize(img_hm_gray)
    hm_overlay = cv2.cvtColor(img_hm_binary, cv2.COLOR_GRAY2RGB)
    hm_overlay[detected_circle == 255] = [255, 0, 0]  # 找到的球染成亮红色

    # ---------- Skeleton 骨架提取 ----------
    if img_skel_input is not None:
        img_skel = load_image(img_skel_input)
    else:
        img_skel = np.zeros((150, 150), dtype=np.uint8)
        cv2.rectangle(img_skel, (30, 30), (120, 50), 255, -1)
        cv2.rectangle(img_skel, (65, 50), (85, 120), 255, -1)

    # 骨架提取算子 (十字形)
    cross_kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)

    skeleton = manual_skeletonize(img_skel, cross_kernel)

    # ---------- 2行3列大图展示 ----------
    plt.figure(figsize=(15, 10))
    plt.suptitle("实验八（二）", fontsize=16, fontweight='bold')

    plt.subplot(2, 3, 1)
    plt.imshow(img_hm_binary, cmap='gray')
    plt.title("1. Hit-or-Miss 输入二值图")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(B1_best * 255 + B2_best * 100, cmap='gray')
    plt.title(f"2. 自动寻得的最佳探针结构元(R={found_R})")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(hm_overlay)
    if cv2.countNonZero(hit_miss) > 0:
        plt.title(f"3. 精准定位图", color='green', fontweight='bold')
    else:
        plt.title("3. 检测定位图", color='red')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(load_and_binarize(img_skel), cmap='gray')
    plt.title("4. 骨架原图 (二值化)")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(skeleton, cmap='gray')
    plt.title("5. 提取出的拓扑中轴骨架线")
    plt.axis('off')

    overlay = cv2.cvtColor(load_and_binarize(img_skel), cv2.COLOR_GRAY2RGB)
    overlay[skeleton == 255] = [255, 0, 0]

    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title("6. 骨架中心线叠加对比图", color='green')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ================= 主函数 =================

def main():
    parser = argparse.ArgumentParser(description="Binary morphology experiments: erosion, dilation, hit-or-miss, skeleton.")
    parser.add_argument("--basic-images", nargs="*", default=["8-1.jpg", "8-2.png"],
                        help="images for basic morphology, default: 8-1.jpg 8-2.png")
    parser.add_argument("--hitmiss-image", default="8-4.png", help="image for hit-or-miss, default: 8-4.png")
    parser.add_argument("--skeleton-image", default="8-3.png", help="image for skeleton extraction, default: 8-3.png")
    args = parser.parse_args()

    for image_path in args.basic_images:
        experiment_1_basic(image_path if os.path.exists(image_path) else None)
    hitmiss_image = args.hitmiss_image if os.path.exists(args.hitmiss_image) else None
    skeleton_image = args.skeleton_image if os.path.exists(args.skeleton_image) else None
    experiment_2_advanced(hitmiss_image, skeleton_image)


if __name__ == "__main__":
    main()
