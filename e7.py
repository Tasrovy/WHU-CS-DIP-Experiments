import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import argparse

# 支持中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 基础梯度算子与拉普拉斯算子 =================

def edge_roberts(img):
    Kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    Ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    Gx = convolve2d(img, Kx, mode='same', boundary='symm')
    Gy = convolve2d(img, Ky, mode='same', boundary='symm')

    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return np.clip(G, 0, 255).astype(np.uint8)


def edge_sobel(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    Gx = convolve2d(img, Kx, mode='same', boundary='symm')
    Gy = convolve2d(img, Ky, mode='same', boundary='symm')

    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return np.clip(G, 0, 255).astype(np.uint8), Gx, Gy, G


def edge_prewitt(img):
    Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    Gx = convolve2d(img, Kx, mode='same', boundary='symm')
    Gy = convolve2d(img, Ky, mode='same', boundary='symm')

    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return np.clip(G, 0, 255).astype(np.uint8)


def edge_laplacian(img):
    K = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    G = convolve2d(img, K, mode='same', boundary='symm')

    # 二阶微分会有负数，取绝对值展示零交叉特征
    G = np.abs(G)
    return np.clip(G, 0, 255).astype(np.uint8)


# ================= 3. 手工底层实现 Canny 边缘检测 =================

def non_maximum_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.float32)
    # 将弧度转换为角度 [0, 180)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # 角度量化为四个方向：0度(水平), 45度(对角), 90度(垂直), 135度(反对角)
            # 0 度方向 (左右比较)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = G[i, j + 1]
                r = G[i, j - 1]
            # 45 度方向 (右上、左下比较)
            elif (22.5 <= angle[i, j] < 67.5):
                q = G[i + 1, j - 1]
                r = G[i - 1, j + 1]
            # 90 度方向 (上下比较)
            elif (67.5 <= angle[i, j] < 112.5):
                q = G[i + 1, j]
                r = G[i - 1, j]
            # 135 度方向 (左上、右下比较)
            elif (112.5 <= angle[i, j] < 157.5):
                q = G[i - 1, j - 1]
                r = G[i + 1, j + 1]

            # 抑制非极大值
            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    return Z


def double_threshold_and_hysteresis(img, low_threshold_ratio, high_threshold_ratio):
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)

    weak = 50
    strong = 255

    # 根据阈值进行分类
    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # 边缘连接：将与强边缘相连的弱边缘变为强边缘
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == weak:
                if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong) or
                        (res[i, j - 1] == strong) or (res[i, j + 1] == strong) or
                        (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = 0
    return res


def manual_canny(img, low, high):
    # 3.1 高斯滤波平滑降噪
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

    # 3.2 使用 Sobel 计算梯度幅值与方向
    _, Gx, Gy, G = edge_sobel(blurred)
    theta = np.arctan2(Gy, Gx)

    # 3.3 非极大值抑制 (让边缘变细)
    nms_img = non_maximum_suppression(G, theta)

    # 3.4 双阈值与滞后边界跟踪 (保留真实边缘，去除噪声伪影)
    final_edges = double_threshold_and_hysteresis(nms_img, low_threshold_ratio=low, high_threshold_ratio=high)
    return final_edges


def run_edge_detection_experiment(img):
    if img is None:
        print("Unable to read input image.")
        return

    # 1. 执行四大经典算子
    res_roberts = edge_roberts(img)
    res_sobel, _, _, _ = edge_sobel(img)
    res_prewitt = edge_prewitt(img)
    res_laplacian = edge_laplacian(img)

    # 2. 测试三种阈值的 Canny 算法
    # A. 低阈值 (要求太松：容易保留背景噪声和虚假的弱边缘)
    canny_low = manual_canny(img, low=0.01, high=0.1)
    # B. 适中阈值 (最优解：有效抑制噪声，同时通过 8 邻域接通断裂边缘)
    canny_mid = manual_canny(img, low=0.08, high=0.25)
    # C. 高阈值 (要求太严：剔除了所有噪声，但真实的边缘线条也会发生严重断裂)
    canny_high = manual_canny(img, low=0.3, high=0.6)

    # 3. 画幅展示 (2行4列布局)
    plt.figure(figsize=(18, 9))
    plt.suptitle("实验七：经典边缘算子", fontsize=18, fontweight='bold')

    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("1. 原始灰度图像")
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(res_roberts, cmap='gray')
    plt.title("2. Roberts 算子")
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(res_sobel, cmap='gray')
    plt.title("3. Sobel 算子")
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow(res_prewitt, cmap='gray')
    plt.title("4. Prewitt 算子")
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.imshow(res_laplacian, cmap='gray')
    plt.title("5. Laplacian 算子")
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(canny_low, cmap='gray')
    plt.title("6. Canny (低阈值 0.01, 0.1)")
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(canny_mid, cmap='gray')
    plt.title("7. Canny (适中阈值 0.08, 0.25)")
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(canny_high, cmap='gray')
    plt.title("8. Canny (高阈值 0.3, 0.6)")
    plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

def add_gaussian_noise(img_path, mean=0, sigma=25):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径是否正确: {img_path}")

    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img_path, amount=0.04):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径是否正确: {img_path}")

    noisy_img = img.copy()
    prob = np.random.random(img.shape)
    noisy_img[prob < (amount / 2)] = 0
    noisy_img[prob > 1 - (amount / 2)] = 255
    return noisy_img

def main():
    parser = argparse.ArgumentParser(description="Classic edge detection and manual Canny experiments.")
    parser.add_argument("images", nargs="*", default=["7-1.bmp", "7-2.bmp", "7-3.bmp", "7-4.bmp"],
                        help="input grayscale images, default: 7-1.bmp 7-2.bmp 7-3.bmp 7-4.bmp")
    parser.add_argument("--no-noise", action="store_true", help="only run original images, without noisy variants")
    args = parser.parse_args()

    for image_path in args.images:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to read input image: {image_path}")
            continue
        run_edge_detection_experiment(img)
        if not args.no_noise:
            run_edge_detection_experiment(add_gaussian_noise(image_path))
            run_edge_detection_experiment(add_salt_pepper_noise(image_path))


if __name__ == "__main__":
    main()
