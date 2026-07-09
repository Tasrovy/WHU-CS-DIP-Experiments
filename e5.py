import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 手动离散傅里叶变换与中心化算法 =================

def manual_dft_2d(f):
    M, N = f.shape
    u = np.arange(M).reshape((M, 1))
    x = np.arange(M).reshape((1, M))
    Wm = np.exp(-2j * np.pi * u * x / M)

    v = np.arange(N).reshape((N, 1))
    y = np.arange(N).reshape((1, N))
    Wn = np.exp(-2j * np.pi * v * y / N)

    return Wm @ f @ Wn


def manual_idft_2d(F):
    M, N = F.shape
    u = np.arange(M).reshape((M, 1))
    x = np.arange(M).reshape((1, M))
    Wm_conj = np.exp(2j * np.pi * u * x / M)

    v = np.arange(N).reshape((N, 1))
    y = np.arange(N).reshape((1, N))
    Wn_conj = np.exp(2j * np.pi * v * y / N)

    return (Wm_conj @ F @ Wn_conj) / (M * N)


def manual_fftshift(F):
    M, N = F.shape
    cy, cx = M // 2, N // 2
    shifted = np.empty_like(F)
    shifted[0:cy, 0:cx] = F[cy:M, cx:N]  # 右下 -> 左上
    shifted[cy:M, cx:N] = F[0:cy, 0:cx]  # 左上 -> 右下
    shifted[0:cy, cx:N] = F[cy:M, 0:cx]  # 左下 -> 右上
    shifted[cy:M, 0:cx] = F[0:cy, cx:N]  # 右上 -> 左下
    return shifted


def manual_ifftshift(F):
    return manual_fftshift(F)


# ================= 2. 手动计算距离矩阵与构建滤波器 =================

def get_distance_matrix(M, N):
    u = np.arange(M) - M / 2
    v = np.arange(N) - N / 2
    U, V = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)


def butterworth_lowpass(M, N, D0, n=2):
    D = get_distance_matrix(M, N)
    return 1.0 / (1.0 + (D / D0) ** (2 * n))


def butterworth_highpass(M, N, D0, n=2):
    D = get_distance_matrix(M, N)
    D_safe = np.where(D == 0, 1e-5, D)  # 防止除以0
    return 1.0 / (1.0 + (D0 / D_safe) ** (2 * n))


def gaussian_lowpass(M, N, D0):
    D = get_distance_matrix(M, N)
    return np.exp(-(D ** 2) / (2 * D0 ** 2))


def gaussian_highpass(M, N, D0):
    return 1.0 - gaussian_lowpass(M, N, D0)


# ================= 3. 核心实验与可视化展示 =================

def run_frequency_experiment(image_path):
    # 读取原始灰度图像
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        src = np.zeros((128, 128), dtype=np.uint8)
        src[32:96, 32:96] = 255
    else:
        src = cv2.resize(src, (128, 128))

    M, N = src.shape
    D0 = 15  # 滤波器截止频率
    b_order = 2  # 巴特沃斯滤波器阶数

    # ---------------- 步骤 1: 傅里叶变换与中心化 ----------------
    F_original = manual_dft_2d(src)  # 1.1 二维 DFT
    F_shifted = manual_fftshift(F_original)  # 1.2 对频谱进行中心化

    # 计算对数幅度谱：log(1 + |F(u,v)|)
    spectrum_uncentered = np.log(1 + np.abs(F_original))
    spectrum_centered = np.log(1 + np.abs(F_shifted))

    # ---------------- 步骤 2: 设计低通和高通滤波器模板 ----------------
    H_blpf = butterworth_lowpass(M, N, D0, n=b_order)
    H_glpf = gaussian_lowpass(M, N, D0)
    H_bhpf = butterworth_highpass(M, N, D0, n=b_order)
    H_ghpf = gaussian_highpass(M, N, D0)

    # ---------------- 步骤 3: 频域相乘与反变换 (去噪/平滑/锐化) ----------------
    # Guv = Huv * Fuv (逐点相乘)
    G_blpf = H_blpf * F_shifted
    G_glpf = H_glpf * F_shifted
    G_bhpf = H_bhpf * F_shifted
    G_ghpf = H_ghpf * F_shifted

    # 进行傅里叶反变换还原图像：反中心化 -> IDFT -> 取实部并溢出截断
    img_blpf = np.clip(np.real(manual_idft_2d(manual_ifftshift(G_blpf))), 0, 255).astype(np.uint8)
    img_glpf = np.clip(np.real(manual_idft_2d(manual_ifftshift(G_glpf))), 0, 255).astype(np.uint8)
    img_bhpf = np.clip(np.real(manual_idft_2d(manual_ifftshift(G_bhpf))), 0, 255).astype(np.uint8)
    img_ghpf = np.clip(np.real(manual_idft_2d(manual_ifftshift(G_ghpf))), 0, 255).astype(np.uint8)

    # ==================== 四、十一合一单画幅展示 ====================
    plt.figure(figsize=(16, 12))

    # ------ 第一排：空间域转频域分析 ------
    plt.subplot(3, 4, 1)
    plt.imshow(src, cmap='gray')
    plt.title("1. 原始灰度图像", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(spectrum_uncentered, cmap='gray')
    plt.title("2. 离散变换幅度谱", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(spectrum_centered, cmap='gray')
    plt.title("3. 频域中心化对数谱", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.axis('off')
    # ------ 第二排：低通滤波 (平滑) ------
    plt.subplot(3, 4, 5)
    plt.imshow(H_blpf, cmap='gray')
    plt.title("4. 巴特沃斯低通模板", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(img_blpf, cmap='gray')
    plt.title("5. 巴特沃斯低通平滑", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(H_glpf, cmap='gray')
    plt.title("6. 高斯低通模板", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(img_glpf, cmap='gray')
    plt.title("7. 高斯低通平滑", fontsize=11)
    plt.axis('off')

    # ------ 第三排：高通滤波 (锐化) ------
    plt.subplot(3, 4, 9)
    plt.imshow(H_bhpf, cmap='gray')
    plt.title("8. 巴特沃斯高通模板", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(img_bhpf, cmap='gray')
    plt.title("9. 巴特沃斯高通锐化", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(H_ghpf, cmap='gray')
    plt.title("10. 高斯高通模板", fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 12)
    plt.imshow(img_ghpf, cmap='gray')
    plt.title("11. 高斯高通锐化", fontsize=11)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Manual DFT frequency-domain filtering experiments.")
    parser.add_argument("images", nargs="*", default=["5-1.bmp", "5-2.bmp", "5-3.png"],
                        help="input images, default: 5-1.bmp 5-2.bmp 5-3.png")
    args = parser.parse_args()
    for image_path in args.images:
        run_frequency_experiment(image_path)


if __name__ == "__main__":
    main()
