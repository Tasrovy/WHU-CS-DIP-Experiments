import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def rgb_to_yuv(rgb_img):
    img_f = rgb_img.astype(np.float32)
    R, G, B = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.147 * R - 0.289 * G + 0.436 * B
    V = 0.615 * R - 0.515 * G - 0.100 * B

    return np.stack([Y, U, V], axis=-1)


def yuv_to_rgb(yuv_img):
    Y, U, V = yuv_img[:, :, 0], yuv_img[:, :, 1], yuv_img[:, :, 2]

    R = Y + 1.140 * V
    G = Y - 0.395 * U - 0.581 * V
    B = Y + 2.032 * U

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def get_file_size(filepath):
    return os.path.getsize(filepath) / 1024.0


# ================= 实验一：JPEG 压缩质量分析 =================

def experiment_1_jpeg_quality(img_path):
    # 1. 读取彩色图像
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(0, 300, 20):
            cv2.line(img, (i, 0), (i, 300), (255, 255, 255), 2)
            cv2.line(img, (0, i), (300, i), (255, 255, 255), 2)
        cv2.putText(img, "JPEG", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    qualities = [95, 80, 50, 10]
    results = []

    for q in qualities:
        tmp_file = f"temp_q{q}.jpg"
        cv2.imwrite(tmp_file, img, [cv2.IMWRITE_JPEG_QUALITY, q])
        size_kb = get_file_size(tmp_file)

        img_compressed = cv2.imread(tmp_file)
        img_compressed = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        cy, cx = h // 2, w // 2
        crop_size = 50
        detail_crop = img_compressed[cy - crop_size: cy + crop_size, cx - crop_size: cx + crop_size]

        results.append((q, size_kb, img_compressed, detail_crop))
        os.remove(tmp_file)

    plt.figure(figsize=(16, 8))
    plt.suptitle("实验六 (一)", fontsize=16, fontweight='bold')

    for i, (q, size, full_img, crop_img) in enumerate(results):
        # 第一排：全图与文件大小
        plt.subplot(2, 4, i + 1)
        plt.imshow(full_img)
        plt.title(f"质量 Q={q} | 大小: {size:.1f} KB", fontsize=12)
        plt.axis('off')

        # 第二排：高频细节放大
        plt.subplot(2, 4, i + 5)
        plt.imshow(crop_img)
        plt.title(f"高频细节局部放大", fontsize=11, color='black')
        plt.axis('off')

    plt.tight_layout()


# ================= 实验二：色彩空间压缩分析 =================

def experiment_2_color_space(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"未找到 {img_path}，已生成彩色测试图2...")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, :100] = [0, 0, 255]  # 红色
        img[:, 100:200] = [0, 255, 0]  # 绿色
        img[:, 200:] = [255, 0, 0]  # 蓝色
        cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    q_test = 15

    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    yuv_img = rgb_to_yuv(img_rgb)
    Y, U, V = yuv_img[:, :, 0], yuv_img[:, :, 1], yuv_img[:, :, 2]

    U_save, V_save = U + 128, V + 128

    channels = {'R': R, 'G': G, 'B': B, 'Y': Y, 'U': U_save, 'V': V_save}
    comp_channels = {}

    for name, channel in channels.items():
        tmp_name = f"temp_{name}.jpg"
        cv2.imwrite(tmp_name, channel.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q_test])
        comp_channels[name] = cv2.imread(tmp_name, 0).astype(np.float32)
        os.remove(tmp_name)

    comp_channels['U'] -= 128
    comp_channels['V'] -= 128

    # --- 1. RGB 压缩后直接重组 ---
    rgb_comp_reconstructed = np.stack([comp_channels['R'], comp_channels['G'], comp_channels['B']], axis=-1)
    rgb_comp_reconstructed = np.clip(rgb_comp_reconstructed, 0, 255).astype(np.uint8)

    # --- 2. YUV 压缩下采样后重组 ---
    h, w = Y.shape
    U_down = cv2.resize(U, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    V_down = cv2.resize(V, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

    U_up = cv2.resize(U_down, (w, h), interpolation=cv2.INTER_LINEAR)
    V_up = cv2.resize(V_down, (w, h), interpolation=cv2.INTER_LINEAR)

    yuv_reconstructed = np.stack([Y, U_up, V_up], axis=-1)
    rgb_reconstructed = yuv_to_rgb(yuv_reconstructed)

    # ================= 2行5列可视化布局 =================
    plt.figure(figsize=(18, 9))
    plt.suptitle(f"实验六 (二) (压缩质量 Q={q_test})", fontsize=16, fontweight='bold')

    # 第一排：RGB 三通道压缩与重组
    plt.subplot(2, 5, 1)
    plt.imshow(img_rgb)
    plt.title("原始彩色图像")
    plt.axis('off')

    plt.subplot(2, 5, 2)
    plt.imshow(comp_channels['R'], cmap='gray')
    plt.title(f"R 通道压缩图 (Q={q_test})")
    plt.axis('off')

    plt.subplot(2, 5, 3)
    plt.imshow(comp_channels['G'], cmap='gray')
    plt.title(f"G 通道压缩图 (Q={q_test})")
    plt.axis('off')

    plt.subplot(2, 5, 4)
    plt.imshow(comp_channels['B'], cmap='gray')
    plt.title(f"B 通道压缩图 (Q={q_test})")
    plt.axis('off')

    plt.subplot(2, 5, 5)
    plt.imshow(rgb_comp_reconstructed)
    plt.title("RGB 压缩后重构图", color='black')
    plt.axis('off')

    # 第二排：YUV 三通道压缩与重组
    plt.subplot(2, 5, 6)
    plt.imshow(img_rgb)
    plt.title("原始彩色图像")
    plt.axis('off')

    plt.subplot(2, 5, 7)
    plt.imshow(comp_channels['Y'], cmap='gray')
    plt.title(f"Y 亮度通道压缩图 (Q={q_test})")
    plt.axis('off')

    plt.subplot(2, 5, 8)
    plt.imshow(comp_channels['U'] + 128, cmap='gray')
    plt.title(f"U 色度通道压缩图 (Q={q_test})")
    plt.axis('off')

    plt.subplot(2, 5, 9)
    plt.imshow(comp_channels['V'] + 128, cmap='gray')
    plt.title(f"V 色度通道压缩图 (Q={q_test})")
    plt.axis('off')

    plt.subplot(2, 5, 10)
    plt.imshow(rgb_reconstructed)
    plt.title("YUV 压缩及下采样重构图", color='black')
    plt.axis('off')

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="JPEG quality and color-space compression experiments.")
    parser.add_argument("--quality-images", nargs="*", default=["6-1.bmp", "6-3.bmp"],
                        help="images for JPEG quality experiment, default: 6-1.bmp 6-3.bmp")
    parser.add_argument("--color-image", default="6-2.bmp",
                        help="image for RGB/YUV color-space experiment, default: 6-2.bmp")
    args = parser.parse_args()
    for image_path in args.quality_images:
        experiment_1_jpeg_quality(image_path)
    experiment_2_color_space(args.color_image)
    plt.show()


if __name__ == "__main__":
    main()
