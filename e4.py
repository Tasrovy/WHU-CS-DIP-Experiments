import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, density=0.05):
    noisy_img = img.copy()
    prob = np.random.random(img.shape)
    noisy_img[prob < density / 2] = 0
    noisy_img[prob > 1 - density / 2] = 255
    return noisy_img

def manual_mean_filter(img, kernel_size=3):
    h, w = img.shape
    pad = kernel_size // 2
    padded_img = np.pad(img, pad, mode='reflect')
    dst = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            dst[i, j] = np.mean(window)

    return dst.astype(np.uint8)


def manual_median_filter(img, kernel_size=3):
    h, w = img.shape
    pad = kernel_size // 2
    padded_img = np.pad(img, pad, mode='reflect')
    dst = np.zeros_like(img, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            dst[i, j] = np.median(window)

    return dst

def analyze_sensor_noise(dark_img_path):
    dark_img = cv2.imread(dark_img_path, cv2.IMREAD_GRAYSCALE)
    if dark_img is None:
        print("请检查全黑图片路径")
        return

    total_pixels = dark_img.size

    hist = np.bincount(dark_img.flatten(), minlength=256)

    prob_hist = (hist / total_pixels) * 100

    plt.figure(figsize=(8, 5))
    plt.bar(range(256), prob_hist, color='black', width=0.8)

    plt.xlim([0, 20])

    plt.title("传感器固有噪声分布")
    plt.xlabel("灰度值")
    plt.ylabel("出现频率 (%)")
    plt.grid(axis='y', alpha=0.3)

    mean_val = np.mean(dark_img)
    std_val = np.std(dark_img)
    plt.text(10, max(prob_hist) * 0.8, f"均值: {mean_val:.2f}\n标准差: {std_val:.2f}",
             bbox=dict(facecolor='white', alpha=0.5))

    plt.show()



def run_experiment(image_path="4-1.png", dark_image_path="dark.jpg"):
    # 读取原始灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("未找到 4-1.png")
        return

    # --- 步骤 1: 加噪 ---
    gaussian_img = add_gaussian_noise(img, sigma=30)
    sp_img = add_salt_pepper_noise(img, density=0.1)

    # --- 步骤 2: 滤波去噪 ---
    # 高斯噪声分别用均值和中值处理
    g_mean = manual_mean_filter(gaussian_img, 3)
    g_median = manual_median_filter(gaussian_img, 3)

    # 椒盐噪声分别用均值和中值处理
    sp_mean = manual_mean_filter(sp_img, 3)
    sp_median = manual_median_filter(sp_img, 3)

    # --- 步骤 3: 结果展示 (按行对齐排列) ---

    # 第一排：高斯对比组
    row1_titles = ['原始图像', '高斯噪声 (σ=30)', '高斯+均值滤波 (3x3)', '高斯+中值滤波 (3x3)']
    row1_images = [img, gaussian_img, g_mean, g_median]

    # 第二排：椒盐对比组
    row2_titles = ['原始图像', '椒盐噪声 (d=0.1)', '椒盐+均值滤波 (3x3)', '椒盐+中值滤波 (3x3)']
    row2_images = [img, sp_img, sp_mean, sp_median]

    # 合并标题和图片，用于循环画图
    all_titles = row1_titles + row2_titles
    all_images = row1_images + row2_images

    # 创建 2行4列 的布局
    plt.figure(figsize=(18, 10))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(all_images[i], cmap='gray')
        plt.title(all_titles[i], fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    # --- 步骤 4: 传感器分析 (需要你提供一张全黑照片) ---
    if dark_image_path:
        analyze_sensor_noise(dark_image_path)


def main():
    parser = argparse.ArgumentParser(description="Add noise, denoise, and optionally analyze dark-frame noise.")
    parser.add_argument("--image", default="4-1.png", help="input grayscale image, default: 4-1.png")
    parser.add_argument("--dark-image", default="dark.jpg", help="dark-frame image path, default: dark.jpg")
    parser.add_argument("--skip-dark", action="store_true", help="skip dark-frame noise analysis")
    args = parser.parse_args()
    run_experiment(args.image, None if args.skip_dark else args.dark_image)


if __name__ == "__main__":
    main()
