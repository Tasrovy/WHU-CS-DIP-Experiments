import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse


def calculate_custom_histogram(image1_path="2-1.bmp", image2_path="2-2.bmp"):
    # ==================== 处理图片 2-1.bmp ====================
    img1 = cv2.imread(image1_path)
    if img1 is None:
        print("无法读取图片 2-1")
        return

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    # 3.1 统计原图直方图
    hist1 = [0] * 256
    height1, width1 = gray1.shape
    for i in range(height1):
        for j in range(width1):
            pixel_value = gray1[i, j]
            hist1[pixel_value] += 1

    # 计算原图概率密度 PDF
    total_pixels1 = height1 * width1
    pdf1 = [n / total_pixels1 for n in hist1]

    # 3.2 计算累积分布 CDF
    cdf1 = [0.0] * 256
    sum_p1 = 0
    for i in range(256):
        sum_p1 += pdf1[i]
        cdf1[i] = sum_p1

    # 构建映射关系 f(k) = round(s_k * 255)
    mapping1 = [int(s * 255 + 0.5) for s in cdf1]

    # 对原图像进行像素值变换
    gray1_eq = np.zeros((height1, width1), dtype=np.uint8)
    for i in range(height1):
        for j in range(width1):
            gray1_eq[i, j] = mapping1[gray1[i, j]]

    # 统计均衡化后的直方图并计算其概率分布 PDF
    hist1_eq = [0] * 256
    for i in range(height1):
        for j in range(width1):
            hist1_eq[gray1_eq[i, j]] += 1
    pdf1_eq = [n / total_pixels1 for n in hist1_eq]

    # ==================== 处理图片 2-2.bmp ====================
    img2 = cv2.imread(image2_path)
    if img2 is None:
        print("无法读取图片 2-2")
        return

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # 3.1 统计原图直方图
    hist2 = [0] * 256
    height2, width2 = gray2.shape
    for i in range(height2):
        for j in range(width2):
            pixel_value = gray2[i, j]
            hist2[pixel_value] += 1

    # 计算原图概率密度 PDF
    total_pixels2 = height2 * width2
    pdf2 = [n / total_pixels2 for n in hist2]

    # 3.2 计算累积分布 CDF
    cdf2 = [0.0] * 256
    sum_p2 = 0
    for i in range(256):
        sum_p2 += pdf2[i]
        cdf2[i] = sum_p2

    # 构建映射关系
    mapping2 = [int(s * 255 + 0.5) for s in cdf2]

    # 像素变换
    gray2_eq = np.zeros((height2, width2), dtype=np.uint8)
    for i in range(height2):
        for j in range(width2):
            gray2_eq[i, j] = mapping2[gray2[i, j]]

    # 统计均衡化后的直方图并计算其概率分布 PDF
    hist2_eq = [0] * 256
    for i in range(height2):
        for j in range(width2):
            hist2_eq[gray2_eq[i, j]] += 1
    pdf2_eq = [n / total_pixels2 for n in hist2_eq]

    # ==================== 可视化结果 ====================
    plt.figure(figsize=(14, 16))

    # --- 图片 2-1 部分 ---
    plt.subplot(4, 2, 1)
    plt.imshow(gray1, cmap='gray')
    plt.title('Original Image 2-1')
    plt.axis('off')

    plt.subplot(4, 2, 2)
    plt.imshow(gray1_eq, cmap='gray')
    plt.title('Equalized Image 2-1')
    plt.axis('off')

    plt.subplot(4, 2, 3)
    plt.bar(range(256), pdf1, color='gray', width=1.0)
    plt.title('Original PDF 2-1')
    plt.ylabel('Probability')

    plt.subplot(4, 2, 4)
    plt.bar(range(256), pdf1_eq, color='gray', width=1.0)
    plt.title('Equalized PDF 2-1')
    plt.ylabel('Probability')

    # --- 图片 2-2 部分 ---
    plt.subplot(4, 2, 5)
    plt.imshow(gray2, cmap='gray')
    plt.title('Original Image 2-2')
    plt.axis('off')

    plt.subplot(4, 2, 6)
    plt.imshow(gray2_eq, cmap='gray')
    plt.title('Equalized Image 2-2')
    plt.axis('off')

    plt.subplot(4, 2, 7)
    plt.bar(range(256), pdf2, color='gray', width=1.0)
    plt.title('Original PDF 2-2')
    plt.ylabel('Probability')

    plt.subplot(4, 2, 8)
    plt.bar(range(256), pdf2_eq, color='gray', width=1.0)
    plt.title('Equalized PDF 2-2')
    plt.ylabel('Probability')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Manual histogram equalization for two images.")
    parser.add_argument("--image1", default="2-1.bmp", help="first input image, default: 2-1.bmp")
    parser.add_argument("--image2", default="2-2.bmp", help="second input image, default: 2-2.bmp")
    args = parser.parse_args()
    calculate_custom_histogram(args.image1, args.image2)


if __name__ == "__main__":
    main()
