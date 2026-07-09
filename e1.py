import cv2
import numpy as np
import os
import argparse

def split_rgb_channels(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    # 2. 拆分通道
    # b, g, r 都是单通道的灰度图，亮度表示该颜色在原图中的强度
    b, g, r = cv2.split(img)
    grey = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    # 3. 保存单通道的灰度图
    cv2.imwrite(f'{image_path}_channel_blue_gray.jpg', b)
    cv2.imwrite(f'{image_path}_channel_green_gray.jpg', g)
    cv2.imwrite(f'{image_path}_channel_red_gray.jpg', r)
    cv2.imwrite(f'{image_path}_channel_grey.jpg', grey)

    zeros = np.zeros(img.shape[:2], dtype="uint8")

    # 合成纯蓝色图 (B, 0, 0)
    blue_colored = cv2.merge([b, zeros, zeros])
    # 合成纯绿色图 (0, G, 0)
    green_colored = cv2.merge([zeros, g, zeros])
    # 合成纯红色图 (0, 0, R)
    red_colored = cv2.merge([zeros, zeros, r])

    # 4. 保存彩色化的通道图
    cv2.imwrite(f'{image_path}_result_B.jpg', blue_colored)
    cv2.imwrite(f'{image_path}_result_G.jpg', green_colored)
    cv2.imwrite(f'{image_path}_result_R.jpg', red_colored)

    # 5. 显示结果
    cv2.imshow('Original', img)
    cv2.imshow('Blue', b)
    cv2.imshow('Green', g)
    cv2.imshow('Red', r)
    cv2.imshow('Grey', grey)
    cv2.imshow('Red Channel', red_colored)
    cv2.imshow('Green Channel', green_colored)
    cv2.imshow('Blue Channel', blue_colored)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Split one image into RGB channels and grayscale results.")
    parser.add_argument("image", nargs="?", default="1.jpg", help="input image path, default: 1.jpg")
    args = parser.parse_args()
    split_rgb_channels(args.image)


if __name__ == "__main__":
    main()
