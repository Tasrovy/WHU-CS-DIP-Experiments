import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def bicubic_weight(x, a=-0.5):
    x = np.abs(x)
    w = np.zeros_like(x)

    mask1 = x < 1
    w[mask1] = (a + 2) * (x[mask1] ** 3) - (a + 3) * (x[mask1] ** 2) + 1

    mask2 = (x >= 1) & (x < 2)
    w[mask2] = a * (x[mask2] ** 3) - 5 * a * (x[mask2] ** 2) + 8 * a * x[mask2] - 4 * a

    return w


def apply_interpolation(img, map_x, map_y, method):
    H, W, C = img.shape

    valid_mask = (map_x >= 0) & (map_x < W - 1) & (map_y >= 0) & (map_y < H - 1)

    map_x = np.clip(map_x, 0, W - 1)
    map_y = np.clip(map_y, 0, H - 1)

    if method == 'nearest':
        src_x = np.round(map_x).astype(int)
        src_y = np.round(map_y).astype(int)
        src_x = np.clip(src_x, 0, W - 1)
        src_y = np.clip(src_y, 0, H - 1)

        out_img = img[src_y, src_x]

    elif method == 'bilinear':
        x0 = np.floor(map_x).astype(int)
        y0 = np.floor(map_y).astype(int)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)

        alpha = map_x - x0
        beta = map_y - y0

        alpha = np.expand_dims(alpha, axis=-1)
        beta = np.expand_dims(beta, axis=-1)

        Q11 = img[y0, x0].astype(float)
        Q21 = img[y0, x1].astype(float)
        Q12 = img[y1, x0].astype(float)
        Q22 = img[y1, x1].astype(float)

        out_img = (1 - alpha) * (1 - beta) * Q11 + \
                  alpha * (1 - beta) * Q21 + \
                  (1 - alpha) * beta * Q12 + \
                  alpha * beta * Q22

    elif method == 'bicubic':
        x0 = np.floor(map_x).astype(int)
        y0 = np.floor(map_y).astype(int)

        u = map_x - x0
        v = map_y - y0

        out_img = np.zeros((map_y.shape[0], map_x.shape[1], C), dtype=float)

        for i in range(-1, 3):
            for j in range(-1, 3):
                xi = np.clip(x0 + i, 0, W - 1)
                yj = np.clip(y0 + j, 0, H - 1)

                wx = bicubic_weight(i - u)
                wy = bicubic_weight(j - v)

                weight = np.expand_dims(wx * wy, axis=-1)

                out_img += img[yj, xi].astype(float) * weight

    else:
        raise ValueError("不支持的插值方法")
    out_img[~valid_mask] = 0
    return np.clip(out_img, 0, 255).astype(np.uint8)


def custom_resize(img, scale, method):
    H, W = img.shape[:2]
    dst_h, dst_w = int(H * scale), int(W * scale)

    X_dst, Y_dst = np.meshgrid(np.arange(dst_w), np.arange(dst_h))

    map_x = X_dst / scale
    map_y = Y_dst / scale

    return apply_interpolation(img, map_x, map_y, method)


def custom_rotate(img, angle_deg, method):
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    max_dim = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))

    new_w = max_dim
    new_h = max_dim
    new_cx, new_cy = new_w / 2.0, new_h / 2.0

    X_dst, Y_dst = np.meshgrid(np.arange(new_w), np.arange(new_h))

    X_c = X_dst - new_cx
    Y_c = Y_dst - new_cy

    map_x = X_c * cos_a + Y_c * sin_a + cx
    map_y = -X_c * sin_a + Y_c * cos_a + cy

    return apply_interpolation(img, map_x, map_y, method)


def experiment_scaling(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scales = [2, 4, 8]
    methods = ['nearest', 'bilinear', 'bicubic']
    titles = ['最近邻', '双线性', '双三次']

    fig, axes = plt.subplots(len(scales), len(methods), figsize=(12, 12))
    fig.suptitle('不同放大倍数与手写插值算法对比', fontsize=16)

    base_crop_size = 50
    max_canvas_size = base_crop_size * max(scales)

    for i, scale in enumerate(scales):
        for j, method in enumerate(methods):
            resized_img = custom_resize(img, scale, method)

            cur_crop_size = base_crop_size * scale
            h, w = resized_img.shape[:2]
            cy, cx = h // 2, w // 2
            half = cur_crop_size // 2

            crop = resized_img[cy - half: cy + half, cx - half: cx + half]

            canvas = np.full((max_canvas_size, max_canvas_size, 3), 220, dtype=np.uint8)
            c_cy, c_cx = max_canvas_size // 2, max_canvas_size // 2
            canvas[c_cy - half: c_cy + half, c_cx - half: c_cx + half] = crop

            axes[i, j].imshow(canvas)
            axes[i, j].set_title(f"{titles[j]} - 放大 {scale} 倍")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def experiment_rotation(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    angles = [30, 45, 90, 77]
    methods = ['nearest', 'bilinear', 'bicubic']
    titles = ['最近邻', '双线性', '双三次']

    fig, axes = plt.subplots(len(angles), len(methods), figsize=(12, 12))
    fig.suptitle('图像旋转与不同插值算法效果及耗时对比', fontsize=16)

    num_tests = 3

    for i, angle in enumerate(angles):
        for j, method in enumerate(methods):

            # 测试运行耗时
            start_time = time.perf_counter()
            for _ in range(num_tests):
                rotated_img = custom_rotate(img, angle, method)
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / num_tests * 1000


            axes[i, j].imshow(rotated_img)
            axes[i, j].set_title(f"{titles[j]}\n旋转 {angle}° ({avg_time:.1f}ms)")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Image scaling and rotation with manual interpolation.")
    parser.add_argument("--scale-image", default="3-1.bmp", help="image used for scaling, default: 3-1.bmp")
    parser.add_argument("--rotate-image", default="3-2.bmp", help="image used for rotation, default: 3-2.bmp")
    args = parser.parse_args()
    experiment_scaling(args.scale_image)
    experiment_rotation(args.rotate_image)


if __name__ == "__main__":
    main()
