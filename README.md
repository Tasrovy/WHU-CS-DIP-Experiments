# WHU CS DIP Experiments

武汉大学计算机学院《数字图像处理》课间实验代码。项目包含 8 个独立 Python 脚本，每个脚本都可以直接单文件运行，并支持通过命令行参数指定输入图片。

## 环境要求

- Python 3.9+
- OpenCV
- NumPy
- Matplotlib
- SciPy

安装依赖：

```bash
pip install opencv-python numpy matplotlib scipy
```

如果使用 Conda，也可以创建独立环境：

```bash
conda create -n dip python=3.10
conda activate dip
pip install opencv-python numpy matplotlib scipy
```

## 文件与素材

脚本默认从项目根目录读取实验图片。例如 `e3.py` 默认读取 `3-1.bmp` 和 `3-2.bmp`。请将课程提供的图片素材放在本 README 同级目录下。

当前代码文件：

| 文件 | 实验内容 | 默认输入 |
| --- | --- | --- |
| `e1.py` | RGB 通道分离与灰度图生成 | `1.jpg` |
| `e2.py` | 手写直方图均衡化 | `2-1.bmp`, `2-2.bmp` |
| `e3.py` | 图像缩放、旋转与插值 | `3-1.bmp`, `3-2.bmp` |
| `e4.py` | 噪声添加、滤波去噪、暗场噪声分析 | `4-1.png`, `dark.jpg` |
| `e5.py` | 手写 DFT 与频域滤波 | `5-1.bmp`, `5-2.bmp`, `5-3.png` |
| `e6.py` | JPEG 压缩质量与 RGB/YUV 压缩分析 | `6-1.bmp`, `6-3.bmp`, `6-2.bmp` |
| `e7.py` | Roberts、Sobel、Prewitt、Laplacian、Canny 边缘检测 | `7-1.bmp` 到 `7-4.bmp` |
| `e8.py` | 二值形态学、击中击不中、骨架提取 | `8-1.jpg`, `8-2.png`, `8-4.png`, `8-3.png` |

## 实验简介

`e1.py` 主要完成彩色图像的 RGB 通道分离。程序会分别提取红、绿、蓝三个通道，生成对应的单通道灰度图和彩色通道图，并按加权公式生成普通灰度图。

`e2.py` 实现灰度直方图统计和直方图均衡化。代码手动统计每个灰度级的概率分布，计算累计分布函数，并根据映射关系得到均衡化后的图像。

`e3.py` 比较不同插值算法在图像缩放和旋转中的效果。实验手写实现最近邻、双线性和双三次插值，并展示不同放大倍数、不同旋转角度下的视觉差异和耗时。

`e4.py` 研究图像噪声与空间域滤波。程序会给灰度图添加高斯噪声和椒盐噪声，再使用均值滤波、中值滤波进行去噪对比，并可额外分析暗场图像中的传感器噪声分布。

`e5.py` 实现二维离散傅里叶变换及频域滤波。代码手写 DFT、IDFT 和频谱中心化，并构造巴特沃斯、高斯低通和高通滤波器，用于观察频域平滑与锐化效果。

`e6.py` 分析图像压缩。第一部分比较不同 JPEG 质量因子下的文件大小和图像细节损失；第二部分比较 RGB 与 YUV 色彩空间下的通道压缩、下采样和重建效果。

`e7.py` 进行边缘检测实验。程序实现 Roberts、Sobel、Prewitt、Laplacian 算子，并手写 Canny 的高斯平滑、非极大值抑制、双阈值和边缘连接流程，同时比较噪声对边缘检测的影响。

`e8.py` 实现二值形态学处理。实验包括腐蚀、膨胀、开运算、闭运算、击中击不中变换和骨架提取，用于观察结构元素对二值目标形状分析的影响。

## 使用方法

在项目根目录运行：

```bash
python e1.py 1.jpg
python e2.py
python e3.py
python e4.py
python e5.py
python e6.py
python e7.py
python e8.py
```

每个脚本都支持 `-h` 或 `--help` 查看参数：

```bash
python e3.py --help
```

常用自定义参数示例：

```bash
python e2.py --image1 path/to/2-1.bmp --image2 path/to/2-2.bmp
python e3.py --scale-image path/to/scale.bmp --rotate-image path/to/rotate.bmp
python e4.py --image path/to/4-1.png --skip-dark
python e5.py path/to/5-1.bmp path/to/5-2.bmp path/to/5-3.png
python e6.py --quality-images path/to/6-1.bmp path/to/6-3.bmp --color-image path/to/6-2.bmp
python e7.py path/to/7-1.bmp path/to/7-2.bmp --no-noise
python e8.py --basic-images path/to/8-1.jpg path/to/8-2.png --hitmiss-image path/to/8-4.png --skeleton-image path/to/8-3.png
```

## 输出说明

- 多数实验通过 Matplotlib 或 OpenCV 窗口显示结果。
- `e1.py` 会在输入图片同目录生成通道分离结果图片。
- `e6.py` 会临时生成 JPEG 文件用于比较文件大小，实验结束后自动删除临时文件。
- `e5.py` 和 `e6.py` 在缺少部分默认图片时会生成简单测试图继续演示。
- `e8.py` 在缺少默认图片时会使用代码内置的示例图演示形态学流程。

## 常见问题

如果运行时报 `ImportError: DLL load failed while importing _path`，通常是 Matplotlib 安装或 Python 环境不完整。建议在干净的虚拟环境中重新安装依赖：

```bash
pip uninstall matplotlib -y
pip install matplotlib
```

如果 OpenCV 读取图片失败，请检查：

- 图片是否放在项目根目录；
- 文件名和扩展名是否与默认值一致；
- 或者运行脚本时显式传入图片路径。
