import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import requests
import io
import zipfile

def get_pavia_dataset():
    """
    下载并加载 Pavia University 数据集 (.mat 格式)
    如果网络不好，请手动下载 'PaviaU.mat' 并放在同目录下
    """
    try:
        # 尝试加载本地文件
        data = loadmat('PaviaU.mat')['paviaU']
        print("已加载本地数据集。")
    except FileNotFoundError:
        print("正在下载 Pavia University 数据集...")
        url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
        r = requests.get(url)
        with open("PaviaU.mat", "wb") as f:
            f.write(r.content)
        data = loadmat('PaviaU.mat')['paviaU']
        print("下载完成。")
    return data


def plot_hyperspectral_cube(data, rgb_bands=(50, 30, 10)):
    # 1. 数据归一化 (0-1) 用于显示
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    H, W, B = data_norm.shape
    # 2. 提取三个面
    # 正面 (Front): 空间图像 (RGB)
    # 选取三个波段合成假彩色 RGB
    front_face = data_norm[:, :, rgb_bands]
    # 顶面 (Top): 空间-光谱切面 (Width x Bands)
    # 取第一行 (或某几行的平均) 的光谱
    # 为了颜色好看，通常使用 'jet' 或 'nipy_spectral' colormap 映射光谱值
    top_face_data = data_norm[0, :, :].T  # 转置后变成 Bands x Width
    # 侧面 (Side): 空间-光谱切面 (Height x Bands)
    # 取最后一列
    side_face_data = data_norm[:, -1, :]  # Height x Bands
    return create_sheared_cube_cv2(front_face, top_face_data, side_face_data)

def create_sheared_cube_cv2(front_img, top_data, side_data):
    """
    使用 OpenCV 进行透视/错切变换，这是最稳健的方法
    """
    import cv2
    # 转换数据格式为 uint8 (0-255)
    def to_uint8(img, cmap=None):
        if cmap is not None:
            # 应用 colormap (例如 jet)
            import matplotlib.cm as cm
            # img 归一化
            img = (img - img.min()) / (img.max() - img.min())
            colored = cmap(img)[:, :, :3]  # 去掉 alpha
            return (colored * 255).astype(np.uint8)
        else:
            return (img * 255).astype(np.uint8)

    front = to_uint8(front_img)

    # 生成侧面和顶面的伪彩色图
    import matplotlib.cm as cm
    # cmap = cm.nipy_spectral
    cmap = cm.jet
    top = to_uint8(top_data, cmap)  # Bands x Width
    side = to_uint8(side_data, cmap)  # Height x Bands

    h, w, _ = front.shape
    bands_h, bands_w, _ = top.shape  # bands_h 是波段数
    # 设置深度（为了视觉效果，把波段数拉伸一下）
    depth_size = int(w * 0.2)
    # 调整尺寸
    top = cv2.resize(top, (w, depth_size))
    side = cv2.resize(side, (depth_size, h))
    # 创建画布
    canvas_h = h + depth_size
    canvas_w = w + depth_size
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    # 1. 放置正面 (Front) - 左下
    # 坐标: y从 depth_size 到 end, x从 0 到 w
    canvas[depth_size:, :w] = front
    # 2. 变换并放置顶面 (Top)
    # 顶面是一个矩形，需要变成平行四边形
    # 原始坐标: (0,0), (w,0), (0, d), (w, d)
    # 目标坐标: (depth_size, 0), (canvas_w, 0), (0, depth_size), (w, depth_size)
    pts1 = np.float32([[0, 0], [w, 0], [0, depth_size]])
    pts2 = np.float32([[depth_size, 0], [canvas_w, 0], [0, depth_size]])
    M_top = cv2.getAffineTransform(pts1, pts2)
    top_warped = cv2.warpAffine(top, M_top, (canvas_w, depth_size))
    # 遮罩融合
    mask_top = top_warped > 0
    canvas[:depth_size, :] = np.where(mask_top[:depth_size, :], top_warped[:depth_size, :], canvas[:depth_size, :])
    # 3. 变换并放置侧面 (Side)
    # 原始坐标: (0,0), (d,0), (0,h)
    # 目标坐标: (w, depth_size), (canvas_w, 0), (w, canvas_h)
    pts1_side = np.float32([[0, 0], [depth_size, 0], [0, h]])
    pts2_side = np.float32([[w, depth_size], [canvas_w, 0], [w, canvas_h]])
    M_side = cv2.getAffineTransform(pts1_side, pts2_side)
    side_warped = cv2.warpAffine(side, M_side, (canvas_w, canvas_h))
    # 侧面主要在右侧区域
    mask_side = side_warped > 0
    canvas[:, w:] = np.where(mask_side[:, w:], side_warped[:, w:], canvas[:, w:])
    # 显示
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.axis('off')
    # plt.title("Hyperspectral Data Cube Visualization")
    # bbox_inches='tight' 和 pad_inches=0 可以去除因为隐藏坐标轴后留下的多余白边
    # dpi=300 用于设置图片的分辨率（清晰度）
    plt.savefig("output_image.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


if __name__ == "__main__":
    # 1. 获取数据
    hsi_data = get_pavia_dataset()
    print(f"数据形状: {hsi_data.shape}")
    # 2. 绘图
    # PaviaU 的 RGB 波段大约是 (50, 30, 10) 或附近
    plot_hyperspectral_cube(hsi_data, rgb_bands=(55, 30, 5))
    # plot_hyperspectral_cube(hsi_data, rgb_bands=(50, 30, 10))