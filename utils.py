import cv2
import numpy as np

# import matplotlib.pyplot as plt
import os

# import rawpy

pic_path = os.path.join(os.path.dirname(__file__), "datas", "input")
out_path = os.path.join(os.path.dirname(__file__), "datas", "output")


def order_points(pts):
    # 初始化一个列表，用于存储排序后的顶点
    rect = np.zeros((4, 2), dtype="int32")

    # 获取左上角和右下角的点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 获取右上角和左下角的点
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def read_cr2_as_rgb(image_path):
    """读取 CR2 并转换为 NumPy 数组"""
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess(output_bps=8)  # 16-bit 以保留更多数据
    return rgb_image


def draw_poly(img, pts):
    return cv2.polylines(
        img,
        [pts.astype(np.int32).reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 0, 255),
        thickness=20,
    )


coords_cache = {}


# 创建坐标矩阵
def get_coords(cols, rows):
    global coords_cache
    if (cols, rows) not in coords_cache:
        coords_cache[(cols, rows)] = np.meshgrid(np.arange(cols), np.arange(rows))
    return coords_cache[(cols, rows)]


def get_black_pixel_count_on_one_side(image, x0, y0, angle):
    """
    计算图像中位于直线 y = kx + b 一侧的黑色像素数量（向量化优化版）

    参数：
    - image: 二值化图像 (0=黑色, 255=白色)
    - x0, y0: 直线上已知的一点 (通常是图像的四个角之一)
    - angle: 直线角度 (以水平向右为 0°，逆时针旋转)

    返回：
    - black_count: 直线 **上方**（y > kx + b） 的黑色像素数量
    """
    global coords_cache

    rows, cols = image.shape
    angle_rad = np.radians(angle)  # 角度转弧度

    # 计算直线斜率 k = tan(θ)
    k = np.tan(angle_rad)

    # 计算直线的截距 b
    b = y0 - k * x0  # 由 y = kx + b 计算

    # 生成像素点坐标矩阵
    x_coords, y_coords = get_coords(cols, rows)

    # 计算直线一侧的像素掩码
    mask = y_coords > (k * x_coords + b)  # 满足 y > kx + b 的像素点

    # 统计黑色像素（值为 0）
    black_count = np.sum((image == 0) & mask)

    return black_count


def intersaction(line1, line2):
    """
    计算两条直线的交点

    参数：
    - line1: (x1, y1, angle1)，直线 1 经过 (x1, y1)，角度 angle1（度）
    - line2: (x2, y2, angle2)，直线 2 经过 (x2, y2)，角度 angle2（度）

    返回：
    - (x, y): 交点坐标 (float, float)
    - None: 若两条直线平行（无交点）
    """
    x1, y1, angle1 = line1
    x2, y2, angle2 = line2

    # 计算斜率 k
    k1 = np.tan(np.radians(angle1))
    k2 = np.tan(np.radians(angle2))

    # 计算截距 b
    b1 = y1 - k1 * x1
    b2 = y2 - k2 * x2

    # 判断是否平行（斜率相等）
    if np.isclose(k1, k2):
        return None  # 平行无交点

    # 计算交点
    x = int((b2 - b1) / (k1 - k2))
    y = int(k1 * x + b1)

    return (x, y)


def detect_quadrilateral_corners(image, step=0.5, angle_range=89.9):
    """
    在图像的四个角旋转扫描，寻找四边形的四个顶点
    """
    rows, cols = image.shape
    corner_positions = [
        (0, 0),  # 左上角
        (cols - 1, 0),  # 右上角
        (cols - 1, rows - 1),  # 右下角
        (0, rows - 1),  # 左下角
    ]

    best_angles = []

    for idx, (x0, y0) in enumerate(corner_positions):
        black_count = []
        if idx % 2 == 0:
            angles = np.arange(0, angle_range, step)
        else:
            angles = np.arange(-angle_range, 0, step)

        # 中间1/3不用算
        angles = angles[len(angles) // 3 : -len(angles) // 3]

        # 计算每个角度的黑白像素比例
        for angle in angles:
            count = get_black_pixel_count_on_one_side(image, x0, y0, angle)
            black_count.append((angle, count))
            # print(angle)

        # 找到比例明显变化的点
        counts = np.array([r[1] for r in black_count])
        diff_counts = np.diff(counts)  # 计算变化率
        # 第一个和最后一个可能会不稳定
        diff_counts[0] = 0
        diff_counts[-1] = 0
        threshold = rows * cols / 6000 / 4000 * 4
        # 找到明显变化的索引
        change_points = np.where(np.abs(diff_counts) > threshold)[0]
        if len(change_points) > 0:
            max_idx = change_points[0]
            min_idx = change_points[-1]
        else:
            first_change_idx = None  # 如果没有明显变化点
        # plt.bar(angles, counts)
        # plt.bar(angles[1:], diff_counts * 10)
        # plt.show()

        # angle小的放前面
        if black_count[max_idx][0] > black_count[min_idx][0]:
            max_idx, min_idx = min_idx, max_idx
        best_angles.append(
            ((x0, y0, black_count[max_idx][0]), (x0, y0, black_count[min_idx][0]))
        )

    # 计算四个交点(tl, tr, br, bl)
    # 按
    detected_corners = [
        intersaction(best_angles[1][1], best_angles[3][0]),
        intersaction(best_angles[0][0], best_angles[2][1]),
        intersaction(best_angles[1][0], best_angles[3][1]),
        intersaction(best_angles[0][1], best_angles[2][0]),
    ]

    return detected_corners


def rotate_img(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    return rotated


def main():
    # 读取二值图像
    input_path = os.path.join(
        pic_path, [p for p in os.listdir(pic_path) if not p.endswith(".CR2")][1]
    )
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 运行四边形检测算法
    corners = detect_quadrilateral_corners(binary)

    # 可视化结果
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)  # 画出角点

    # 显示结果
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
