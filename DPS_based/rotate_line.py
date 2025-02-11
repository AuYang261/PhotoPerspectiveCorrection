# 基于旋转直线的方法找到四个角点，而后透视变换。
# 具体来说，分别以图像的四个角为原点，以一定步长旋转直线，每次旋转后计算在直线一侧的黑色像素数量。
# 得到黑色像素数与旋转角度的关系，当黑色像素数开始从0明显增加或减少到接近0时，认为直线旋转到了角点。
# 每个原点都可以得到两条直线，最后得到8条直线，由此确定四个角点。
# 由于是检测四个角点得到任意四边形，因此可以实现透视矫正。
# 要求二值化时非文档区域不要有较多黑色像素，故有时效果不是很理想。

import cv2
import numpy as np
import os
import utils
import multiprocessing


pic_path = utils.pic_path
out_path = utils.out_path


def four_point_transform(image, rect):
    """透视变换矫正文档"""
    rect = np.array(rect, dtype="float32")
    (tl, tr, br, bl) = rect

    # cv2.imshow("img", utils.draw_poly(image, rect))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 计算新图像的宽度
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图像的高度
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # 变换目标点
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def perspective_correction(image_path, output_path):
    """读取文档图片，检测边缘并矫正"""
    print(f"正在处理: {image_path}")

    # 读取图像并转换为灰度
    image = utils.read_cr2_as_rgb(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    # cv2.imshow("blurred", blurred)
    ret, binary = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)

    corners = utils.detect_quadrilateral_corners(binary)

    # 可视化结果
    # for x, y in corners:
    #     cv2.circle(image, (x, y), 50, (0, 0, 255), -1)  # 画出角点

    # cv2.imshow("corners", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 透视变换矫正
    warped = four_point_transform(image, corners)

    # 保存结果
    cv2.imwrite(output_path, warped)
    print(f"处理完成，矫正后图像保存在: {output_path}")


def main():
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for pic in os.listdir(pic_path):
        if "CR2" not in pic:
            continue
        pool.apply_async(
            perspective_correction,
            args=(
                os.path.join(pic_path, pic),
                os.path.join(out_path, pic.replace("CR2", "JPG")),
            ),
        )
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
