# -*- coding: utf-8 -*-
#
#  基于opencv的cornerHarris角点检测

import cv2
import numpy as np
import os
import multiprocessing

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

pic_path = utils.pic_path
out_path = utils.out_path
multiprocess = True


def preprocess(input_dir):
    if input_dir.lower().endswith(".cr2"):
        original_img = utils.read_cr2_as_rgb(input_dir)
    else:
        original_img = cv2.imread(input_dir)
    scaled_img = cv2.convertScaleAbs(original_img, alpha=1.2, beta=0)
    gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    # 闭运算（链接块）
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=10)
    # 开运算（去噪点）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=10)
    return original_img, scaled_img, gray_img, closed, opened


def cornerHarris_img(original_img, img):
    # 检测角点
    dst = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
    # 膨胀角点
    dst = cv2.dilate(dst, None, iterations=10)
    cv2.imshow("dst", np.uint8(dst > 0.01 * dst.max()) * 255)
    # 获取最角落的点坐标
    box = []
    box = np.array(box)
    # 标记角点
    draw_img = original_img.copy()
    draw_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return box, draw_img


def goodFeaturesToTrack_img(original_img, img):
    corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 1000)
    h, w = img.shape[:2]
    # 选出最角落的点坐标
    box = [
        sorted(corners, key=lambda x: np.linalg.norm(x - (0, 0)))[0][0],
        sorted(corners, key=lambda x: np.linalg.norm(x - (w, 0)))[0][0],
        sorted(corners, key=lambda x: np.linalg.norm(x - (w, h)))[0][0],
        sorted(corners, key=lambda x: np.linalg.norm(x - (0, h)))[0][0],
    ]
    box = np.array(box, dtype=np.int32)
    draw_img = original_img.copy()
    for corner in box:
        x, y = corner.ravel()
        cv2.circle(draw_img, (x, y), 30, 255, -1)
    return box, draw_img


def perspective_correction(image_path, output_path, q):
    """读取文档图片，检测角点并矫正"""
    print(f"正在处理: {image_path}")

    original_img, scaled_img, gray_img, closed, opened = preprocess(image_path)
    # box, draw_img = cornerHarris_img(original_img, opened)
    box, draw_img = goodFeaturesToTrack_img(original_img, opened)
    box = utils.order_points(box)
    q.put((os.path.splitext(os.path.basename(image_path))[0], box))
    result_img = utils.Perspective_transform(box, original_img)

    # print(box)
    # cv2.imshow("original", original_img)
    # cv2.imshow("opened", opened)
    # cv2.imshow("closed", closed)
    # cv2.imshow("opened", opened)
    # cv2.imshow("draw_img", draw_img)
    # cv2.imshow("result_img", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite(output_path, result_img)
    # cv2.imwrite(output_path.replace("jpg", "scaled.jpg"), scaled_img)
    # cv2.imwrite(output_path.replace("jpg", "opened.jpg"), opened)
    # cv2.imwrite(output_path.replace("jpg", "draw.jpg"), draw_img)
    # cv2.imwrite(output_path.replace("jpg", "mask.jpg"), mask)

    print(f"处理完成: {output_path}")


def main():
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    q = multiprocessing.Manager().Queue()
    for pic in os.listdir(pic_path):
        pic = pic.lower()
        if "cr2" in pic:
            continue
        if multiprocess:
            pool.apply_async(
                perspective_correction,
                args=(
                    os.path.join(pic_path, pic),
                    os.path.join(out_path, pic),
                    q,
                ),
            )
        else:
            perspective_correction(
                os.path.join(pic_path, pic),
                os.path.join(out_path, pic),
                q,
            )
    if multiprocess:
        pool.close()
        pool.join()
    boxes = []
    while not q.empty():
        boxes.append(q.get())
    boxes = sorted(boxes, key=lambda x: x[0])
    with open(
        os.path.join(os.path.dirname(__file__), "box_by_cornerHarris.txt"), "w"
    ) as f:
        for name, box in boxes:
            f.write(
                f"{name}:({box[0][0]},{box[0][1]});({box[1][0]},{box[1][1]});({box[2][0]},{box[2][1]});({box[3][0]},{box[3][1]})\n"
            )


if __name__ == "__main__":
    main()
