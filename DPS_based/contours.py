# 基于轮廓检测的方法找到包围轮廓的最大矩形，而后透视变换。
# 由于只是找包围轮廓的矩形，而非四边形，因此无法实现透视矫正，只是裁边。

import cv2
import numpy as np
import os
import utils
import multiprocessing
import math

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

pic_path = utils.pic_path
out_path = utils.out_path
multiprocess = False


def preprocess(input_dir):
    if input_dir.upper().endswith(".CR2"):
        original_img = utils.read_cr2_as_rgb(input_dir)
    else:
        original_img = cv2.imread(input_dir)
    scaled_img = cv2.convertScaleAbs(original_img, alpha=1.2, beta=0)
    gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    # equalized = cv2.equalizeHist(gray_img)
    equalized = gray_img
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算（去噪点）
    return original_img, scaled_img, gray_img, equalized, binary, closed, opened


def perspective_correction(image_path, output_path, q):
    """读取文档图片，检测边缘并矫正"""
    print(f"正在处理: {image_path}")

    original_img, scaled_img, gray_img, equalized, binary, closed, opened = preprocess(
        image_path
    )
    box, draw_img = utils.findContours_img(
        original_img, opened, perspective_correction=False
    )
    q.put((os.path.splitext(os.path.basename(image_path))[0], box))
    result_img = utils.Perspective_transform(box, original_img)

    # cv2.imshow("original", original_img)
    # cv2.imshow("gray", gray_img)
    # cv2.imshow("closed", closed)
    # cv2.imshow("opened", opened)
    # cv2.imshow("draw_img", draw_img)
    # cv2.imshow("result_img", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite(output_path, result_img)
    # cv2.imwrite(output_path.replace("jpg", "scaled.jpg"), scaled_img)
    # cv2.imwrite(output_path.replace("jpg", "equalized.jpg"), equalized)
    # cv2.imwrite(output_path.replace("jpg", "binary.jpg"), binary)
    # cv2.imwrite(output_path.replace("jpg", "opened.jpg"), opened)
    # cv2.imwrite(output_path.replace("jpg", "draw.jpg"), draw_img)

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
    with open(os.path.join(os.path.dirname(__file__), "box_by_contours.txt"), "w") as f:
        for name, box in boxes:
            f.write(
                f"{name}:({box[0][0]},{box[0][1]});({box[1][0]},{box[1][1]});({box[2][0]},{box[2][1]});({box[3][0]},{box[3][1]})\n"
            )


if __name__ == "__main__":
    main()
