# 基于grabCut的方法
# 根据实验，此方法效果较好

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

scale = 0.3


def preprocess(input_dir):
    if input_dir.lower().endswith(".cr2"):
        original_img = utils.read_cr2_as_rgb(input_dir)
    else:
        original_img = cv2.imread(input_dir)
    scaled_img = cv2.convertScaleAbs(original_img, alpha=1.2, beta=0)
    scaled_img = cv2.resize(scaled_img, (0, 0), fx=scale, fy=scale)
    blurred = cv2.GaussianBlur(scaled_img, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    # 闭运算（链接块）
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=3)
    # 开运算（去噪点）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=3)
    return original_img, scaled_img, closed, opened


def grabCut(img):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask1 = np.zeros(img.shape[:2], np.uint8)
    rect1 = (0, 0, img.shape[1] - 10, img.shape[0] - 10)
    cv2.grabCut(img, mask1, rect1, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.zeros(img.shape[:2], np.uint8)
    rect2 = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask2, rect2, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask = np.where(
        (mask1 == 2) | (mask1 == 0) | (mask2 == 2) | (mask2 == 0), 0, 1
    ).astype("uint8")
    img = img * mask[:, :, np.newaxis]
    mask = mask * 255
    return mask, img


def perspective_correction(image_path, output_path, q):
    """读取文档图片，检测边缘并矫正"""
    print(f"正在处理: {image_path}")

    original_img, scaled_img, closed, opened = preprocess(image_path)
    mask, masked_img = grabCut(opened)
    mask = cv2.resize(mask, (0, 0), fx=1 / scale, fy=1 / scale)
    box, draw_img = utils.findContours_img(
        original_img, mask, perspective_correction=False
    )
    q.put((os.path.splitext(os.path.basename(image_path))[0], box))
    result_img = utils.Perspective_transform(box, original_img)

    # cv2.imshow("original", original_img)
    # cv2.imshow("opened", opened)
    # cv2.imshow("masked_img", masked_img)
    # cv2.imshow("mask", mask)
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
    with open(os.path.join(os.path.dirname(__file__), "box_by_grabcut.txt"), "w") as f:
        for name, box in boxes:
            f.write(
                f"{name}:({box[0][0]},{box[0][1]});({box[1][0]},{box[1][1]});({box[2][0]},{box[2][1]});({box[3][0]},{box[3][1]})\n"
            )


if __name__ == "__main__":
    main()
