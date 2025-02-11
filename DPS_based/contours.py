# 基于轮廓检测的方法找到包围轮廓的最大矩形，而后透视变换。
# 由于只是找包围轮廓的矩形，而非四边形，因此无法实现透视矫正，只是裁边。

import cv2
import numpy as np
import os
import utils
import multiprocessing
import math

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


def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 计算最大轮廓的旋转包围盒
    sorted_c = sorted(contours, key=cv2.contourArea, reverse=True)
    c = sorted_c[0]
    # 获取包围盒（中心点，宽高，旋转角度）
    # 直接得到矩形而不是四边形，所以无法做到透视矫正，只是裁边
    rect = cv2.minAreaRect(c)
    # 计算面积占比,如果面积占比过大则取第二大的轮廓
    height, width, channels = original_img.shape
    if rect[1][0] * rect[1][1] / (height * width) > 0.9:
        c = sorted_c[1]
        rect = cv2.minAreaRect(c)
    # 获取四个顶点坐标
    box = np.int32(cv2.boxPoints(rect))
    box = utils.order_points(box)
    draw_img = cv2.drawContours(original_img.copy(), contours, -1, (0, 0, 255), 3)
    draw_img = utils.draw_poly(draw_img, box)
    draw_img = cv2.putText(
        draw_img,
        f"{cv2.contourArea(sorted_c[0])/(height*width):.2f};{cv2.contourArea(sorted_c[1])/(height*width):.2f}",
        (100, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        10,
        (0, 0, 255),
        20,
    )

    # print("box[0]:", box[0])
    # print("box[1]:", box[1])
    # print("box[2]:", box[2])
    # print("box[3]:", box[3])
    return box, draw_img


def Perspective_transform(box, original_img):
    # 获取画框宽高(x=orignal_W,y=orignal_H)
    orignal_H = math.ceil(
        np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2)
    )
    orignal_W = math.ceil(
        np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2)
    )

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[1], box[2], box[3], box[0]])
    pts2 = np.float32(
        [
            [0, 0],
            [int(orignal_W + 1), 0],
            [int(orignal_W + 1), int(orignal_H + 1)],
            [0, int(orignal_H + 1)],
        ]
    )

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(
        original_img, M, (int(orignal_W + 3), int(orignal_H + 1))
    )

    return result_img


def perspective_correction(image_path, output_path, q):
    """读取文档图片，检测边缘并矫正"""
    print(f"正在处理: {image_path}")

    original_img, scaled_img, gray_img, equalized, binary, closed, opened = preprocess(
        image_path
    )
    box, draw_img = findContours_img(original_img, opened)
    q.put((os.path.splitext(os.path.basename(image_path))[0], box))
    result_img = Perspective_transform(box, original_img)

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
        # if "CR2" not in pic:
        #     continue
        if multiprocess:
            pool.apply_async(
                perspective_correction,
                args=(
                    os.path.join(pic_path, pic),
                    os.path.join(out_path, pic.replace("CR2", "JPG")),
                    q,
                ),
            )
        else:
            perspective_correction(
                os.path.join(pic_path, pic),
                os.path.join(out_path, pic.replace("CR2", "JPG")),
                q,
            )
    if multiprocess:
        pool.close()
        pool.join()
    boxes = []
    while not q.empty():
        boxes.append(q.get())
    boxes = sorted(boxes, key=lambda x: x[0])
    with open(os.path.join(os.path.dirname(__file__), "box.txt"), "w") as f:
        for name, box in boxes:
            f.write(
                f"{name}:({box[0][0]},{box[0][1]});({box[1][0]},{box[1][1]});({box[2][0]},{box[2][1]});({box[3][0]},{box[3][1]})\n"
            )


if __name__ == "__main__":
    main()
