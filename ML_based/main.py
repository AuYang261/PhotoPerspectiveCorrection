# 基于机器学习的方法: Unet

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from segmentation_models_pytorch import Unet
import os
import json
import numpy as np

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

datas_path = os.path.join(os.path.dirname(__file__), "..", "datas", "private")
imgs_path = os.path.join(datas_path, "imgs")
labels_path = os.path.join(datas_path, "labels")
masks_path = os.path.join(datas_path, "masks")


def json_to_mask(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    cv2.imwrite(output_path, mask)


class DocumentDataset(Dataset):
    def __init__(self, image_paths, mask_paths=[]):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx]) / 255.0
        h, w = img.shape[:2]
        # resize to that can be divided by 32
        img = cv2.resize(img, (w // 10 // 32 * 32, h // 10 // 32 * 32))
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if len(self.mask_paths) <= idx:
            return torch.tensor(img, dtype=torch.float32).permute(
                2, 0, 1
            ), torch.tensor(np.zeros_like(img), dtype=torch.float32).unsqueeze(0)
        # 随机改变对比度
        # img = cv2.convertScaleAbs(img, alpha=np.random.randint(9, 13) / 10, beta=0)
        # 随机旋转
        # angel = np.random.randint(0, 20)
        # img = utils.rotate_img(img, angel)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        # mask = utils.rotate_img(mask, angel)
        h, w = mask.shape[:2]
        mask = cv2.resize(mask, (w // 10 // 32 * 32, h // 10 // 32 * 32))
        if h > w:
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), torch.tensor(
            mask, dtype=torch.float32
        ).unsqueeze(0)


def main():
    global datas_path, imgs_path, labels_path, masks_path
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if not os.path.exists(datas_path):
        os.makedirs(datas_path)
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    if len(os.listdir(masks_path)) == 0:
        for json_path in os.listdir(labels_path):
            json_to_mask(
                os.path.join(labels_path, json_path),
                os.path.join(masks_path, json_path.replace("json", "png")),
            )
    # 加载数据集
    dataloader = DataLoader(
        DocumentDataset(
            [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)],
            [os.path.join(masks_path, x) for x in os.listdir(masks_path)],
        ),
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 定义 U-Net
    model = Unet(
        "resnet34", encoder_weights="imagenet", classes=1, activation=torch.nn.ReLU
    )

    # 加载模型
    if os.path.exists(os.path.join(datas_path, "model.pth")):
        model.load_state_dict(torch.load(os.path.join(datas_path, "model.pth")))

    model.to(device)
    # model = torch.compile(model)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(100):
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(datas_path, "model.pth"))

    # 测试
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            preds = torch.where(
                preds > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device)
            )
            for i in range(len(images)):
                cv2.imshow("img", images[i].permute(1, 2, 0).cpu().numpy())
                cv2.imshow("mask", masks[i].squeeze().cpu().numpy())
                cv2.imshow("pred", preds[i].squeeze().cpu().numpy())
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    imgs_path = "/Users/bytedance/Desktop/Personal/照片/老照片"
    dataloader = DataLoader(
        DocumentDataset(
            [
                os.path.join(imgs_path, x)
                for x in os.listdir(imgs_path)
                if x.lower().endswith(".jpg")
            ],
        ),
    )
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            preds = model(images)
            preds = torch.where(
                preds > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device)
            )
            for i in range(len(images)):
                cv2.imshow("img", images[i].permute(1, 2, 0).cpu().numpy())
                cv2.imshow("pred", preds[i].squeeze().cpu().numpy())
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
