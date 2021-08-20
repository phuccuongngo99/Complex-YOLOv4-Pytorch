import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from torch.utils.data import DataLoader
from data_process.kitti_json_dataset import KittiDataset as KittiJsonDataset
from data_process.kitti_dataset import KittiDataset

kitti_dataset_dir = '/home/deeplearning/Code_2021/AIDrivers/3d_detection/repo/Complex-YOLOv4-Pytorch/dataset/kitti'
json_kitti_dataset_dir = '/home/deeplearning/Code_2021/AIDrivers/3d_detection/repo/Complex-YOLOv4-Pytorch/dataset/kitti_json'

#train_dataset = KittiDataset(kitti_dataset_dir)
train_dataset = KittiJsonDataset(json_kitti_dataset_dir)
train_dataloader = DataLoader(train_dataset, batch_size=1,
                                collate_fn=train_dataset.collate_fn)

IMG_SIZE = 800

for batch_i, (img_files, imgs, targets) in enumerate(train_dataloader):
    img_bev = imgs.squeeze() * 255
    img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
    img_bev = cv2.resize(img_bev, (IMG_SIZE, IMG_SIZE))

    # Rescale target
    targets[:, 2:6] *= IMG_SIZE

    # Get yaw angle
    targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

    for c, x, y, w, l, yaw in targets[:, 1:7].numpy():
        # Draw rotated box
        kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])
    
    img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)

    cv2.imshow('test-img', img_bev)
    print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
    if cv2.waitKey(0) & 0xFF == 27:
        break

    cv2.destroyAllWindows()