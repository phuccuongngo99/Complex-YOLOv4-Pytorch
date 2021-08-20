"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Ngo Phuc Cuong
# DoC: 2021.07.28
# email: phuccuongngo99@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset to load LIDAR Only annotation
file in json format from this annotation tool: 
https://github.com/songanz/3D-LiDAR-annotator

# Borrowed extensively from:
https://github.com/maudzung/Complex-YOLOv4-Pytorch
"""
import json
import sys
import os
import random
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2
from easydict import EasyDict

sys.path.append('../')

from data_process import transformation, kitti_bev_utils, kitti_data_utils
import config.kitti_config as cnf

class ObjectJson3d(object):
    ''' 3d object label for json file from 3D Annotator Tool'''
    '''
    An example of box annotation dictionary 
    {
    'center': {'x': 3.156369430522359,
                'y': 8.834011474234416,
                'z': 1.0342535972595215},
    'width': 4.475703220592086,
    'length': 3.2934427372386126,
    'height': 2.6601858139038086,
    'angle': 2.4860782874388785,
    'object_id': 'car'}
    '''

    def __init__(self, box_annot_dict: dict):
        box_annot_dict = EasyDict(box_annot_dict)

        # extract label
        self.type = box_annot_dict.object_id  # 'Car', 'Pedestrian', ...
        self.cls_id = self.cls_type_to_id(self.type)

        # extract 3d bounding box information
        self.h = box_annot_dict.height  # box height
        self.w, self.l = self.annotator_wl_to_lidar_wl(
                        box_annot_dict.width,
                        box_annot_dict.length) # box width, length
        self.t = (box_annot_dict.center.x, 
                box_annot_dict.center.y, 
                box_annot_dict.center.z)  # location (x,y,z) in camera coord.
        
        # Convert Annotator's yaw angle to Lidar coord's yaw angle
        self.ry = self.annotator_yaw_to_lidar_yaw(box_annot_dict.angle)

    def cls_type_to_id(self, cls_type) -> int:
        # Car and Van ==> Car class
        # Pedestrian and Person_Sitting ==> Pedestrian Class
        CLASS_NAME_TO_ID = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            'Van': 0,
            'Person_sitting': 1
        }
        if cls_type not in CLASS_NAME_TO_ID.keys():
            return -1
        return CLASS_NAME_TO_ID[cls_type]

    def annotator_wl_to_lidar_wl(self, annotator_w: int, annotator_l: int) -> Tuple[int]:
        '''
        This will return (lidar_width, lidar_length)
        '''
        # For our annotation convention,
        # Width and height are flipped
        return annotator_l, annotator_w


    def annotator_yaw_to_lidar_yaw(self, annotator_yaw: float) -> float:
        '''
        Function to convert angle given in 3D annotation tool
        to yaw angle in Lidar coordinate

        The return angle will be in range [0, 2pi]
        '''
        lidar_y = annotator_yaw
        lidar_y = (lidar_y + np.pi) % (2 * np.pi)
        lidar_y = - (lidar_y + np.pi/2)
        
        return lidar_y


class JsonAnnotUtil:
    def read_label(label_file: str) -> List[ObjectJson3d]:
        # Load json file
        annot_dict = json.load(open(label_file, "r"))

        return [ObjectJson3d(box_annot_dict)
                for box_annot_dict in annot_dict['bounding boxes']]


class KittiDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', lidar_transforms=None, aug_transforms=None, multiscale=False,
                 num_samples=None, mosaic=False, random_padding=False):
        self.dataset_dir = dataset_dir
        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.multiscale = multiscale
        self.lidar_transforms = lidar_transforms
        self.aug_transforms = aug_transforms
        self.img_size = cnf.BEV_WIDTH
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.mosaic = mosaic
        self.random_padding = random_padding
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]

        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]

        if self.is_test:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        else:
            self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            if self.mosaic:
                img_files, rgb_map, targets = self.load_mosaic(index)

                return img_files[0], rgb_map, targets
            else:
                return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""

        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)
        rgb_map = kitti_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION_X, cnf.DISCRETIZATION_Y, cnf.boundary)
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

        return img_file, rgb_map

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""

        sample_id = int(self.sample_id_list[index])

        lidarData = self.get_lidar(sample_id)
        # We override get_label method here to read
        # from json file and return a list of our custom object
        objects = self.get_label(sample_id)
        calib = self.get_calib(sample_id)

        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)

        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        # TODO: Check whether other augmentation agrees with us or not

        if self.lidar_transforms is not None:
            lidarData, labels[:, 1:] = self.lidar_transforms(lidarData, labels[:, 1:])

        b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)
        rgb_map = kitti_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION_X, cnf.DISCRETIZATION_Y, cnf.boundary)
        target = kitti_bev_utils.build_yolo_target(labels)
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

        # on image space: targets are formatted as (box_idx, class, x, y, w, l, im, re)
        n_target = len(target)
        targets = torch.zeros((n_target, 8))
        if n_target > 0:
            targets[:, 1:] = torch.from_numpy(target)

        rgb_map = torch.from_numpy(rgb_map).float()

        if self.aug_transforms is not None:
            rgb_map, targets = self.aug_transforms(rgb_map, targets)

        return img_file, rgb_map, targets

    def load_mosaic(self, index):
        """loads images in a mosaic
        Refer: https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        """

        targets_s4 = []
        img_file_s4 = []
        if self.random_padding:
            yc, xc = [int(random.uniform(-x, 2 * self.img_size + x)) for x in self.mosaic_border]  # mosaic center
        else:
            yc, xc = [self.img_size, self.img_size]  # mosaic center

        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            img_file, img, targets = self.load_img_with_targets(index)
            img_file_s4.append(img_file)

            c, h, w = img.size()  # (3, 608, 608), torch tensor

            # place img in img4
            if i == 0:  # top left
                img_s4 = torch.full((c, self.img_size * 2, self.img_size * 2), 0.5, dtype=torch.float)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img_s4[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]  # img_s4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # on image space: targets are formatted as (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
            if targets.size(0) > 0:
                targets[:, 2] = (targets[:, 2] * w + padw) / (2 * self.img_size)
                targets[:, 3] = (targets[:, 3] * h + padh) / (2 * self.img_size)
                targets[:, 4] = targets[:, 4] * w / (2 * self.img_size)
                targets[:, 5] = targets[:, 5] * h / (2 * self.img_size)

            targets_s4.append(targets)
        if len(targets_s4) > 0:
            targets_s4 = torch.cat(targets_s4, 0)
            torch.clamp(targets_s4[:, 2:4], min=0., max=(1. - 0.5 / self.img_size), out=targets_s4[:, 2:4])

        return img_file_s4, img_s4, targets_s4

    def __len__(self):
        return len(self.sample_id_list)

    def remove_invalid_idx(self, image_idx_list):
        """Discard samples which don't have current training class objects, which will not be used for training."""

        sample_id_list = []
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)

            print(labels[..., 1:4])
            if not noObjectLabels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                                   calib.P)  # convert rect cam to velo cord

            print(labels[..., 1:4])
            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_point_cloud_range(labels[i, 1:4]):
                        valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)

        return sample_id_list

    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if (self.batch_count % 10 == 0) and self.multiscale and (not self.mosaic):
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack(imgs)
        if self.img_size != cnf.BEV_WIDTH:
            imgs = F.interpolate(imgs, size=self.img_size, mode="bilinear", align_corners=True)
        self.batch_count += 1

        return paths, imgs, targets

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        # assert os.path.isfile(img_file)
        return cv2.imread(img_file)  # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return kitti_data_utils.Calibration(calib_file)

    # Override this method
    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.json'.format(idx))
        # assert os.path.isfile(label_file)
        return JsonAnnotUtil.read_label(label_file)

    # def get_label(self, idx):
    #     label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
    #     # assert os.path.isfile(label_file)
    #     return kitti_data_utils.read_label(label_file)