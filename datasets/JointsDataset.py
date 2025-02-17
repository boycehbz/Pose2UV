# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.imutils import vis_img
import torch.nn as nn
import pycocotools.mask as maskUtils
from utils.imutils import vis_img
from utils.imutils import *
logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, root, image_set, is_train, transform=None, visible_only=False):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []
        self.visible_only = visible_only

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = None #cfg.OUTPUT_DIR
        self.data_format = 'jpg' #cfg.DATASET.DATA_FORMAT

        self.scale_factor = 0.35 #cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = 45 #cfg.DATASET.ROT_FACTOR
        self.flip = True #cfg.DATASET.FLIP
        self.num_joints_half_body = 8 #cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = 0.3 #cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = True #cfg.DATASET.COLOR_RGB

        self.target_type = 'gaussian' #  cfg.MODEL.TARGET_TYPE
        self.image_size = np.array([256, 256])
        self.heatmap_size = np.array([64, 64])
        self.sigma = 2 #cfg.MODEL.SIGMA
        self.use_different_joints_weight = False #cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0] + 1
        h = right_bottom[1] - left_top[1] + 1

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def annToMask(self, segm, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        def _annToRLE(segm, height, width):
            """
            Convert annotation which can be polygons, uncompressed RLE to RLE.
            :return: binary mask (numpy 2D array)
            """
            if isinstance(segm, list):
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, height, width)
                rle = maskUtils.merge(rles)
            elif isinstance(segm['counts'], list):
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, height, width)
            else:
                # rle
                rle = segm
            return rle

        rle = _annToRLE(segm, height, width)
        mask = maskUtils.decode(rle)
        return mask

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        img_h, img_w = data_numpy.shape[:2]

        mask = db_rec['mask']
        mask = self.annToMask(mask, data_numpy.shape[0], data_numpy.shape[1])

        joints = db_rec['joints_3d'].copy()
        joints_vis = db_rec['joints_3d_vis']
        det_pose = db_rec['det_pose'].copy()
        det_vis = np.ones_like(det_pose) 
        det_vis[:,2] = det_pose[:,2].copy()
        bbox = db_rec['bbox']

        if self.visible_only:
            for i in range(self.num_joints):
                if int(joints[i][1]) >= img_h or int(joints[i][1]) < 0:
                    joints[i] = 0.
                    joints_vis[i] = 0.
                    continue
                if int(joints[i][0]) >= img_w or int(joints[i][0]) < 0:
                    joints[i] = 0.
                    joints_vis[i] = 0.
                    continue
                if mask[int(joints[i][1]), int(joints[i][0])] == 0:
                    joints[i] = 0.
                    joints_vis[i] = 0.

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                det_pose, det_vis = fliplr_joints(
                    det_pose, det_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        joints_heatmap = joints.copy()
        joints_heatmap_16 = joints.copy()
        det_pose_heatmap = det_pose.copy()
        det_pose_heatmap_16 = det_pose.copy()
        trans = get_affine_transform(c, s, r, self.image_size)
        trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)
        trans_heatmap_16 = get_affine_transform(c, s, r, np.array([16, 16]))

        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(
            mask,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if mask.max() > 0:
            mask_flag = 1
        else:
            mask_flag = 0

        # vis_img('img', input)
        # vis_img('mask', mask*255)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)
                joints_heatmap_16[i, 0:2] = affine_transform(joints_heatmap_16[i, 0:2], trans_heatmap_16)

            det_pose[i, 0:2] = affine_transform(det_pose[i, 0:2], trans)
            det_pose_heatmap[i, 0:2] = affine_transform(det_pose_heatmap[i, 0:2], trans_heatmap)
            det_pose_heatmap_16[i, 0:2] = affine_transform(det_pose_heatmap_16[i, 0:2], trans_heatmap_16)

        target, target_weight = self.generate_target(joints_heatmap, joints_vis, self.heatmap_size)
        target_16, _ = self.generate_target(joints_heatmap_16, joints_vis, np.array([16, 16]))

        full_heat, full_heat_weight = self.generate_input(det_pose_heatmap, det_vis, self.heatmap_size)
        full_heat_16, _ = self.generate_input(det_pose_heatmap_16, det_vis, np.array([16, 16]))

        target_16 = torch.from_numpy(target_16)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        full_heat_16 = torch.from_numpy(full_heat_16)
        full_heat = torch.from_numpy(full_heat)
        full_heat_weight = torch.from_numpy(full_heat_weight)

        target_256 = self.upsample(target[None,:])
        target_256 = self.upsample(target_256)[0]

        full_heat_256 = self.upsample(full_heat[None,:])
        full_heat_256 = self.upsample(full_heat_256)[0]

        mask_16 = cv2.resize(mask, (16,16), interpolation=cv2.INTER_CUBIC)
        mask_64 = cv2.resize(mask, (64,64), interpolation=cv2.INTER_CUBIC)

        mask = torch.from_numpy(mask).float()
        mask_16 = torch.from_numpy(mask_16).float()
        mask_64 = torch.from_numpy(mask_64).float()

        # vis_img('mask16', mask_16*255)
        # vis_img('mask64', mask_64*255)
        # vis_img('mask', mask*255)

        meta = {
            'img': input.float(),
            'input_heat': [full_heat_16.float(), full_heat_256.float()],
            'gt_heat': [target.float(), target_256.float()],
            # 'input_heat': [target_16.float()],
            'vis': target_weight[:,0].float(),
            'img_path': image_file,
            'img_id': os.path.basename(image_file).split('.')[0],
            'bbox': torch.from_numpy(bbox).float(),
            'mask': [mask_16, mask_64, mask],
            'imgnum': imgnum,
            'joints': torch.from_numpy(joints.astype(np.float32)).float(),
            'joints_vis': torch.from_numpy(joints_vis).float(),
            'center': torch.from_numpy(np.array(c)).float(),
            'scale1': torch.from_numpy(np.array(s)).float(),
            'scale': torch.from_numpy(np.array([trans[0][0], trans[1][1]]).astype(np.float32)).float(),
            'offset': torch.from_numpy(np.array([trans[0][2], trans[1][2]]).astype(np.float32)).float(),
            'rotation': torch.from_numpy(np.array(r)).float(),
            'score': torch.from_numpy(np.array(score)).float(),
            'uv_flag': torch.from_numpy(np.array([0], dtype=np.float32)).float(),
            'pose_flag': torch.from_numpy(np.array([1], dtype=np.float32)).float(),
            'mask_flag': torch.from_numpy(np.array([mask_flag], dtype=np.float32)).float(),
        }

        # # vis
        # img = np.ascontiguousarray(img[:,:,::-1])
        # for kp in joints[:,:2].astype(np.int):
        #     img = cv2.circle(img, tuple(kp), 2, (0,0,255), -1)
        # vis_img('img', img)

        # origin_img = cv2.imread(image_file)
        # origin_joints = (joints[:,:2] - meta['offset'][np.newaxis]) / meta['scale'][np.newaxis]
        # for kp in origin_joints[:,:2].astype(np.int):
        #     origin_img = cv2.circle(origin_img, tuple(kp), 2, (0,0,255), -1)
        # vis_img('origin_img', origin_img)
        return meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_input(self, joints, joints_vis, heatmap_size):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, -1]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
                
                if target_weight[joint_id] == 0:
                    continue

                mu_x = joints[joint_id][0]
                mu_y = joints[joint_id][1]
                
                # 生成过程与hrnet的heatmap size不一样
                x = np.arange(0, heatmap_size[0], 1, np.float32)
                y = np.arange(0, heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                v = target_weight[joint_id]
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2)) * v

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_target(self, joints, joints_vis, heatmap_size):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
                
                if target_weight[joint_id] == 0:
                    continue

                mu_x = joints[joint_id][0]
                mu_y = joints[joint_id][1]
                
                # 生成过程与hrnet的heatmap size不一样
                x = np.arange(0, heatmap_size[0], 1, np.float32)
                y = np.arange(0, heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight
