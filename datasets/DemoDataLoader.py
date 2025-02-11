import os
import sys
import numpy as np
import pickle
import torch
import cv2
from tqdm import tqdm
import torch.utils.data as data
from utils.imutils import *
import torchvision.transforms as transforms

class DemoData(data.Dataset):
    def __init__(self, train=True, use_mask=False, data_folder='', smpl=None, uv_generator=None, occlusions=None, poseseg=False, use_gt=False):
        self.label_format = 'coco_13'
        self.occlusions = occlusions
        self.poseseg = poseseg
        self.is_train = train
        self.use_mask = use_mask
        self.data_type = torch.float32
        self.np_type = np.float32
        self.use_gt = use_gt
        self.dataset_dir = data_folder

        self.images = os.listdir(self.dataset_dir)

        self.device = torch.device('cpu')
        self.smpl = smpl
        self.generator = uv_generator
        self.len = len(self.images)

        self.normalize_img = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.sigma = 2

        self.image_width = 256 #cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = 256 # cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        self.halpe_to_lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img, ul, br, new_shape, new_x, new_y, old_x, old_y = crop(rgb_img, center, scale, 
                      [256, 256], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img, ul, br, new_shape, new_x, new_y, old_x, old_y

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2], center, scale, 
                                  [256, 256], rot=r)
        # # convert to normalized coordinates
        # kp[:,:-1] = 2.*kp[:,:-1]/256 - 1.
        # # flip the x coordinates
        # if f:
        #      kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def generate_input(self, joints, heatmap_size):
        '''
        :param joints:  [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_joints = joints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints[:, -1]

        target = np.zeros((num_joints,
                            heatmap_size[1],
                            heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(num_joints):

            mu_x = joints[joint_id][0]
            mu_y = joints[joint_id][1]
            
            # 生成过程与hrnet的heatmap size不一样
            x = np.arange(0, heatmap_size[0], 1, np.float32)
            y = np.arange(0, heatmap_size[1], 1, np.float32)
            y = y[:, np.newaxis]

            v = target_weight[joint_id]
            target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2)) * v

        return target

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + (w - 1) * 0.5
        center[1] = y + (h - 1) * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def prepare(self, img_orig, bboxes, poses, device):
        data = {}

        # Load image
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        orig_shape = np.array(img_orig.shape)[:2]
        img_h, img_w = orig_shape

        imgs, full_heats, scales, centers = [], [], [], []
        flip, pn, rot, sc, gt_input = 0, np.ones(3), 0, 1, 0 
        for bbox, pose in zip(bboxes, poses):


            bbox = bbox.reshape(-1,)
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]

            scale = np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])/200.
            bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=[192, 256]).max()
            scale = bbox_size/200.

            focal_length = 5000 / 256 * orig_shape.max()

            img, crop_ul, crop_br, new_shape, new_x, new_y, old_x, old_y = self.rgb_processing(img_orig.copy(), center, sc*scale, rot, flip, pn)
            img = torch.from_numpy(img).float()

            pred_keypoints = pose[:17].copy()
            pred_keypoints = self.j2d_processing(pred_keypoints, center, sc*scale, rot, flip)

            pred_keypoints_16 = pred_keypoints.copy()
            pred_keypoints_16[:,:2] = pred_keypoints[:,:2] / 16.

            full_heat_16 = self.generate_input(pred_keypoints_16, np.array([16, 16]))
            # bbox = bbox.reshape(2, 2)
            # lt = np.array(bbox[0])
            # rb = np.array(bbox[1])

            # kp_2d = pose.reshape(-1, 3)[self.halpe_to_lsp][self.lsp14_to_lsp13]

            # assert kp_2d.shape == (13, 3) and kp_2d[:,2].max() <= 1.5

            # rgb_img, full_heat_inp, scale, offset = create_demo_data(img.copy(), lt, rb, kp_2d, occlusions=self.occlusions)
            imgs.append(self.normalize_img(img[None,:]))
            full_heats.append(torch.from_numpy(full_heat_16).float()[None,:])
            scales.append(scale)
            centers.append(center)

        data['img'] = torch.cat(imgs).to(device)
        data['input_heat'] = [torch.cat(full_heats).to(device)]
        data['scale'] = np.array(scales)
        data['center'] = np.array(centers)

        return data

    def create_demo_data(self, index=0):
        # import time
        # time_start = time.time()
        # load data
        image_path = os.path.join(self.dataset_dir, self.images[index].replace('\\', '/')) 

        # time_end = time.time()
        # print('dataloader time: %f' %(time_end - time_start))
        return image_path

    def __getitem__(self, index):
        data = self.create_demo_data(index)
        return data

    def __len__(self):
        return self.len

