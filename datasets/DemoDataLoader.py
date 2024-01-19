import os
import sys
import numpy as np
import pickle
import torch
import cv2
from tqdm import tqdm
import torch.utils.data as data
from utils.imutils import *
from utils.dataset_handle import create_demo_data

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

        self.lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        self.halpe_to_lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def prepare(self, img, bboxes, poses, device):
        data = {}
        imgs, full_heats, scales, offsets = [], [], [], []
        for bbox, pose in zip(bboxes, poses):

            bbox = bbox.reshape(2, 2)
            lt = np.array(bbox[0])
            rb = np.array(bbox[1])

            kp_2d = pose.reshape(-1, 3)[self.halpe_to_lsp][self.lsp14_to_lsp13]

            assert kp_2d.shape == (13, 3) and kp_2d[:,2].max() <= 1.5

            rgb_img, full_heat_inp, scale, offset = create_demo_data(img.copy(), lt, rb, kp_2d, occlusions=self.occlusions)

            imgs.append(rgb_img[None,:])
            full_heats.append(torch.from_numpy(full_heat_inp[0])[None,:])
            scales.append(scale)
            offsets.append(offset)

        data['img'] = torch.cat(imgs).to(device)
        data['fullheat'] = [torch.cat(full_heats).to(device)]
        data['scale'] = np.array(scales)
        data['offset'] = np.array(offsets)

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

