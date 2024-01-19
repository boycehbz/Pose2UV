import os
import sys
import numpy as np
import pickle
import torch
import cv2
from tqdm import tqdm
import torch.utils.data as data
from utils.imutils import *
from utils.dataset_handle import create_UV_maps, create_poseseg

class MPData(data.Dataset):
    def __init__(self, train=True, use_mask=False, data_folder='', smpl=None, uv_generator=None, occlusions=None, poseseg=False, name='', use_gt=False):
        self.label_format = 'coco_13'
        self.occlusions = occlusions
        self.poseseg = poseseg
        self.is_train = train
        self.use_mask = use_mask
        self.data_type = torch.float32
        self.np_type = np.float32
        self.use_gt = use_gt
        self.name = name
        self.dataset_dir = os.path.join(data_folder, self.name)
        if self.is_train:
            dataset = os.path.join(self.dataset_dir, 'annot/train.pkl')
        else:
            dataset = os.path.join(self.dataset_dir, 'annot/test.pkl')

        self.lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        self.halpe_to_lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

        self.features = []

        self.images = []
        self.masks = []
        self.img_size = []
        self.boxs = []
        self.shapes = []
        self.poses = []

        self.intris = []

        self.gt_2ds = []
        self.gt_3ds = []
        self.pred_2ds = []

        params = self.load_pkl(dataset)

        for seq in tqdm(params, total=len(params)):
            if len(seq) < 1:
                continue

            for i, frame in enumerate(seq):
                # print("frame:", frame)
                for key in frame.keys():
                    if key in ['img_path', 'h_w']:
                        continue

                    self.images.append(frame['img_path'])
                    self.img_size.append(frame['h_w'])

                    self.shapes.append(np.array(frame[key]['betas'], dtype=self.np_type))
                    self.poses.append(np.array(frame[key]['pose'], dtype=self.np_type))
                    self.boxs.append(np.array(frame[key]['bbox'], dtype=self.np_type))
                    # self.gt_2ds.append(np.array(frame[key]['lsp_joints_2d'], dtype=self.np_type)[self.lsp14_to_lsp13])
                    # self.gt_3ds.append(np.array(frame[key]['lsp_joints_3d'], dtype=self.np_type))
                    self.pred_2ds.append(np.array(frame[key]['halpe_joints_2d_pred'], dtype=self.np_type).reshape(-1, 3)[self.halpe_to_lsp][self.lsp14_to_lsp13])
                    self.gt_2ds.append(np.array(frame[key]['halpe_joints_2d'], dtype=self.np_type)[self.halpe_to_lsp][self.lsp14_to_lsp13])

                    gt_3d = np.array(frame[key]['halpe_joints_3d'], dtype=self.np_type)
                    if gt_3d.shape[1] == 3:
                        gt_3d = np.insert(gt_3d, 3, values=1.0, axis=1)
                    self.gt_3ds.append(gt_3d)
                    self.intris.append(np.array(frame[key]['intri'], dtype=self.np_type))
                    self.masks.append(None)

        del frame
        del params

        self.device = torch.device('cpu')
        self.smpl = smpl
        self.generator = uv_generator
        self.len = len(self.images)

    def load_pkl(self, path):
        """"
        load pkl file
        """
        param = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
        return param

    def create_UV_maps(self, index=0):
        # import time
        # time_start = time.time()
        # load data
        image_path = os.path.join(self.dataset_dir, self.images[index].replace('\\', '/')) 
        mask_path = self.masks[index]
        pose = self.poses[index]
        shape = self.shapes[index]
        bbox = self.boxs[index]
        lt = np.array(bbox[0])
        rb = np.array(bbox[1])

        if self.pred_2ds[index] is not None:
            kp_2d = self.pred_2ds[index]
        else:
            kp_2d = np.zeros((13, 3), dtype=self.np_type)
        assert kp_2d.shape == (13, 3) and kp_2d[:,2].max() <= 1.5

        data = create_UV_maps(image_path, mask_path, lt, rb, kp_2d, pose, shape, self.smpl, self.generator, occlusions=self.occlusions, is_train=self.is_train)
        data['gt_3d'] = self.gt_3ds[index]

        # time_end = time.time()
        # print('dataloader time: %f' %(time_end - time_start))
        return data

    def create_poseseg(self, index=0):
        # load data
        image_path = self.images[index]
        mask_path = self.masks[index]
        bbox = self.boxs[index]
        lt = np.array(bbox[0])
        rb = np.array(bbox[1])

        if self.pred_2ds[index] is not None:
            kp_2d = self.pred_2ds[index]
        else:
            kp_2d = np.zeros((13, 3), dtype=self.np_type)
        assert kp_2d.shape == (13, 3) and kp_2d[:, 2].max() <= 1.5

        data = create_poseseg(image_path, mask_path, lt, rb, kp_2d, self.smpl, self.generator, occlusions=self.occlusions, is_train=self.is_train)
        data['gt_3d'] = self.gt_3ds[index]
        return data

    def __getitem__(self, index):
        if self.poseseg:
            data = self.create_poseseg(index)
        else:
            data = self.create_UV_maps(index)
        return data

    def __len__(self):
        return self.len

