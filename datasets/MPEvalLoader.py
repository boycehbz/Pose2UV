import os
import sys
import numpy as np
import pickle
import torch
import cv2
from tqdm import tqdm
import torch.utils.data as data
from utils.imutils import *
from utils.dataset_handle import create_UV_maps, eval_handle, eval_poseseg_handle

class MPeval(data.Dataset):
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
        dataset_dir = os.path.join(data_folder, self.name)
        dataset = os.path.join(dataset_dir,'annot/test.pkl')

        self.alphapose = []
        self.openpose = []
        self.alphapose_to_lsp13 = [16,14,12,11,13,15,10,8,6,5,7,9,17]
        self.openpose_to_lsp13 = [16,14,12,11,13,15,10,8,6,5,7,9,0]
        self.lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        self.images = []
        self.masks = []
        self.gt_2ds = []
        self.boxs = []
        self.gt_3ds = []
        self.intris = []
        self.shapes = []
        self.poses = []
        param = self.load_pkl(dataset)

        if self.poseseg:
            for i, index in enumerate(param):
                frame = param[index]
                ids = len(frame['bbox'])
                for p in range(ids):
                    self.images.append(os.path.join(dataset_dir, frame['img_path']))
                    self.boxs.append(np.array(frame['bbox'][p], dtype=self.np_type))
                    try:
                        gt_2d = np.array(frame['lsp_joints_2d'][p], dtype=self.np_type)[self.lsp14_to_lsp13]
                        if gt_2d.max() < 0:
                            gt_2d = None
                    except:
                        gt_2d = None
                    try:
                        alphapose = np.array(frame['alphapose'][p], dtype=self.np_type)[self.alphapose_to_lsp13]
                        if alphapose.max() < 0:
                            alphapose = None
                    except:
                        alphapose = None
                    
                    self.gt_2ds.append(gt_2d)
                    self.alphapose.append(alphapose)
                    if self.use_mask and frame['mask_path'][p] is not None:
                        self.masks.append(os.path.join(dataset_dir, frame['mask_path'][p]))
                    else:
                        self.masks.append(None)
        else:
            for i, index in enumerate(param):
                frame = param[index]
                ids = len(frame['bbox'])
                for p in range(ids):
                    # load different GT joints for different dataset
                    if self.name in ['VCLMP', 'h36m_valid_protocol1', 'h36m_valid_protocol2'] : # VCLMP, Human36M use LSP format for evaluation
                        if frame['lsp_joints_3d'] is not None:
                            self.gt_3ds.append(np.array(frame['lsp_joints_3d'][p], dtype=self.np_type))
                        else:
                            continue
                    elif self.name in ['MPI3DPW'] : # 3DPW use SMPL format for evaluation
                        if frame['smpl_joints_3d'] is not None:
                            self.gt_3ds.append(np.array(frame['smpl_joints_3d'][p], dtype=self.np_type))
                        else:
                            continue
                    elif self.name in ['Panoptic_haggling1', 'Panoptic_mafia2', 'Panoptic_pizza1', 'Panoptic_ultimatum1', 'Panoptic_Eval'] : # 3DPW use H36M format for evaluation
                        if frame['h36m_joints_3d'] is not None:
                            self.gt_3ds.append(np.array(frame['h36m_joints_3d'][p], dtype=self.np_type))
                        else:
                            continue
                    elif self.name in ['MuPoTS_origin'] : # 3DPW use MPI format for evaluation
                        if frame['mpi_joints_3d'] is not None:
                            self.gt_3ds.append(np.array(frame['mpi_joints_3d'][p], dtype=self.np_type))
                        else:
                            continue
                    if 'intri' in frame.keys():
                        self.intris.append(np.array(frame['intri'], dtype=self.np_type))
                    else:
                        self.intris.append(None)
                    if frame['betas'][p] is not None:
                        self.shapes.append(np.array(frame['betas'][p], dtype=self.np_type))
                    else:
                        self.shapes.append(None)
                    if frame['pose'][p] is not None:
                        self.poses.append(np.array(frame['pose'][p], dtype=self.np_type))
                    else:
                        self.poses.append(None)
                    self.images.append(os.path.join(dataset_dir, frame['img_path']))
                    self.boxs.append(np.array(frame['bbox'][p], dtype=self.np_type))
                    try:
                        gt_2d = np.array(frame['lsp_joints_2d'][p], dtype=self.np_type)[self.lsp14_to_lsp13]
                        if gt_2d.max() < 0:
                            gt_2d = None
                    except:
                        gt_2d = None
                    # try:
                    #     openpose = np.array(frame['openpose'][p], dtype=self.np_type)[self.openpose_to_lsp13]
                    #     if openpose.max() < 0:
                    #         openpose = None
                    # except:
                    #     openpose = None
                    try:
                        alphapose = np.array(frame['alphapose'][p], dtype=self.np_type)[self.alphapose_to_lsp13]
                        if alphapose.max() < 0:
                            alphapose = None
                    except:
                        alphapose = None
                    # # process the openpose head
                    # if openpose is not None:
                    #     if alphapose is not None:
                    #         openpose[-1] = alphapose[-1]
                    #     elif gt_2d is not None:
                    #         openpose[-1] = gt_2d[-1]
                    
                    self.gt_2ds.append(gt_2d)
                    # self.openpose.append(openpose)
                    self.alphapose.append(alphapose)
                    # if self.use_mask and frame['mask_path'][p] is not None:
                    #     self.masks.append(os.path.join(dataset_dir, frame['mask_path'][p]))
                    # else:
                    #     self.masks.append(None)
        # Release the memory
        del frame
        del param
        
        self.device = torch.device('cpu')
        self.smpl = smpl
        self.generator = uv_generator
        self.len = len(self.images)
        
    def load_pkl(self, path):
        """"
        load pkl file
        """
        param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
        return param

    def eval_handle(self, index=0):
        # load data
        image_path = self.images[index]
        intri = self.intris[index]
        gt_3d = self.gt_3ds[index]
        # mask_path = self.masks[index]
        pose = self.poses[index]
        shape = self.shapes[index]
        bbox = self.boxs[index]
        lt = np.array(bbox[0])
        rb = np.array(bbox[1])
        # random select 2D pose
        if self.alphapose[index] is not None and not self.use_gt:
            kp_2d = self.alphapose[index]
        elif self.gt_2ds[index] is not None:
            kp_2d = self.gt_2ds[index]
        else:
            kp_2d = np.zeros((13, 3), dtype=self.np_type)
        assert kp_2d.shape == (13,3) and kp_2d[:,2].max() <= 1.5

        data = eval_handle(image_path, lt, rb, kp_2d, intri, gt_3d, pose, shape, smpl=self.smpl, occlusions=self.occlusions, is_train=self.is_train)
        return data

    def eval_poseseg_handle(self, index=0):
        # load data
        image_path = self.images[index]
        mask_path = self.masks[index]
        bbox = self.boxs[index]
        lt = np.array(bbox[0])
        rb = np.array(bbox[1])
        # random select 2D pose
        if self.alphapose[index] is not None:
            alpha_2d = self.alphapose[index]
        else:
            alpha_2d = np.zeros((13, 3), dtype=self.np_type)

        if self.gt_2ds[index] is not None:
            gt_2d = self.gt_2ds[index]
        else:
            gt_2d = np.zeros((13, 3), dtype=self.np_type)

        data = eval_poseseg_handle(image_path, lt, rb, alpha_2d, gt_2d, mask_path, smpl=self.smpl, occlusions=self.occlusions, is_train=self.is_train)
        return data

    def __getitem__(self, index):
        if self.poseseg:
            data = self.eval_poseseg_handle(index)
        else:
            data = self.eval_handle(index)
        return data

    def __len__(self):
        return self.len

# if __name__ == "__main__":
#     from utils.smpl_torch_batch import SMPLModel
#     from utils.uv_map_generator import UV_Map_Generator
#     data_folder = 'E:\MP-data'
#     model_smpl = SMPLModel(
#                         device=torch.device('cpu'),
#                         model_path='./data/model_lsp.pkl', 
#                         data_type=torch.float32,
#                     )
#     # load UV generator
#     generator = UV_Map_Generator(
#         UV_height=256,
#         UV_pickle='./data/param.pkl'
#     )
#     dataset = VCL_MP(train=True, use_mask=True, data_folder=data_folder, smpl=model_smpl, uv_generator=generator, poseseg=False)
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=10, shuffle=True,
#         num_workers=0, pin_memory=True
#     )
#     num =0 
#     for i, data in enumerate(tqdm(data_loader, total=len(data_loader))):
#         # print(num)
#         num+=1
#     print(num)
        
