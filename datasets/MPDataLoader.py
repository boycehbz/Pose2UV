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
import torchvision.transforms as transforms
from utils.transforms import get_affine_transform, affine_transform

class MPData(data.Dataset):
    def __init__(self, train=True, use_mask=False, data_folder='', smpl=None, uv_generator=None, occlusions=None, poseseg=False, name='', use_gt=False):
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
            self.dataset = os.path.join(self.dataset_dir, 'annot/train.pkl')
        else:
            self.dataset = os.path.join(self.dataset_dir, 'annot/test.pkl')

        self.sigma = 2
        self.noise_factor = 0.4
        self.rot_factor = 30
        self.scale_factor = 0.25
        self.normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.aspect_ratio = 1.0
        self.pixel_std = 200
        self.image_size = [256, 256]
        self.heatmap_size = [256, 256]

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
        self.valids = []

        params = self.load_pkl(self.dataset)

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

                    if frame[key]['betas'] is not None:
                        self.shapes.append(np.array(frame[key]['betas'], dtype=self.np_type))
                        self.poses.append(np.array(frame[key]['pose'], dtype=self.np_type))
                    else:
                        self.shapes.append(np.zeros((10,), dtype=self.np_type))
                        self.poses.append(np.zeros((72,), dtype=self.np_type))
                    self.boxs.append(np.array(frame[key]['bbox'], dtype=self.np_type))

                    self.pred_2ds.append(np.array(frame[key]['halpe_joints_2d_pred'], dtype=self.np_type).reshape(-1, 3))
                    self.gt_2ds.append(np.array(frame[key]['halpe_joints_2d'], dtype=self.np_type))
                    gt_3d = np.array(frame[key]['halpe_joints_3d'], dtype=self.np_type)
                    if gt_3d.shape[1] == 3:
                        gt_3d = np.insert(gt_3d, 3, values=1.0, axis=1)
                    self.gt_3ds.append(gt_3d)
                    self.intris.append(np.array(frame[key]['intri'], dtype=self.np_type))
                    self.masks.append(None)
                    self.valids.append(np.array(frame[key]['valid'], dtype=self.np_type))

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

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        gt_input = 0
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
            if np.random.uniform() <= 0.5:
                gt_input = 1

        return flip, pn, rot, sc, gt_input

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        trans_img = get_affine_transform(center, scale, rot, self.image_size)

        rgb_img = cv2.warpAffine(
            rgb_img,
            trans_img,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        # rgb_img, ul, br, new_shape, new_x, new_y, old_x, old_y = crop(rgb_img, center, scale, 
        #               [256, 256], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        # rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, size):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        trans = get_affine_transform(center, scale, r, size)

        nparts = kp.shape[0]

        for i in range(nparts):
            kp[i, 0:2] = affine_transform(kp[i, 0:2], trans)

        #     kp[i,0:2] = transform(kp[i,0:2], center, scale, 
        #                           [256, 256], rot=r)
        # # convert to normalized coordinates
        # kp[:,:-1] = 2.*kp[:,:-1]/256 - 1.
        # # flip the x coordinates
        # if f:
        #      kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def fliplr_joints(self, joints, width, matched_parts):
        """
        flip coords
        """
        # Flip horizontal
        joints[:, 0] = width - joints[:, 0] - 1

        # Change left-right parts
        for pair in matched_parts:
            joints[pair[0], :], joints[pair[1], :] = \
                joints[pair[1], :], joints[pair[0], :].copy()

        return joints

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def estimate_trans_hmr2(self, joints, keypoints, focal_length, img_h, img_w):
        
        joints = joints.detach().numpy()
        
        return estimate_translation_np(
            joints,
            keypoints[:, :2],
            keypoints[:, 2],
            focal_length=focal_length,
            cx=img_w / 2., 
            cy=img_h / 2.,
        )

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


    def vis_input(self, load_data):
        
        print('visulizing input......')

        img_vis = img.permute((1,2,0)).detach().numpy()*255
        for kp in keypoints[:,:2].astype(np.int):
            img_vis = cv2.circle(img_vis, tuple(kp), 3, (0,0,255), -1)
        for kp in pred_keypoints[:,:2].astype(np.int):
            img_vis = cv2.circle(img_vis, tuple(kp), 3, (0,255,0), -1)
        vis_img('img', img_vis)

        # Show image
        image = image.clone().numpy().transpose(1,2,0)
        image = image[:,:,::-1] * 255.
        cv2.imwrite('vis_input/img.png', image)
        # vis_img('img', image)

        # Show keypoints
        keypoints = (keypoints[:,:-1].detach().numpy() + 1.) * constants.IMG_RES * 0.5
        keypoints = keypoints.astype(np.int32)
        for k in keypoints:
            image = cv2.circle(image, tuple(k), 3, (0,0,255), -1)
        # vis_img('keyp', image)
        cv2.imwrite('vis_input/keyp.png', image)

        # Show SMPL
        pose = pose.reshape(-1, 72)
        betas = betas.reshape(-1, 10)
        trans = trans.reshape(-1, 3)
        extri = np.eye(4)
        intri = intri
        verts, joints = self.smpl(betas, pose, trans)
        verts = verts.detach().numpy()[0]
        projs, image = surface_projection(verts, self.smpl.faces, extri, intri, image.copy(), viz=False)
        # vis_img('smpl', image)
        cv2.imwrite('vis_input/smpl.png', image)

    def generate_target(self, joints, joints_vis, heatmap_size):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((joints.shape[0], 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]


        if True:
            target = np.zeros((joints.shape[0],
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(joints.shape[0]):
                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size, heatmap_size)
                
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


        return target, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size, heatmap_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight


    def create_UV_maps(self, index=0):
        # load data
        load_data = {}

        # Get augmentation parameters
        flip, pn, rot, sc, gt_input = self.augm_params()
        
        # # disable data augmentation 
        # flip, pn, rot, sc, gt_input = 0, np.ones(3), 0, 1, 0            
        
        # Load image
        imgname = os.path.join(self.dataset_dir, self.images[index])
        try:
            origin_img = cv2.imread(
                imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        except TypeError:
            print(imgname)

        orig_shape = np.array(img.shape)[:2]
        img_h, img_w = orig_shape
        
        # Load mask
        if self.masks[index] is not None:
            mask = cv2.imread(self.masks[index], 0)
        else:
            mask = np.ones((img_h, img_w), dtype=np.uint8)*255

        pose = self.poses[index].copy()
        betas = self.shapes[index].copy()
        
        bbox = self.boxs[index].reshape(-1,)
        center = np.array([(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2])

        scale = np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])/200.
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=[192, 256]).max()
        scale = bbox_size/200.

        focal_length = 5000 / 256 * orig_shape.max()

        # Process image
        if self.occlusions is not None:
            i = random.randint(0, len(self.occlusions)-1)
            patch = cv2.imread(self.occlusions[i])
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_mask = cv2.imread(self.occlusions[i].replace('images', 'masks').replace('instance', 'mask'), 0)

            img, mask = synthesize_occlusion(img, patch, patch_mask, bbox[:2], bbox[2:], mask)

        # path = os.path.join('output/syn_occ', self.images[index].replace('/', '_')[:-4])
        # save_syn_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path + '_img.jpg', origin_img)
        # cv2.imwrite(path + '_syn.jpg', save_syn_img)

        # vis_img('img', img)
        # vis_img('mask1', mask)
        # vis_img('patch', patch)
        # vis_img('patch_mask', patch_mask)

        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = img.astype(np.uint8)

        if self.transform:
            input = self.transform(img)

        keypoints = self.gt_2ds[index][:17].copy()
        keypoints = self.j2d_processing(keypoints, center, sc*scale, rot, flip, self.heatmap_size)

        pred_keypoints = self.pred_2ds[index][:17].copy()
        pred_keypoints = self.j2d_processing(pred_keypoints, center, sc*scale, rot, flip, self.heatmap_size)

        pred_keypoints_16 = self.pred_2ds[index][:17].copy()
        pred_keypoints_16 = self.j2d_processing(pred_keypoints_16, center, sc*scale, rot, flip, [16, 16])

        if flip:
            keypoints = self.fliplr_joints(
                keypoints, img.shape[1], self.flip_pairs)
            pred_keypoints = self.fliplr_joints(
                pred_keypoints, img.shape[1], self.flip_pairs)
            pred_keypoints_16 = self.fliplr_joints(
                pred_keypoints_16, 16, self.flip_pairs)

        pred_keypoints_vis = pred_keypoints.copy()
        pred_keypoints_vis[:,:2] = 1.

        pred_keypoints_16_vis = pred_keypoints_16.copy()
        pred_keypoints_16_vis[:,:2] = 1.

        full_heat, _ = self.generate_target(pred_keypoints, pred_keypoints_vis, np.array([256, 256]))
        full_heat_16, _ = self.generate_target(pred_keypoints_16, pred_keypoints_16_vis, np.array([16, 16]))

        # full_heat_16 = np.zeros((17,16,16), dtype=np.float32)
        # full_heat = np.zeros((17,256,256), dtype=np.float32)

        full_heat_16 = torch.from_numpy(full_heat_16)
        full_heat = torch.from_numpy(full_heat)


        # # img = torch.from_numpy(img).float()

        # # Get 2D keypoints and apply augmentation transforms
        # keypoints = self.gt_2ds[index][:17].copy()
        # keypoints = self.j2d_processing(keypoints, center, sc*scale, rot, flip)

        # pred_keypoints = self.pred_2ds[index][:17].copy()
        # pred_keypoints = self.j2d_processing(pred_keypoints, center, sc*scale, rot, flip)



        # pred_keypoints_vis = pred_keypoints.copy()
        # pred_keypoints_vis[:,:2] = 1.

        # pred_keypoints_16 = pred_keypoints.copy()
        # pred_keypoints_16[:,:2] = pred_keypoints[:,:2] / 16.
        # pred_keypoints_16_vis = pred_keypoints_16.copy()
        # pred_keypoints_16_vis[:,:2] = 1.

        # full_heat = self.generate_input(pred_keypoints, np.array([256, 256]))
        # full_heat_16 = self.generate_input(pred_keypoints_16, np.array([16, 16]))
        # full_heat, _ = self.generate_target(pred_keypoints, pred_keypoints_vis, np.array([256, 256]))
        # full_heat_16, _ = self.generate_target(pred_keypoints_16, pred_keypoints_16_vis, np.array([16, 16]))

        keypoints = torch.from_numpy(keypoints).float()
        pred_keypoints = torch.from_numpy(pred_keypoints).float()

        pose = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        betas = torch.from_numpy(betas).float()

        temp_pose = pose.clone().reshape(-1, 72)
        temp_shape = betas.clone().reshape(-1, 10)
        temp_trans = torch.zeros((temp_pose.shape[0], 3), dtype=temp_pose.dtype, device=temp_pose.device)
        verts, joints = self.smpl(temp_shape, temp_pose, temp_trans, halpe=True)
        
        verts = verts.squeeze(0)
        joints = joints.squeeze(0)
        conf = torch.ones((len(joints), 1)).float()
        joints = torch.cat([joints, conf], dim=1)

        uv, vmin, vmax = self.generator.get_UV_map(verts.detach().numpy())
        uv = uv_to_torch_noModifyChannel(uv)

        trans = self.estimate_trans_hmr2(joints, self.gt_2ds[index], focal_length, img_h, img_w) # crop translation
        trans = torch.from_numpy(trans).float()

        center = torch.from_numpy(np.array(center)).float()  

        load_data['uv_flag'] = np.array([1], dtype=np.float32)
        load_data['pose_flag'] = np.array([1], dtype=np.float32)
        load_data['mask_flag'] = np.array([0], dtype=np.float32)
        load_data['has_3d'] = np.ones(1)
        load_data['has_smpl'] = np.ones(1)
        load_data['valid'] = self.valids[index]
        load_data['verts'] = verts
        load_data['gt_3d'] = joints
        load_data['img'] = self.transform(img)
        load_data['origin_img'] = img
        load_data['pose'] = pose
        load_data['betas'] = betas
        load_data['gt_cam_t'] = trans
        load_data['img_path'] = imgname
        # load_data['img_id'] = int(os.path.basename(imgname).split('.')[0])
        load_data['center'] = center
        load_data['scale'] = sc*scale
        load_data['scale1'] = sc*scale
        load_data['bbox'] = bbox
        load_data['img_h'] = img_h
        load_data['img_w'] = img_w
        # load_data['pred_keypoints'] = pred_keypoints
        load_data['keypoints'] = keypoints
        load_data['gt_uv'] = uv
        load_data['input_heat'] = [full_heat_16.float(), full_heat.float()]

        vis = False
        if vis:
            self.vis_input(load_data)

        return load_data

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

