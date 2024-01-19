import os
import sys
sys.path.append('./')
import numpy as np
import json
import torch
import cv2
from tqdm import tqdm
import torch.utils.data as data
from utils.heatmap import gen_heatmap, heatmap_stand
from utils.imutils import *

class halpe(data.Dataset):
    def __init__(self, train=True, use_mask=False, data_folder='', smpl=None, uv_generator=None, occlusions=None, poseseg=False):
        self.label_format = 'coco_13'
        self.use_smpl = False
        self.occlusions = occlusions
        self.poseseg = poseseg
        self.is_train = train
        dataset_dir = os.path.join(data_folder, 'HALPE')
        self.use_mask = use_mask

        assert self.is_train, 'train only'
        self.image_dir = os.path.join(dataset_dir,'images/train2015')
        dataset = os.path.join(dataset_dir,'annot-alphapose/halpe_train_v1.json')
        self.hmmask_dir = ' '
        

        self.alphapose = []
        self.alphapose_to_lsp = [16,14,12,11,13,15,10,8,6,5,7,9,0,0]
        self.images = []
        self.masks = []
        self.gt_2ds = []
        self.boxs = []
        self.shapes = []
        self.poses = []

        params = self.load_json(dataset)

        len_ = len(params) 
        
        for index in params:
            param = params[index]
            self.gt_2ds.append(np.array(param['lsp_joints_2d'][0], dtype=np.float))
            self.images.append(param['filename'])
            self.boxs.append(param['bbox'])
            if np.array(param['openpose_ant'][0]).max() > 0:
                self.alphapose.append(np.array(param['openpose_ant'][0])[self.alphapose_to_lsp])
            else:
                self.alphapose.append([])


        self.data_type = torch.float32
        self.device = torch.device('cpu')
        self.smpl = smpl
        self.uv_generator = uv_generator
        self.len = len(self.images)
        
    def load_json(self, path):
        with open(path) as f:
            param = json.load(f)
        return param

    def create_UV_maps(self, index=0):
        data = {}
        image_path = os.path.join(self.image_dir, self.images[index])
        rate = random.random()
    
        if len(self.alphapose[index]) > 0 and rate < 0.8:
            kp_2d = self.alphapose[index]
            kp_2d[:,2] = 1
            use_ap = True
        else:
            kp_2d = self.gt_2ds[index]
            kp_2d[:,2][np.where(kp_2d[:,2]>0)] = 1
            use_ap = False
      
        image = cv2.imread(image_path)
        box = self.boxs[index]
        lt = np.array([box[0], box[1]])
        rb = np.array([box[0] + box[2], box[1] + box[3]])


        if self.use_mask:
            mask_path = os.path.join(self.hmmask_dir, self.masks[index])
            img_mask = cv2.imread(mask_path, 0)
            if img_mask is None:
                img_mask = np.ones(image.shape[:2])
        else:
            img_mask = np.ones(image.shape[:2])


        if self.is_train:
            # color adjustment
            if self.occlusions is not None:
                image, patch = color_gamma_contrast_patch(image, patch)
            else:
                image = color_gamma_contrast(image)
            # scale
            image, img_mask, kp_2d, lt0, rb0, scale_ = scale(image, img_mask, kp_2d, lt, rb)
            # used for the image that target person is not in the center
            image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt0, rb0, kp_2d)
        else:
            image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt, rb, kp_2d)
       
        if self.occlusions is not None:
            image, img_mask = synthesize_occlusion(image, patch, patch_mask, lt, rb, img_mask)

        dst_image = image
        dst_mask = img_mask
        
        # generate heatmap 256*256*1
        kp_2d = convert_to_coco_pts(kp_2d)
        if self.label_format == 'coco_13':
            coco_kp = np.zeros((13,3)).astype(np.float32)
            coco_kp[1:13] = kp_2d[5:17]
            coco_kp[0] = kp_2d[0]
            kp_out = np.ones_like(coco_kp) * -1
        else:
            kp_out = np.ones_like(kp_2d) * -1
        
        for ind, kp in enumerate(coco_kp):
            # if kp[2] > 0.2:
            kp_out[ind] = kp
                   
        heatmap_size = [16, 256]
        k_size = [1, 3]
        heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(heatmap_size, k_size)]
        full_heatmaps = gen_heatmap(kp_out, heatmap_st, label_format= self.label_format)
        fullheatmap = full_heatmaps[1]
        data['full_heat'] = full_heatmaps
        # #### visualize heatmap
        # merge_heatmap = np.max(fullheatmap , axis=0)
        # gtt = convert_color(merge_heatmap*255)
        # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
        # vis_img("hm", dst)


        # heatmaps = gen_heatmap(kp_out, heatmap_st, label_format= self.label_format)
        # merge_heatmap = np.max(heatmaps[0], axis=0)
        # gtt = convert_color(merge_heatmap*255)
        # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
        # vis_img("hm", dst)


        mask_torch = mask_to_torch(dst_mask)
        rgb_img = im_to_torch(dst_image)
        heatmap_torch = mask_to_torch(fullheatmap).squeeze(0)
        cat_mask = torch.cat([rgb_img, mask_torch], 0)
        cat_heat = torch.cat([rgb_img, heatmap_torch], 0)
        uv_gt = uv_to_torch_noModifyChannel(np.ones((256,256,3))*-1)
        
        # if self.use_smpl:
        #     smpl_param = torch.cat([pose[0], shape[0]], 0)
        #     data['smpl'] = smpl_param
        
        data['img'] = rgb_img
        data['f_uv'] = uv_gt
        data['cat_mask'] = cat_mask
        data['cat_heat'] = cat_heat
        return data

    def __getitem__(self, index):
        data = self.create_UV_maps(index)
        return data

    def __len__(self):
        return self.len

if __name__ == "__main__":
    from utils.smpl_torch_batch import SMPLModel
    from utils.uv_map_generator import UV_Map_Generator
    data_folder = 'E:\MP-data'
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./data/model_lsp.pkl', 
                        data_type=torch.float32,
                    )
    # load UV generator
    generator = UV_Map_Generator(
        UV_height=256,
        UV_pickle='./data/param.pkl'
    )
    dataset = lsp_mpii(train=True, use_mask=False, data_folder=data_folder, smpl=model_smpl, uv_generator=generator, poseseg=False)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True,
        num_workers=0, pin_memory=True
    )
    num =0 
    for i, data in enumerate(data_loader): # enumerate(tqdm(data_loader, total=len(data_loader))):
        # print(num)
        num+=1
    
          
# import os
# import sys

# # from torch import imag 
# sys.path.append('')
# import numpy as np
# import json
# import torch
# import cv2
# import torch.utils.data as data
# from utils.imutils import *
# from utils.heatmap import gen_heatmap, heatmap_stand
# from tqdm import tqdm
# from utils.uv_gen import Index_UV_Generator
# class lsp_mpii(data.Dataset):
#     def __init__(self, train=True, use_mask=False, data_folder='', smpl=None, uv_generator=None, occlusions=None, poseseg=False):
#         self.label_format = 'coco_13'
#         self.occlusions = None
#         self.poseseg = poseseg
#         self.is_train = train
#         dataset_dir = os.path.join(data_folder, 'data_2d')
#         self.use_mask = False #use_mask
#         if self.is_train:
#             self.image_dir = os.path.join(dataset_dir,'images')
#             dataset = os.path.join(dataset_dir,'data2d.npz') #data2d lsp_eft
#             self.hmmask_dir = '' #os.path.join(dataset_dir, '3DPW/masks')
#         else:
#             print('train only')
       
#         self.train, self.valid = [], []
#         self.images = []
#         self.masks = []
#         self.gt_2ds = []
#         self.boxs = []
#         self.shapes = []
#         self.poses = []
#         self.extris = []
#         self.boxs = []
#         param = np.load(dataset, allow_pickle=True)
        
#         len_ = len(param['img_path'])

#         sp= param['betas']
#         ps = param['pose']
#         ip = param['img_path']
#         kps = param['kp_2d']
#         bx = param['bbox']
#         for index in range(len_):
#             self.gt_2ds.append(np.array(kps[index], dtype=np.float)) #proj_2d
#             self.shapes.append(np.array(sp[index], dtype=np.float))
#             self.poses.append(np.array(ps[index], dtype=np.float))
#             self.images.append(ip[index])
#             self.boxs.append(bx[index])

#             # no mask
#             # self.masks.append(param['img_path'][index])
         
#         self.data_type = torch.float32
#         self.device = torch.device('cpu')
#         self.smpl = smpl
#         self.uv_generator = uv_generator
#         self.len = len(self.images) if self.is_train else 5000


#     def create_UV_maps(self, index=0):
#         data = {}
#         image_path = os.path.join(self.image_dir, self.images[index])
#         pose = self.poses[index]
#         shape = self.shapes[index]
#         kp_2d = self.gt_2ds[index]
#         image = cv2.imread(image_path)
       
    
#         box = self.boxs[index]
       
#        # bbox = [y for x in [box[0], box[1]] for y in x]
#         lt = np.array([box[0], box[1]])
#         rb = np.array([box[2], box[3]])

#         if self.use_mask:
#             mask_path = os.path.join(self.hmmask_dir, self.masks[index])
#             img_mask = cv2.imread(mask_path, 0)
#         else:
#             img_mask = np.ones(image.shape[:2])

#         if self.occlusions is not None:
#             i = random.randint(0,len(self.occlusions)-1)
#             patch = cv2.imread(self.occlusions[i])
#             patch_mask = cv2.imread(self.occlusions[i].replace('images', 'masks').replace('instance', 'mask'), 0)

#         if self.is_train:
#             # color adjustment
#             if self.occlusions is not None:
#                 image, patch = color_gamma_contrast_patch(image, patch)
#             else:
#                 image = color_gamma_contrast(image)
#             # scale
#             image, img_mask, kp_2d, lt0, rb0, scale_ = scale(image, img_mask, kp_2d, lt, rb)
#             # used for the image that target person is not in the center
#             image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt0, rb0, kp_2d)
#         else:
#             image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt, rb, kp_2d)
       
#         if self.occlusions is not None:
#             image, img_mask = synthesize_occlusion(image, patch, patch_mask, lt, rb, img_mask)

#         dst_image = image
#         dst_mask = img_mask
#         # generate heatmap 256*256*1
#         kp_2d_coco = convert_to_coco_pts(kp_2d)
#         if self.label_format == 'coco_13':
#             coco_kp = np.zeros((13,3)).astype(np.float32)
#             coco_kp[1:13] = kp_2d_coco[5:17]
#             coco_kp[0] = kp_2d_coco[0]
#             kp_out = np.ones_like(coco_kp) * -1
#         else:
#             coco_kp = kp_2d
#             kp_out = np.ones_like(kp_2d) * -1
#         ind = 0
#         for kp in coco_kp:
#             if max(int(kp[1]),int(kp[0])) > 255 or min(int(kp[1]),int(kp[0]))<0 or int(kp[2]) == 0:
#                 ind += 1
#                 continue
#             else:
#                 kp_out[ind] = kp
#                 ind += 1
#         heatmap_size = [256]
#         heatmap_st = [heatmap_stand(s, s, 3) for s in heatmap_size]
#         masks = [np.clip(cv2.resize(dst_mask, (s, s), interpolation=cv2.INTER_CUBIC), 0, 255).reshape(1, s, s) for s in heatmap_size]
#         heatmaps = gen_heatmap(kp_out, heatmap_st, label_format=self.label_format)
#         merge_heatmap = np.max(heatmaps[0], axis=0)
#         # gtt = convert_color(merge_heatmap*255)
#         # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
#         # vis_img("hm", dst)

#         # print(image_path)
#         # print(kp_2d)
#         # if self.images[index] == 'im05942.jpg':
#         #     gtt = convert_color(merge_heatmap*255)
#         #     dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
#         #     vis_img("hm", dst)

#         # kp_2d = np.insert(kp_2d, 2,values=1.,axis=1)
#         shape = torch.from_numpy(shape).type(torch.FloatTensor).unsqueeze(0).to(self.device)
#         pose = torch.from_numpy(pose).type(torch.FloatTensor).unsqueeze(0).to(self.device)
#         _trans = torch.zeros((1, 3), dtype=self.data_type, device=self.device)
#         mesh, lsp_joints = self.smpl(shape, pose, _trans)
#         # uv = self.uv_generator.get_UV_t(mesh)[0].permute(2,0,1)
#         # im = tmp_uv[0].numpy()
#         # obj = resample_np(self.uv_generator, im)
#         # self.smpl.write_obj(obj, 'torch_uv.obj')
#         # vis_img('t', im+0.5)
#         # tmp = Index_UV_Generator(128, 128, 'BF')
#         # vmin = torch.min(mesh, axis=1)[0] 
#         # vmax = torch.max(mesh, axis=1)
#         # #box = (vmax-vmin).max() #2019.11.9 vmax.max()
#         # box = 2 # define 2 meters bounding-box  @buzhenhuang 21/04/2020
#         # verts = mesh #(mesh - vmin) / box
#         # t1 = tmp.get_UV_map(verts) #[0].numpy()
#         # obj = tmp.resample(t1)
#         # self.smpl.write_obj(obj[0],'new_uv.obj')
#         # vis_img('t', t1)

#         mesh_3d = mesh[0].numpy()
#         uv, vmin, vmax = self.uv_generator.get_UV_map(mesh_3d)
        
#         # partmask = np.loadtxt('./data/part_mask.txt', dtype=np.float32)
#         # for i in range(6):
#         #     i += 1.0
#         #     part = np.where(partmask==i)
#         #     part_value = uv[part]
#         #     back = np.zeros((part[0].max()-part[0].min()+1,part[1].max()-part[1].min()+1,3))
#         #     np0 = part[0] - part[0].min()
#         #     np1 = part[1] - part[1].min()
#         #     back[(np0,np1)] = part_value 
#         #     out = padding_reshape(back)
#         #     vis_img('p', out+0.5)

#       #  ocuv = self.uv_generator.get_ocuv(uv, mesh_3d,mesh_2d, dst_mask, vmin, vmax)

#         # #visualize occlusion uv
#         # cv2.imshow('rgb', dst_image/255)
#         # cv2.imshow('mask', dst_mask/255)
#         # cv2.imshow('uv',(uv+0.5))
#         # cv2.imshow('oc uv',(ocuv+0.5))
#         # cv2.waitKey()
#         # dst_mask = np.stack((dst_mask, dst_mask, dst_mask), axis=-1).astype(np.float32)
#         # mask_torch = im_to_torch(dst_mask)
#         mask_torch = mask_to_torch(dst_mask)
#         rgb_img = im_to_torch(dst_image)
#         heatmap_torch = mask_to_torch(merge_heatmap)
#         cat_mask = torch.cat([rgb_img, mask_torch], 0)
#         cat_heat = torch.cat([rgb_img, heatmap_torch], 0)
#         uv_gt = uv_to_torch_noModifyChannel(uv)

#         data['img'] = rgb_img
#         data['f_uv'] = uv_gt
#         data['cat_mask'] = cat_mask
#         data['cat_heat'] = cat_heat
#         return data       
    

#     def __getitem__(self, index):
#         data  = self.create_UV_maps(index)
#         return data

#     def __len__(self):
#         return self.len

# if __name__ == "__main__":
#     from utils.smpl_torch_batch import SMPLModel
#     from utils.uv_map_generator import UV_Map_Generator
#     from open3d import *
#     import open3d
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
#     dataset = lsp_mpii(train=True, use_mask=False, data_folder=data_folder, smpl=model_smpl, uv_generator=generator, poseseg=False)
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=10, shuffle=True,
#         num_workers=0, pin_memory=True
#     )
#     num =0 
#     for i, data in enumerate(tqdm(data_loader, total=len(data_loader))):
#         print(i)
#         # uv = data['f_uv']
#         # out = generator.resample_t(uv)
#         # lsp_joints = torch.tensordot(out, model_smpl.joint_regressor, dims=([1], [0])).transpose(1, 2)
#         # v_point = out[0].cpu().numpy()

#         # J_regressor = np.load('data/J_regressor_h36m.npy')
      
#         # # gt_3djoint = np.dot(J_regressor, v_point)
#         # J_regressor = torch.from_numpy(J_regressor).float()
#         # joints = torch.tensordot(out, J_regressor, dims=([1], [1])).transpose(1, 2)
#         # # bl = np.load('bone_length.npy')
#         # # tmp = cal_bonelength(joints) 
#         # # np.save('bone_length.npy', tmp[0].numpy())
#         # gt_3djoint = joints[0].cpu().numpy()

#         # mesh = open3d.geometry.PointCloud()
#         # mesh.points = open3d.utility.Vector3dVector(v_point)
#         # colors = [[0.5, 0.5, 0.5] for i in range(len(v_point))]
#         # mesh.colors = open3d.utility.Vector3dVector(colors)

#         # points = lsp_joints[0] # gt_3djoint
#         # point_cloud = open3d.geometry.PointCloud()
#         # point_cloud.points = open3d.utility.Vector3dVector(points)
#         # colors = [[1, 0, 0] for i in range(len(points))]
#         # point_cloud.colors = open3d.utility.Vector3dVector(colors)

#         # # lines = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[8,11],[13,12],[12,11],[8,14],[14,15],[16,15],[10,8],[8,7]]
#         # lines = [[0,1],[1,2],[2,12],[3,12],[4,3],[5,4],[6,7],[7,8],[8,12],[9,12],[10,9],[11,10],[12,13]]
#         # line_pcd = open3d.geometry.LineSet()
#         # line_pcd.lines = open3d.utility.Vector2iVector(lines)
#         # line_pcd.points = open3d.utility.Vector3dVector(points)
#         # colors = [[1, 0, 0] for i in range(len(lines))]
#         # line_pcd.colors = open3d.utility.Vector3dVector(colors)

#         # points2 = gt_3djoint
#         # point_cloud2 = open3d.geometry.PointCloud()
#         # point_cloud2.points = open3d.utility.Vector3dVector(points2)
#         # colors = [[0, 1, 0] for i in range(len(points2))]
#         # point_cloud2.colors = open3d.utility.Vector3dVector(colors)

#         # lines2 = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[8,11],[13,12],[12,11],[8,14],[14,15],[16,15],[10,8],[8,7]]
#         # line_pcd2 = open3d.geometry.LineSet()
#         # line_pcd2.lines = open3d.utility.Vector2iVector(lines2)
#         # line_pcd2.points = open3d.utility.Vector3dVector(points2)
#         # colors2 = [[0, 1, 0] for i in range(len(lines2))]
#         # line_pcd2.colors = open3d.utility.Vector3dVector(colors2)

#         # open3d.visualization.draw_geometries([mesh]+[point_cloud]+[point_cloud2]+[line_pcd]+[line_pcd2]) # 

#         # visualize the joint order of H36M regressor
#         # for ss in range(17):
#         #     points = gt_3djoint[:(ss+1)]
#         #     point_cloud = open3d.geometry.PointCloud()
#         #     point_cloud.points = open3d.utility.Vector3dVector(points)
#         #     colors = [[1, 0, 0] for i in range(len(points))]
#         #     point_cloud.colors = open3d.utility.Vector3dVector(colors)
#         #     open3d.visualization.draw_geometries([mesh]+[point_cloud])

#        # open3d.visualization.draw_geometries([mesh]+[point_cloud]+[line_pcd])

    

#         # faces = np.array(model_smpl.faces)
#         # mesh = open3d.geometry.TriangleMesh()
#         # mesh.vertices  = open3d.utility.Vector3dVector(v_point)
#         # mesh.triangles = open3d.utility.Vector3iVector(faces)
#         # mesh.compute_vertex_normals()
#         # # open3d.visualization.draw_geometries([mesh])

#         # point_cloud = open3d.geometry.PointCloud()
#         # point_cloud.points = open3d.utility.Vector3dVector(points)
        
#         # mesh = open3d.geometry.PointCloud()
#         # mesh.points = open3d.utility.Vector3dVector(v_point)

#         # lines = [[0,1],[1,2],[2,12],[3,12],[4,3],[5,4],[6,7],[7,8],[8,12],[9,12],[10,9],[11,10],[12,13]]
#         # line_pcd = open3d.geometry.LineSet()
#         # line_pcd.lines = open3d.utility.Vector2iVector(lines)
#         # line_pcd.points = open3d.utility.Vector3dVector(points)
#         # colors = [[1, 0, 0] for i in range(len(lines))]
#         # line_pcd.colors = open3d.utility.Vector3dVector(colors)
#         # # open3d.visualization.draw_geometries([mesh]+[point_cloud]+[line_pcd])
#         # open3d.visualization.draw_geometries([point_cloud]+[line_pcd]) 
#         # tmp1 = out[0].numpy()
#         # model_smpl.write_obj(tmp1, 'test1.obj')
#         # num+=1
#     print(num)
        