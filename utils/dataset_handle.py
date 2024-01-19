import os
import cv2
import torch
import numpy as np
import random
import pycocotools.mask as mask_util
from utils.imutils import *
from utils.heatmap import gen_heatmap, heatmap_stand

def create_UV_maps(image_path, mask, lt, rb, kp_2d, pose, shape, smpl, generator, occlusions=None, is_train=False, supervise_2d=False):
    data = {}
    # image_path = 'C:\\Users\\123\Documents\Human-Training-v3.12\Human36M_MOSH\\' + image_path
    # load data
    image = cv2.imread(image_path)
    if mask is not None:
        img_mask = cv2.imread(mask, 0)
        img_mask = cv2.threshold(img_mask, 127, 255, type=cv2.THRESH_BINARY)[1]
        mask_flag = np.array([1], dtype=np.float32)
        pose_flag = np.array([1], dtype=np.float32)
    else:
        img_mask = np.ones(image.shape[:2]) * 255
        mask_flag = np.array([0], dtype=np.float32)
        pose_flag = np.array([0], dtype=np.float32)

    if kp_2d is None or kp_2d.max() < 1:
        pose_flag = np.array([0], dtype=np.float32)
        kp_2d = np.zeros((13, 3))

    if occlusions is not None:
        i = random.randint(0,len(occlusions)-1)
        patch = cv2.imread(occlusions[i])
        patch_mask = cv2.imread(occlusions[i].replace('images', 'masks').replace('instance', 'mask'), 0)

    if is_train:
        # color adjustment
        if occlusions is not None:
            image, patch = color_gamma_contrast_patch(image, patch)
        else:
            image = color_gamma_contrast(image)
        # used for the image that the target person is not in the center
        image, img_mask, kp_2d, lt, rb, s = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=True)
        image, img_mask, kp_2d, lt, rb = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=True)
        # image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt, rb, kp_2d)
    else:
        image, img_mask, kp_2d, lt, rb, s = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=False)
        image, img_mask, kp_2d, lt, rb = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=False)
    
    if occlusions is not None:
        image, img_mask = synthesize_occlusion(image, patch, patch_mask, lt, rb, img_mask)

    dst_image = image
    dst_mask = img_mask

    heatmap_size = [16, 256]
    k_size = [1, 3]
    # generate full heatmap 256*256*1
    coco_kp = kp_2d
    heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(heatmap_size, k_size)]
    full_heatmaps = gen_heatmap(coco_kp, heatmap_st)
    full_heat_inp = full_heatmaps

    if supervise_2d:
        # generate gt partial heatmap
        kp_out = np.zeros_like(coco_kp)
        ind = 0
        for kp in coco_kp:
            if max(int(kp[1]),int(kp[0])) > 255 or min(int(kp[1]),int(kp[0]))<0:
                ind += 1
                continue
            elif dst_mask[int(kp[1]),int(kp[0])] < 127:
                ind += 1
                continue
            else:
                kp_out[ind] = kp
                ind += 1
        pheatmap_size = [16, 32, 64, 256]
        pk_size = [1, 1, 1, 3]
        heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(pheatmap_size, pk_size)]
        partial_heatmaps = gen_heatmap(kp_out, heatmap_st)
        partial_heat_inp = partial_heatmaps

        mask_size = [16, 32, 64, 256]
        mask_st = [mask_to_torch(cv2.resize(dst_mask, (s,s))) for s in mask_size]

        data['mask'] = mask_st
        data['partialheat'] = partial_heat_inp

    if shape is not None and pose is not None:
        shape = torch.from_numpy(shape).type(torch.float32).reshape(1, 10)
        pose = torch.from_numpy(pose).type(torch.float32).reshape(1, 72)
        trans = torch.zeros((1, 3), dtype=torch.float32)
        mesh, lsp_joints = smpl(shape, pose, trans)
        mesh_3d = mesh[0].numpy()

        uv, vmin, vmax = generator.get_UV_map(mesh_3d)
        uv_flag = np.array([1], dtype=np.float32)
    else:
        uv = np.zeros((256,256,3))
        uv_flag = np.array([0], dtype=np.float32)

    # #visualize occlusion uv
    # cv2.imshow('rgb', dst_image/255)
    # cv2.imshow('mask', dst_mask/255)
    # cv2.imshow('uv',(uv+0.5))
    # tt = np.max(full_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('fullheat',dst)
    # tt = np.max(partial_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('partialheat',dst)
    # cv2.waitKey()

    rgb_img = im_to_torch(dst_image)
    uv = uv_to_torch_noModifyChannel(uv)
    
    data['verts'] = mesh_3d
    data['uv_flag'] = uv_flag
    data['pose_flag'] = pose_flag
    data['mask_flag'] = mask_flag
    data['fullheat'] = full_heat_inp
    data['img'] = rgb_img
    data['gt_uv'] = uv
    return data

def create_demo_data(image, lt, rb, kp_2d, occlusions=None, mask=None, is_train=False, supervise_2d=False):
    data = {}
    # image_path = 'C:\\Users\\123\Documents\Human-Training-v3.12\Human36M_MOSH\\' + image_path
    # load data

    if mask is not None:
        img_mask = cv2.imread(mask, 0)
        img_mask = cv2.threshold(img_mask, 127, 255, type=cv2.THRESH_BINARY)[1]
        mask_flag = np.array([1], dtype=np.float32)
        pose_flag = np.array([1], dtype=np.float32)
    else:
        img_mask = np.ones(image.shape[:2]) * 255
        mask_flag = np.array([0], dtype=np.float32)
        pose_flag = np.array([0], dtype=np.float32)

    if kp_2d is None or kp_2d.max() < 1:
        pose_flag = np.array([0], dtype=np.float32)
        kp_2d = np.zeros((13, 3))

    if occlusions is not None:
        i = random.randint(0,len(occlusions)-1)
        patch = cv2.imread(occlusions[i])
        patch_mask = cv2.imread(occlusions[i].replace('images', 'masks').replace('instance', 'mask'), 0)

    if is_train:
        # color adjustment
        if occlusions is not None:
            image, patch = color_gamma_contrast_patch(image, patch)
        else:
            image = color_gamma_contrast(image)
        # used for the image that the target person is not in the center
        image, img_mask, kp_2d, lt, rb, s = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=True)
        image, img_mask, kp_2d, lt, rb = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=True)
        # image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt, rb, kp_2d)
    else:
        image, img_mask, kp_2d, lt, rb, s = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=False)
        image, img_mask, kp_2d, lt, rb, offset = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=False, return_trans=True)
    
    if occlusions is not None:
        image, img_mask = synthesize_occlusion(image, patch, patch_mask, lt, rb, img_mask)

    dst_image = image
    dst_mask = img_mask

    heatmap_size = [16, 256]
    k_size = [1, 3]
    # generate full heatmap 256*256*1
    coco_kp = kp_2d
    heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(heatmap_size, k_size)]
    full_heatmaps = gen_heatmap(coco_kp, heatmap_st)
    full_heat_inp = full_heatmaps

    # #visualize occlusion uv
    # cv2.imshow('rgb', dst_image/255)
    # cv2.imshow('mask', dst_mask/255)
    # cv2.imshow('uv',(uv+0.5))
    # tt = np.max(full_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('fullheat',dst)
    # tt = np.max(partial_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('partialheat',dst)
    # cv2.waitKey()

    rgb_img = im_to_torch(dst_image)
    # data['fullheat'] = full_heat_inp
    # data['img'] = rgb_img
    return rgb_img, full_heat_inp, s, offset

def create_poseseg(image, mask, lt, rb, input_2d, gt_2d, smpl, generator, occlusions=None, is_train=False):
    data = {}
    # image_path = 'C:\\Users\\123\Documents\Human-Training-v3.12\Human36M_MOSH\\' + image_path
    # load data

    # print(image_path, lt, rb)
    if mask is not None:
        img_mask = mask * 255
        img_mask = cv2.threshold(img_mask, 127, 255, type=cv2.THRESH_BINARY)[1]
        mask_flag = np.array([1], dtype=np.float32)
        pose_flag = np.array([1], dtype=np.float32)
    else:
        img_mask = np.ones(image.shape[:2]) * 255
        mask_flag = np.array([0], dtype=np.float32)
        pose_flag = np.array([0], dtype=np.float32)

    if input_2d is None or input_2d.max() < 1:
        pose_flag = np.array([0], dtype=np.float32)
        input_2d = np.zeros((13, 3))

    if occlusions is not None:
        i = random.randint(0,len(occlusions)-1)
        patch = cv2.imread(occlusions[i])
        patch_mask = cv2.imread(occlusions[i].replace('images', 'masks').replace('instance', 'mask'), 0)

    if is_train:
        # color adjustment
        if occlusions is not None:
            image, patch = color_gamma_contrast_patch(image, patch)
        else:
            image = color_gamma_contrast(image)
        # used for the image that the target person is not in the center
        image, img_mask, input_2d, gt_2d, lt, rb, s = scale_kp(image, img_mask, input_2d, gt_2d, lt, rb, img_size=256, aug=True)
        image, img_mask, input_2d, gt_2d, lt, rb = croppad_kp(image, img_mask, input_2d, gt_2d, lt, rb, f=255, img_size=256, aug=True)
    else:
        image, img_mask, input_2d, gt_2d, lt, rb, s = scale_kp(image, img_mask, input_2d, gt_2d, lt, rb, img_size=256, aug=False)
        image, img_mask, input_2d, gt_2d, lt, rb = croppad_kp(image, img_mask, input_2d, gt_2d, lt, rb, f=255, img_size=256, aug=False)

    if occlusions is not None:
        image, img_mask = synthesize_occlusion(image, patch, patch_mask, lt, rb, img_mask)

    dst_image = image
    dst_mask = img_mask

    heatmap_size = [16, 256]
    k_size = [1, 3]
    # generate full heatmap 256*256*1
    coco_kp = input_2d
    heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(heatmap_size, k_size)]
    full_heatmaps = gen_heatmap(coco_kp, heatmap_st)
    full_heat_inp = full_heatmaps

    # generate gt partial heatmap
    kp_out = np.zeros_like(coco_kp)
    ind = 0
    for kp in gt_2d:
        if max(int(kp[1]), int(kp[0])) > 255 or min(int(kp[1]), int(kp[0]))<0:
            ind += 1
            continue
        elif dst_mask[int(kp[1]),int(kp[0])] < 127:
            ind += 1
            continue
        else:
            kp_out[ind] = kp
            ind += 1
    pheatmap_size = [16, 32, 64, 256]
    pk_size = [1, 1, 1, 3]
    heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(pheatmap_size, pk_size)]
    partial_heatmaps = gen_heatmap(kp_out, heatmap_st)
    partial_heat_inp = partial_heatmaps

    # #visualize occlusion uv
    # cv2.imshow('rgb', dst_image/255)
    # cv2.imshow('mask', dst_mask/255)
    # tt = np.max(full_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('fullheat',dst)
    # tt = np.max(partial_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('partialheat',dst)
    # cv2.waitKey()

    mask_size = [16, 32, 64, 256]
    mask_st = [mask_to_torch(cv2.resize(dst_mask, (s,s))) for s in mask_size]

    rgb_img = im_to_torch(dst_image)
    
    data['uv_flag'] = np.array([0], dtype=np.float32)
    data['pose_flag'] = pose_flag
    data['mask_flag'] = mask_flag
    data['mask'] = mask_st
    data['fullheat'] = full_heat_inp
    data['partialheat'] = partial_heat_inp
    data['img'] = rgb_img
    data['input_kp2d'] = input_2d.copy()
    data['scale'] = s

    return data

def eval_handle(image_path, lt, rb, kp_2d, intri, gt_3d, pose, shape, smpl=None, occlusions=None, is_train=False):
    data = {}

    # load data
    image = cv2.imread(image_path)
    if intri is None:
        intri = np.array([[2000,0,image.shape[0]/2], [0,2000,image.shape[1]/2], [0,0,1]])
    img_mask = np.ones(image.shape[:2])
    data['raw_img'] = image_path

    if kp_2d is None or kp_2d.max() < 1:
        kp_2d = np.zeros((13, 3))

    if occlusions is not None:
        i = random.randint(0,len(occlusions)-1)
        patch = cv2.imread(occlusions[i])
        patch_mask = cv2.imread(occlusions[i].replace('images', 'masks').replace('instance', 'mask'), 0)

    if is_train:
        # color adjustment
        if occlusions is not None:
            image, patch = color_gamma_contrast_patch(image, patch)
        else:
            image = color_gamma_contrast(image)
        # used for the image that the target person is not in the center
        image, img_mask, kp_2d, lt, rb, ratio = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=True)
        image, img_mask, kp_2d, lt, rb = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=True)
        # image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt, rb, kp_2d)
    else:
        image, img_mask, kp_2d, lt, rb, ratio = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=False)
        image, img_mask, kp_2d, lt, rb, trans = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=False, return_trans=True)
    
    if occlusions is not None:
        image, img_mask = synthesize_occlusion(image, patch, patch_mask, lt, rb, img_mask)

    dst_image = image
    dst_mask = img_mask

    heatmap_size = [16, 256]
    k_size = [1, 3]
    # generate full heatmap 256*256*1
    coco_kp = kp_2d
    heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(heatmap_size, k_size)]
    full_heatmaps = gen_heatmap(coco_kp, heatmap_st)
    full_heat_inp = full_heatmaps

    if shape is not None and pose is not None:
        shape = torch.from_numpy(shape).type(torch.float32).reshape(1, 10)
        pose = torch.from_numpy(pose).type(torch.float32).reshape(1, 72)
        transl = torch.zeros((1, 3), dtype=torch.float32)
        mesh, lsp_joints = smpl(shape, pose, transl)
        mesh_3d = mesh[0] #.numpy()
    else:
        mesh_3d = torch.zeros((1), dtype=torch.float32)

    # #visualize occlusion uv
    # cv2.imshow('rgb', dst_image/255)
    # cv2.imshow('mask', dst_mask/255)
    # cv2.imshow('uv',(uv+0.5))
    # tt = np.max(full_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('fullheat',dst)
    # tt = np.max(partial_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('partialheat',dst)
    # cv2.waitKey()

    rgb_img = im_to_torch(dst_image)

    data['intri'] = intri
    data['scale'] = ratio
    data['trans'] = trans
    data['fullheat'] = full_heat_inp
    data['img'] = rgb_img
    data['gt_3d'] = gt_3d
    data['gt_mesh'] = mesh_3d
    return data


def eval_poseseg_handle(image_path, lt, rb, kp_2d, gt_2d, mask_path, smpl=None, occlusions=None, is_train=False):
    data = {}

    # load data
    image = cv2.imread(image_path)
    img_mask = cv2.imread(mask_path, 0)

    data['raw_img'] = image_path
    data['image_id'] = int(image_path.split('\\')[-1].split('.')[0])
    data['img_shape'] = np.array(image.shape[:2])
    if kp_2d is None or kp_2d.max() < 1:
        kp_2d = np.zeros((13, 3))
    data['input_kp2d'] = kp_2d.copy()
    data['gt_kp2d'] = gt_2d.copy()

    if occlusions is not None:
        i = random.randint(0,len(occlusions)-1)
        patch = cv2.imread(occlusions[i])
        patch_mask = cv2.imread(occlusions[i].replace('images', 'masks').replace('instance', 'mask'), 0)

    if is_train:
        # color adjustment
        if occlusions is not None:
            image, patch = color_gamma_contrast_patch(image, patch)
        else:
            image = color_gamma_contrast(image)
        # used for the image that the target person is not in the center
        image, img_mask, kp_2d, lt, rb, ratio = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=True)
        image, img_mask, kp_2d, lt, rb = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=True)
        # image, img_mask, kp_2d, lt, rb = crop_target_person(image, img_mask, lt, rb, kp_2d)
    else:
        image, img_mask, kp_2d, lt, rb, ratio = scale(image, img_mask, kp_2d, lt, rb, img_size=256, aug=False)
        image, img_mask, kp_2d, lt, rb, trans = croppad(image, img_mask, kp_2d, lt, rb, f=255, img_size=256, aug=False, return_trans=True)
    
    if occlusions is not None:
        image, img_mask = synthesize_occlusion(image, patch, patch_mask, lt, rb, img_mask)

    dst_image = image
    dst_mask = img_mask

    heatmap_size = [16, 256]
    k_size = [1, 3]
    # generate full heatmap 256*256*1
    coco_kp = kp_2d
    heatmap_st = [heatmap_stand(s, s, k) for s, k in zip(heatmap_size, k_size)]
    full_heatmaps = gen_heatmap(coco_kp, heatmap_st)
    full_heat_inp = full_heatmaps

    # #visualize occlusion uv
    # cv2.imshow('rgb', dst_image/255)
    # cv2.imshow('mask', dst_mask/255)
    # # cv2.imshow('uv',(uv+0.5))
    # tt = np.max(full_heat_inp[-1], axis=0)
    # gtt = convert_color(tt*255)
    # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # cv2.imshow('fullheat',dst)
    # # tt = np.max(partial_heat_inp[-1], axis=0)
    # # gtt = convert_color(tt*255)
    # # dst = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
    # # cv2.imshow('partialheat',dst)
    # cv2.waitKey()

    rgb_img = im_to_torch(dst_image)

    data['scale'] = ratio
    data['trans'] = trans
    data['fullheat'] = full_heat_inp
    data['img'] = rgb_img
    return data
