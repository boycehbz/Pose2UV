from __future__ import absolute_import
from utils.smpl_torch_batch import *

import torch
import torch.nn as nn
import numpy as np
import scipy.misc

import cv2
from math import sqrt, isnan
from .misc import *
import math
from random import seed, gauss, random, uniform, randrange
import random
from copy import deepcopy
from utils.renderer_torch import human_render
import time
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]
# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def im_to_torch_Channel(img):
    img = img[:, :, [2, 1, 0]]
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def uv_to_torch_noModifyChannel(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def uv_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    # img = (img + 0.5) * 255
   # img *= 255
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def calc_aabb(ptSets):
    lt = np.array([ptSets[0][0], ptSets[0][1]])
    rb = lt.copy()
    for pt in ptSets:
        if pt[0] == 0 and pt[1] == 0:
            continue
        lt[0] = min(lt[0], pt[0])
        lt[1] = min(lt[1], pt[1])
        rb[0] = max(rb[0], pt[0])
        rb[1] = max(rb[1], pt[1])

    return lt, rb

def drawkp(src_image, pts):
    ## pts: [14,3] format: lsp14
    bones = [
        [0, 1, 255, 0, 0],
        [1, 2, 255, 0, 0],
        [2, 12, 255, 0, 0],
        [3, 12, 0, 0, 255],
        [3, 4, 0, 0, 255],
        [4, 5, 0, 0, 255],
        [12, 9, 0, 0, 255],
        [9, 10, 0, 0, 255],
        [10, 11, 0, 0, 255],
        [12, 8, 255, 0, 0],
        [8, 7, 255, 0, 0],
        [7, 6, 255, 0, 0],
        [12, 13, 0, 255, 0]
    ]
    for pt in pts:
        if pt[2] > 0:
            cv2.circle(src_image, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
    for line in bones:
        pa = pts[line[0]]
        pb = pts[line[1]]
        xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
        if pa[2] >= 0 and pb[2] >= 0:
            cv2.line(src_image, (xa, ya), (xb, yb), (line[2], line[3], line[4]), 2)
    cv2.imshow('kp2d',src_image/255)
    cv2.waitKey(0)

def drawkp_coco(src_image, pts):
    ## pts: [17,3] format: coco
    bones = [
        [0, 1, 255, 0, 0],
        [0, 2, 255, 0, 0],
        [1, 3, 255, 0, 0],
        [2, 4, 255, 0, 0],
        [5, 6, 255, 125, 0],
        [11, 12, 255, 125, 0],
        [5, 7, 0, 0, 255],
        [7, 9, 0, 0, 255],
        [5, 11, 0, 0, 255],
        [11, 13, 0, 0, 255],
        [13, 15, 0, 0, 255],
        [6, 8, 0, 255, 0],
        [8, 10, 0, 255, 0],
        [6, 12, 0, 255, 0],
        [12, 14, 0, 255, 0],
        [14, 16, 0, 255, 0]
    ]
    for pt in pts:
        if pt[2] > 0:
            cv2.circle(src_image, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
    for line in bones:
        pa = pts[line[0]]
        pb = pts[line[1]]
        xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
        if pa[2] > 0 and pb[2] > 0:
            cv2.line(src_image, (xa, ya), (xb, yb), (line[2], line[3], line[4]), 4)
    name = 'kp_2d'
    im = src_image
    ratiox = 300/int(im.shape[0])
    ratioy = 300/int(im.shape[1])
    ratio = ratiox if ratiox < ratioy else ratioy
    # if ratiox < ratioy:
    #     ratio = ratiox
    # else:
    #     ratio = ratioy
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    cv2.waitKey()
# def crop(img, ul, br, res, rot=0):
#     img = im_to_numpy(img)
#     # Padding so that when rotated proper amount of context is included
#     pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
#     if not rot == 0:
#         ul -= pad
#         br += pad

#     new_shape = [br[1] - ul[1], br[0] - ul[0]]
#     if len(img.shape) > 2:
#         new_shape += [img.shape[2]]
#     new_img = np.zeros(new_shape)

#     # Range to fill new array
#     new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
#     new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
#     # Range to sample from original image
#     old_x = max(0, ul[0]), min(len(img[0]), br[0])
#     old_y = max(0, ul[1]), min(len(img), br[1])
#     new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

#     if not rot == 0:
#         # Remove padding
#         new_img = scipy.misc.imrotate(new_img, rot)
#         new_img = new_img[pad:-pad, pad:-pad]

#     new_img = im_to_torch(scipy.misc.imresize(new_img, res))
#     return new_img


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def get_transform_new(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t



def trans(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform_new(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.ndimage.rotate(new_img, rot, reshape=False)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, tuple(res), interpolation=cv2.INTER_LINEAR) #scipy.misc.imresize(new_img, res)
    return new_img, ul, br, new_shape, new_x, new_y, old_x, old_y

def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(trans([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(trans([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = scipy.misc.imresize(img, crop_shape, interp='nearest')
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose

def croppad(image, mask, label, lt, rb):
    w, h, c = image.shape
    center = (rb + lt) / 2
    f = 255
    img_size = 256
    content_size = rb - lt
    pd = content_size.max()*0.0
    offset = np.array([random.uniform(-lt[0]-pd,f-rb[0]+pd), random.uniform(-lt[1]-pd,f-rb[1]+pd)])
    offlt = lt + offset
    offrb = rb + offset
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    image = cv2.warpAffine(image, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0))
    label[:, 0] = label[:, 0] + offset[0]
    label[:, 1] = label[:, 1] + offset[1]
    return image, mask, label, offlt, offrb

def color_gamma_contrast(image):
    alpha = 1.0 + uniform(-0.5, 0.5)
    beta = (1.0 - alpha)*0.5
    gamma = uniform(0.5, 1.5) #<1, brighter; >1, darker 
    image[:] = (pow(image[:]/255.0, gamma)*alpha + beta).clip(0,1) * 255.0
    return image

def color_gamma_contrast_patch(image, patch):
    alpha = 1.0 + uniform(-0.5, 0.5)
    beta = (1.0 - alpha)*0.5
    gamma = uniform(0.2, 1.5) #<1, brighter; >1, darker 
    image[:] = (pow(image[:]/255.0, gamma)*alpha + beta).clip(0,1) * 255.0
    patch[:] = (pow(patch[:]/255.0, gamma)*alpha + beta).clip(0,1) * 255.0
    return image, patch
 
def convert_color(gray):
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(gray, alpha=1),cv2.COLORMAP_JET)
    return im_color

def synthesize_occlusion(img, occlusion, mask, lt_s, rb_s, out_mask):
    # occlusion size
    lt = lt_s.copy()
    rb = rb_s.copy()
    lt = np.clip(lt, 0, max(img.shape[:2]))
    rb = np.clip(rb, 0, max(img.shape[:2]))

    human_size = (rb-lt).max()
    oc_size = np.array(occlusion.shape[:2]).max()
    ratio = (human_size * random.uniform(0.1, 0.9)) / oc_size
    sizex = int(occlusion.shape[1] * ratio)
    sizey = int(occlusion.shape[0] * ratio)
    if sizex < 5 or sizey < 5:
        return img, out_mask
    occlusion = cv2.resize(occlusion, (int(occlusion.shape[1] * ratio), int(occlusion.shape[0] * ratio)))
    mask = cv2.resize(mask, (int(mask.shape[1] * ratio), int(mask.shape[0] * ratio)))
    mask[np.where(mask<127)] = 0
    
    # occlusion position
    temp = np.zeros((img.shape[0]*3, img.shape[1]*3, 3))
    temp_mask = np.zeros((img.shape[0]*3, img.shape[1]*3))
    
    randx = img.shape[0] + random.randint(int(lt[1] - occlusion.shape[1]), int(rb[1]))
    randy = img.shape[1] + random.randint(int(lt[0] - occlusion.shape[0]), int(rb[0]))

    temp[randx:randx+occlusion.shape[0],randy:randy+occlusion.shape[1],:] = occlusion
    temp_mask[randx:randx+occlusion.shape[0],randy:randy+occlusion.shape[1]] = mask

    occlusion = temp[img.shape[0]:img.shape[0]*2,img.shape[1]:img.shape[1]*2,:]
    mask = temp_mask[img.shape[0]:img.shape[0]*2,img.shape[1]:img.shape[1]*2]

    img[np.where(mask>0)] = occlusion[np.where(mask>0)]
    out_mask[np.where(mask>0)] = 0.

    # cv2.imshow("img", img/255.)
    # cv2.imshow("mask", out_mask/255.)
    # cv2.imshow("temp", temp/255.)
    # cv2.imshow("occlusion", occlusion/255.)
    # cv2.waitKey(0)

    return img, out_mask

def scale(image, img_mask, label, lt, rb, img_size=256, aug=False):
    if aug:
        s1 = random.uniform(0.5, 0.9) # scale range
    else:
        s1 = 0.7
    h, w = image.shape[:2]
    content_size = rb - lt
    # To prevent wrong bbox
    if content_size.min() < 1 or content_size.max() > max(h, w):
        lt = np.array([0, 0], np.float32)
        rb = np.array([h, w], np.float32)
        content_size = rb - lt
    elif content_size.min() < 20:
        ratio = 20 / content_size.min()
        content_size = content_size * ratio

    s = min((img_size-1) * s1 / content_size[0], (img_size-1) * s1 / content_size[1])
    image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    img_mask = cv2.resize(img_mask, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    label[:, :2] = label[:, :2] * s
    lt[0] = int(lt[0] * s)
    lt[1] = int(lt[1] * s)
    rb[0] = int(rb[0] * s)
    rb[1] = int(rb[1] * s)
    return image, img_mask, label, lt, rb, s

def scale_kp(image, img_mask, input_2d, gt_2d, lt, rb, img_size=256, aug=False):
    if aug:
        s1 = random.uniform(0.5, 0.9) # scale range
    else:
        s1 = 0.7
    h, w = image.shape[:2]
    content_size = rb - lt
    # To prevent wrong bbox
    if content_size.min() < 1 or content_size.max() > max(h, w):
        lt = np.array([0, 0], np.float32)
        rb = np.array([h, w], np.float32)
        content_size = rb - lt
    elif content_size.min() < 20:
        ratio = 20 / content_size.min()
        content_size = content_size * ratio

    s = min((img_size-1) * s1 / content_size[0], (img_size-1) * s1 / content_size[1])
    image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    img_mask = cv2.resize(img_mask, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    input_2d[:, :2] = input_2d[:, :2] * s
    gt_2d[:, :2] = gt_2d[:, :2] * s
    lt[0] = int(lt[0] * s)
    lt[1] = int(lt[1] * s)
    rb[0] = int(rb[0] * s)
    rb[1] = int(rb[1] * s)
    return image, img_mask, input_2d, gt_2d, lt, rb, s

def random_mask(image, lt, rb):
    num_rect = random.randint(1, 4)
    mask = np.ones(image.shape[:2])
    u1 = lt[0]
    u2 = rb[0]
    v1 = lt[1]
    v2 = rb[1]

    for i in range(num_rect):
        x = np.random.randint(u1, u2, size=2)
        y = np.random.randint(v1, v2, size=2)
        # 小于5像素作为噪声
        if abs(x[0] - x[1]) < 5 and abs(x[0] - x[1]) < 5:
            continue
        else:
            mask = cv2.rectangle(mask, (x[0], y[0]), (x[1], y[1]), (0), -1)

    return mask

def mask_to_torch(img):
    img = to_torch(img).float()
    img = img.unsqueeze(dim=0)
    if img.max() > 1:
        img /= 255
    return img

def resize(image, label, cropsize):
        w, h = image.shape[:2]
        dst_image = cv2.resize(image, (cropsize, cropsize), interpolation=cv2.INTER_CUBIC)
        ratio = cropsize / w
        label[:, :2] = label[:, :2] * ratio
        return dst_image, label

def estimate_translation(S, joints_2d, focal_length=5000, img_size=256):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
   # weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, cx=128., cy=128.):
    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length, focal_length])
    # optical center
   # center = np.array([img_size/2., img_size/2.])
    center = np.array([cx, cy])
    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def rot_mesh(mesh, J3d, gt3d):
    G3d = gt3d.copy()
    J = J3d.copy()
    cent_J = np.mean(J, axis=0, keepdims=True)
    J -= cent_J
    cent_G = np.mean(G3d, axis=0, keepdims=True)
    G3d -= cent_G
    M = np.dot(J.T, G3d)
    U, D, V = np.linalg.svd(M) 
    R = np.dot(V.T, U.T)
    out_mesh = np.dot(mesh, R)
    out_joint = np.dot(J3d, R)
    return out_mesh, out_joint, R

def surface_project(vertices, exter, intri):
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(exter, temp_v)
    mesh_3d = out_point.transpose(1,0)[:,:3]
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    mesh_2d = (out_point.astype(np.int32)).transpose(1,0)
    return mesh_3d, mesh_2d


def wp_project(mesh, J3d, J2d, face, image, focal=5000.):
    f = focal
    cy = 0. #image.shape[0] / 2.
    cx = 0. #image.shape[1] / 2.
    wp_cam_intri  = np.array([[f,0,cx], [0,f,cy], [0,0,1]])
    init_extri = np.eye(4)
    j_conf = J2d[:,2] 
    gt_cam_t = estimate_translation_np(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, focal_length=f)
    init_extri = np.eye(4)
    init_extri[:3,3] = gt_cam_t
    mesh_3d, mesh_2d = surface_project(mesh, init_extri, wp_cam_intri)
    return mesh_3d, mesh_2d, gt_cam_t

def cal_cam_t(J3d, J2d, image, focal=5000.):
    f = focal
    cy = image.shape[0] / 2.
    cx = image.shape[1] / 2.
    j_conf = J2d[:,2] 
    gt_cam_t = estimate_translation_np(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy)
    return gt_cam_t

def render_mesh(mesh, J3d, J2d, face, image, focal=5000., viz=False, color='pink'):
    f = focal
    cy = image.shape[0] / 2.
    cx = image.shape[1] / 2.
    j_conf = J2d[:,2] 
    gt_cam_t = estimate_translation_np(J3d, J2d[:,:2], j_conf, focal_length=focal, cx=cx, cy=cy)
    R = torch.eye(3)[None]
    T = torch.from_numpy(gt_cam_t)[None]
    mesh = torch.from_numpy(mesh).float()
    mesh += T
    T = torch.zeros_like(T)
    face = torch.from_numpy(face.astype(np.int32))
    back, mask, img = human_render(mesh, face, R, T, f, cx, cy, image, viz=viz, mesh_color=color)
    return img




def simple_render(mesh, face, image, cam_t, focal=5000., viz=False):
    f = focal
    cy = image.shape[0] / 2.
    cx = image.shape[1] / 2.
    R = torch.eye(3)[None]
    T = torch.from_numpy(cam_t)[None]
    mesh = torch.from_numpy(mesh).float()
    face = torch.from_numpy(face.astype(np.int32))
    img, mask = human_render(mesh, face, R, T, f, cx, cy, image, viz=viz)
    return img, mask


def wp_project_viz(mesh, J3d, J2d, face, image, focal=5000.):
    f = focal
    cy = image.shape[0] / 2.
    cx = image.shape[1] / 2.
    wp_cam_intri  = np.array([[f,0,cx], [0,f,cy], [0,0,1]])
    init_extri = np.eye(4)
    j_conf = J2d[:,2] 
    gt_cam_t = estimate_translation_np(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, focal_length=f)
    init_extri = np.eye(4)
    init_extri[:3,3] = gt_cam_t
    out_point, im = surface_projection_viz(mesh, face, J3d, init_extri, wp_cam_intri, image, 1)
    return out_point, im

def surface_projection_viz(vertices, faces, joint, exter, intri, image, op):
    im = deepcopy(image)
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(exter, temp_v)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = (out_point.astype(np.int32)).transpose(1,0)
    max = dis.max()
    min = dis.min()
    t = 255./(max-min)
    img_faces = []
    i = 0
    j = 0
    for f in faces:
        i+=1
        if i%2 == 0 :
            j+=1
            continue
        if j%2 == 0 :
            continue
        #color = int((dis[f[0]] - min)*t)
        color = 255
        point = out_point[f]
        im = cv2.polylines(im,[point],True,(color,color,color),1)
    #     img_faces.append(point)
        
    # im = cv2.polylines(im,img_faces,True,(color,color,color))
    
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(exter, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)
    for i in range(len(out_point)):
        if i == op:
            im = cv2.circle(im, tuple(out_point[i]), 3, (0,0,255),-1)
        else:
            im = cv2.circle(im, tuple(out_point[i]), 3, (255,0,0),-1)

    # ratiox = 800/int(im.shape[0])
    # ratioy = 800/int(im.shape[1])
    # if ratiox < ratioy:
    #     ratio = ratiox
    # else:
    #     ratio = ratioy

    # cv2.namedWindow("mesh",0)
    # cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    # cv2.moveWindow("mesh",0,0)
    cv2.imshow('mesh',im/255.)
    cv2.waitKey()
    return out_point, im

def kp_proj(image, joint, cam_t, focal=5000., viz=False):
    f = focal
    cy = image.shape[0] / 2.
    cx = image.shape[1] / 2.
    cam_intri  = np.array([[f,0,cx], [0,f,cy], [0,0,1]])
    intri_ = np.insert(cam_intri, 3, values=0., axis=1)
    cam_extri = np.eye(4)
    cam_extri[:3,3] = cam_t
    temp_joint = np.insert(joint, 3, values=1., axis=1).transpose((1,0))
    out_point = np.dot(cam_extri, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)
    if viz:
        for i in range(len(out_point)):
            im = cv2.circle(image, tuple(out_point[i]), 3, (0,0,255),-1)
            cv2.imshow('joint_proj', im)
        cv2.waitKey()
    return out_point


# padding and resize with keypoints
def img_reshape(image, kp=None):
    w, h = image.shape[:2]
    f = max(w, h)
    if w > h: 
        M1 = np.float32([[1, 0, (w-h)/2.], [0, 1, 0]])
        kp[:,0] += (w-h)/2.
    else:
        M1 = np.float32([[1, 0, 0], [0, 1, (h-w)/2.]])
        kp[:,1] += (h-w)/2.   
    image = cv2.warpAffine(image, M1, (f, f), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    return image, kp

# image padding two sides
def padding_reshape(image):
        h, w = image.shape[:2]
        f = max(w, h)
        if h > w: 
            M1 = np.float32([[1, 0, (h-w)/2.], [0, 1, 0]])
            kp_c = 0
            offset = (h-w)/2.
        else:
            M1 = np.float32([[1, 0, 0], [0, 1, (w-h)/2.]])  
            kp_c = 1
            offset = (w-h)/2. 
        image = cv2.warpAffine(image, M1, (f, f), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
        return image, kp_c, offset

def get_norm_param(mesh_3d):
    xmin = mesh_3d[:, 0].min()
    xmax = mesh_3d[:, 0].max()
    ymin = mesh_3d[:, 1].min()
    ymax = mesh_3d[:, 1].max()
    zmin = mesh_3d[:, 2].min()
    zmax = mesh_3d[:, 2].max()
    vmin = np.array([xmin, ymin, zmin])
    vmax = np.array([xmax, ymax, zmax])
    box = (vmax-vmin).max()
    return vmin, box

def get_crop(image, bbox):
    h, w, c = image.shape
    bh = int(bbox[3]-bbox[1])
    bw = int(bbox[2]-bbox[0])
    crop_y = [0,0]
    crop_x = [0,0]
    ratio_y = 0.1
    ratio_x = 0.2
    if bbox[1] - ratio_y * bh > 0.:
        crop_y[0] = int(bbox[1] - ratio_y * bh) 
    else:
        crop_y[0] = 0
    if bbox[3] + ratio_y * bh < h:
        crop_y[1] = int(bbox[3] + ratio_y * bh) 
    else:
        crop_y[1] = h
    if bbox[0] - ratio_x  * bw > 0.:
        crop_x[0] = int(bbox[0] - ratio_x  * bw) 
    else:
        crop_x[0] = 0
    if bbox[2] + ratio_x  * bw < w:
        crop_x[1] = int(bbox[2] + ratio_x  * bw) 
    else:
        crop_x[1] = w
    return crop_x, crop_y

def draw_keyp(img, joints, color=None, format='coco17', thickness=3):
    skeletons = {'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],    [14,16],[11,12]],
            'halpe':[[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]],
            'MHHI':[[0,1],[1,2],[3,4],[4,5],[0,6],[3,6],[6,13],[13,7],[13,10],[7,8],[8,9],[10,11],[11,12]],
            'Simple_SMPL':[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]],
            'LSP':[[0,1],[1,2],[2,3],[5,4],[4,3],[3,9],[9,8],[8,2],[6,7],[7,8],[9,10],[10,11]],
            }
    colors = {'coco17':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127)],
                'halpe':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), ],
                'MHHI':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127)],
                'Simple_SMPL':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127)],
                'LSP':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127)]}

    if joints.shape[1] == 3:
        confidence = joints[:,2]
    else:
        confidence = np.ones((joints.shape[0], 1))
    joints = joints[:,:2].astype(np.int)
    for bone, c in zip(skeletons[format], colors[format]):
        if color is not None:
            c = color
        # c = (0,255,255)
        if confidence[bone[0]] > 0.1 and confidence[bone[1]] > 0.1:
            # pass
            img = cv2.line(img, tuple(joints[bone[0]]), tuple(joints[bone[1]]), c, thickness=int(thickness))
    
    for p in joints:
        img = cv2.circle(img, tuple(p), int(thickness * 5/3), c, -1)
        # vis_img('img', img)
    return img

def vis_img(name, im):
    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    cv2.waitKey()

def croppad(image, mask, label, lt, rb, f=255, img_size=256, aug=False, return_trans=False):
    h, w, c = image.shape
    center = (rb + lt) / 2
    content_size = rb - lt
    offset = np.array([img_size/2-center[0], img_size/2-center[1]])
    if aug:
        pd = content_size.max() * 0.05
        offset[0] = offset[0] + random.uniform(-pd, pd)
        offset[1] = offset[1] + random.uniform(-pd, pd)

    offlt = lt + offset
    offrb = rb + offset
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    image = cv2.warpAffine(image, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0))
    label[:, 0] = label[:, 0] + offset[0]
    label[:, 1] = label[:, 1] + offset[1]
    if return_trans:
        return image, mask, label, offlt, offrb, offset
    else:
        return image, mask, label, offlt, offrb

def croppad_kp(image, mask, input_2d, gt_2d, lt, rb, f=255, img_size=256, aug=False, return_trans=True):
    h, w, c = image.shape
    center = (rb + lt) / 2
    content_size = rb - lt
    offset = np.array([img_size/2-center[0], img_size/2-center[1]])
    if aug:
        pd = content_size.max() * 0.05
        offset[0] = offset[0] + random.uniform(-pd, pd)
        offset[1] = offset[1] + random.uniform(-pd, pd)

    offlt = lt + offset
    offrb = rb + offset
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    image = cv2.warpAffine(image, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0))
    input_2d[:, 0] = input_2d[:, 0] + offset[0]
    input_2d[:, 1] = input_2d[:, 1] + offset[1]
    gt_2d[:, 0] = gt_2d[:, 0] + offset[0]
    gt_2d[:, 1] = gt_2d[:, 1] + offset[1]

    if return_trans:
        return image, mask, input_2d, gt_2d, offlt, offrb, offset
    else:
        return image, mask, input_2d, gt_2d, offlt, offrb

def draw_keyp(img, joints, color=None, format='coco17', thickness=3):
    skeletons = {'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],    [14,16],[11,12]],
            'halpe':[[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]],
            'MHHI':[[0,1],[1,2],[3,4],[4,5],[0,6],[3,6],[6,13],[13,7],[13,10],[7,8],[8,9],[10,11],[11,12]],
            'Simple_SMPL':[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]],
            'LSP':[[0,1],[1,2],[2,3],[5,4],[4,3],[3,9],[9,8],[8,2],[6,7],[7,8],[9,10],[10,11]],
            }
    colors = {'coco17':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127)],
                'halpe':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), ],
                'MHHI':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127)],
                'Simple_SMPL':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127)],
                'LSP':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127)]}

    if joints.shape[1] == 3:
        confidence = joints[:,2]
    else:
        confidence = np.ones((joints.shape[0], 1))
    joints = joints[:,:2].astype(np.int32)
    for bone, c in zip(skeletons[format], colors[format]):
        if color is not None:
            c = color
        # c = (0,255,255)
        if confidence[bone[0]] > 0.1 and confidence[bone[1]] > 0.1:
            # pass
            img = cv2.line(img, tuple(joints[bone[0]]), tuple(joints[bone[1]]), c, thickness=int(thickness))
    
    for p in joints:
        img = cv2.circle(img, tuple(p), int(thickness * 5/3), c, -1)
        # vis_img('img', img)
    return img

def crop_target_person(rgb_img, mask, lt, rb, kp_2d):
    # size: 1.3 times of bbox
    ratio = 1.3
    center = [int((lt[0] + rb[0])/2), int((lt[1] + rb[1])/2)]
    size = int(max(rb[0]-lt[0], rb[1]-lt[1]) * ratio)
    bb_center = [int((rb[0] - lt[0])/2), int((rb[1] - lt[1])/2)]
    h_size = int(size/2)
    img_crop = np.zeros((size,size,3))
    mask_crop = np.zeros((size,size))
    
    if center[0] - h_size < 0:
        lty = 0
    else:
        lty = center[0] - h_size

    if center[1] - h_size < 0:
        ltx = 0
    else:
        ltx = center[1] - h_size

    if center[0] + h_size > rgb_img.shape[1]:
        rby = rgb_img.shape[1]
    else:
        rby = center[0] + h_size

    if center[1] + h_size > rgb_img.shape[0]:
        rbx = rgb_img.shape[0]
    else:
        rbx = center[1] + h_size
    
    # translation
    # trans = [h_size - center[0] + lty, h_size - center[1] + ltx]
    # rgb_patch = rgb_img[ltx:rbx,lty:rby,:]
    # mask_patch = mask[ltx:rbx,lty:rby]
    # img_crop[trans[1]:trans[1]+(rbx-ltx),trans[0]:trans[0]+(rby-lty),:] = rgb_patch
    # mask_crop[trans[1]:trans[1]+(rbx-ltx),trans[0]:trans[0]+(rby-lty)] = mask_patch

    trans0 = [h_size - center[0] + lty, h_size - center[1] + ltx]
    rgb_patch = rgb_img[ltx:rbx,lty:rby,:]
    mask_patch = mask[ltx:rbx,lty:rby]
    img_crop[trans0[1]:trans0[1]+(rbx-ltx),trans0[0]:trans0[0]+(rby-lty),:] = rgb_patch
    mask_crop[trans0[1]:trans0[1]+(rbx-ltx),trans0[0]:trans0[0]+(rby-lty)] = mask_patch

    trans = [center[0]-h_size,center[1]-h_size]

    lt[0] -= trans[0]
    rb[0] -= trans[0]
    lt[1] -= trans[1]
    rb[1] -= trans[1]

    kp_2d[:,0] -= trans[0]
    kp_2d[:,1] -= trans[1]

    # scale
    ratio = size / 256
    rgb_img = cv2.resize(img_crop, (int(img_crop.shape[0]/ratio), int(img_crop.shape[1]/ratio)))
    mask = cv2.resize(mask_crop, (int(mask_crop.shape[0]/ratio), int(mask_crop.shape[1]/ratio)))

    lt = lt / ratio
    rb = rb / ratio

    kp_2d[:, :2] = kp_2d[:, :2] / ratio

    # crop_param = [ltx, rbx, lty, rby, trans0[0], trans0[1], int(img_crop.shape[0]), 
    #             int(img_crop.shape[1]), int(rgb_img.shape[0]), 
    #             int(rgb_img.shape[1])]
    return rgb_img, mask, kp_2d, lt, rb#, crop_param

def crop_target_person_ratio(rgb_img, mask, lt, rb, kp_2d):
    lt = np.clip(lt, 0, max(rgb_img.shape[:2]))
    rb = np.clip(rb, 0, max(rgb_img.shape[:2]))
    # size: 1.3 times of bbox
    center = [int((lt[0] + rb[0])/2), int((lt[1] + rb[1])/2)]
    size = int(max(rb[0]-lt[0], rb[1]-lt[1]) * 1.3)

    h_size = int(size/2)
    img_crop = np.zeros((size,size,3))
    mask_crop = np.zeros((size,size))
    
    if center[0] - h_size < 0:
        lty = 0
    else:
        lty = center[0] - h_size

    if center[1] - h_size < 0:
        ltx = 0
    else:
        ltx = center[1] - h_size

    if center[0] + h_size > rgb_img.shape[1]:
        rby = rgb_img.shape[1]
    else:
        rby = center[0] + h_size

    if center[1] + h_size > rgb_img.shape[0]:
        rbx = rgb_img.shape[0]
    else:
        rbx = center[1] + h_size
    
    # translation
    trans = [h_size - center[0] + lty, h_size - center[1] + ltx]
    rgb_patch = rgb_img[ltx:rbx,lty:rby,:]
    mask_patch = mask[ltx:rbx,lty:rby]

    img_crop[trans[1]:trans[1]+(rbx-ltx),trans[0]:trans[0]+(rby-lty),:] = rgb_patch
    mask_crop[trans[1]:trans[1]+(rbx-ltx),trans[0]:trans[0]+(rby-lty)] = mask_patch

    trans = [center[0]-h_size,center[1]-h_size]

    lt[0] -= trans[0]
    rb[0] -= trans[0]
    lt[1] -= trans[1]
    rb[1] -= trans[1]

    kp_2d[:,0] -= trans[0]
    kp_2d[:,1] -= trans[1]

    # scale
    ratio = size / 256
    rgb_img = cv2.resize(img_crop, (int(img_crop.shape[0]/ratio), int(img_crop.shape[1]/ratio)))
    mask = cv2.resize(mask_crop, (int(mask_crop.shape[0]/ratio), int(mask_crop.shape[1]/ratio)))

    lt = lt / ratio
    rb = rb / ratio

    kp_2d[:, :2] = kp_2d[:, :2] / ratio

    return rgb_img, mask, kp_2d, lt ,rb, trans, ratio

def crop_target_person_vis(rgb_img, mask, lt, rb, kp_2d):
    # size: 1.3 times of bbox
    center = [int((lt[0] + rb[0])/2), int((lt[1] + rb[1])/2)]
    size = int(max(rb[0]-lt[0], rb[1]-lt[1]) * 1.3)
    bb_center = [int((rb[0] - lt[0])/2), int((rb[1] - lt[1])/2)]
    h_size = int(size/2)
    img_crop = np.zeros((size,size,3))
    mask_crop = np.zeros((size,size))
    
    if center[0] - h_size < 0:
        lty = 0
    else:
        lty = center[0] - h_size

    if center[1] - h_size < 0:
        ltx = 0
    else:
        ltx = center[1] - h_size

    if center[0] + h_size > rgb_img.shape[1]:
        rby = rgb_img.shape[1]
    else:
        rby = center[0] + h_size

    if center[1] + h_size > rgb_img.shape[0]:
        rbx = rgb_img.shape[0]
    else:
        rbx = center[1] + h_size
    
    # translation
    # trans = [h_size - center[0] + lty, h_size - center[1] + ltx]
    # rgb_patch = rgb_img[ltx:rbx,lty:rby,:]
    # mask_patch = mask[ltx:rbx,lty:rby]
    # img_crop[trans[1]:trans[1]+(rbx-ltx),trans[0]:trans[0]+(rby-lty),:] = rgb_patch
    # mask_crop[trans[1]:trans[1]+(rbx-ltx),trans[0]:trans[0]+(rby-lty)] = mask_patch

    trans0 = [h_size - center[0] + lty, h_size - center[1] + ltx]
    rgb_patch = rgb_img[ltx:rbx,lty:rby,:]
    mask_patch = mask[ltx:rbx,lty:rby]
    img_crop[trans0[1]:trans0[1]+(rbx-ltx),trans0[0]:trans0[0]+(rby-lty),:] = rgb_patch
    mask_crop[trans0[1]:trans0[1]+(rbx-ltx),trans0[0]:trans0[0]+(rby-lty)] = mask_patch

    trans = [center[0]-h_size,center[1]-h_size]

    lt[0] -= trans[0]
    rb[0] -= trans[0]
    lt[1] -= trans[1]
    rb[1] -= trans[1]

    kp_2d[:,0] -= trans[0]
    kp_2d[:,1] -= trans[1]

    # scale
    # ratio = size / 256
    # rgb_img = cv2.resize(img_crop, (int(img_crop.shape[0]/ratio), int(img_crop.shape[1]/ratio)))
    # mask = cv2.resize(mask_crop, (int(mask_crop.shape[0]/ratio), int(mask_crop.shape[1]/ratio)))

    # lt = lt / ratio
    # rb = rb / ratio

    # kp_2d[:, :2] = kp_2d[:, :2] / ratio

    crop_param = [ltx, rbx, lty, rby, trans0[0], trans0[1], int(img_crop.shape[0]), 
                int(img_crop.shape[1]), int(rgb_img.shape[0]), 
                int(rgb_img.shape[1])]
    rgb_img = img_crop
    return rgb_img, mask, kp_2d, lt, rb, crop_param

def do_crop_padding(image, mask, bbox, keypoints, cropsize, viz=False):
    crop_x, crop_y = get_crop(image, bbox)
    keypoints[:,0] -= crop_x[0]
    keypoints[:,1] -= crop_y[0]
    image = image[int(crop_y[0]):int(crop_y[1]), int(crop_x[0]):int(crop_x[1]), :]
    mask = mask[int(crop_y[0]):int(crop_y[1]), int(crop_x[0]):int(crop_x[1])]
    w, h, c = image.shape
    f = max(w, h) 
    if w > h: 
        M1 = np.float32([[1, 0, (w-h)/2.], [0, 1, 0]])
        keypoints[:,0] += (w-h)/2.
    else:
        M1 = np.float32([[1, 0, 0], [0, 1, (h-w)/2.]])
        keypoints[:,1] += (h-w)/2. 
    image = cv2.warpAffine(image, M1, (f, f), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M1, (f, f), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    image = cv2.resize(image, (cropsize, cropsize), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (cropsize, cropsize), interpolation=cv2.INTER_CUBIC)
    ratio = cropsize / f
    keypoints[:, :2] = keypoints[:, :2] * ratio
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]                 
    if viz:
        viz_img = image.copy()
        drawkp(viz_img, keypoints)
        back =np.zeros_like(viz_img)
        back[np.where(mask!=0)] = viz_img[np.where(mask!=0)]
        cv2.imshow('crop_image', back)
        cv2.imshow('crop_mask', mask)
        cv2.waitKey()
    return image, mask, keypoints

def resample_np(uv_generator, UV_map):
    new_vts = uv_generator.refine_vts
    vt_3d = [None] * new_vts.shape[0]
    resmaple_vvt = uv_generator.resample_v_to_vt
    vt_3d = UV_map[new_vts.T[0], new_vts.T[1]]
    vt_3d = np.stack(vt_3d)
    opt_v_3d = vt_3d[resmaple_vvt]
    return opt_v_3d

def resample_torch(uv_generator, UV_map, device):
    uv = UV_map.to(device)#torch.from_numpy(UV_map).to(device)
    new_vts = uv_generator.refine_vts
    resmaple_vvt = uv_generator.resample_v_to_vt
    vt_3d = UV_map[new_vts.T[0], new_vts.T[1]]
    opt_v_3d = vt_3d[resmaple_vvt].to(device) 
    return opt_v_3d

#### resmaple function and backward
class resample_mesh(torch.autograd.Function):
    @staticmethod
    def forward(self, input_uv, new_vts, resmaple_vvt, bary_weights, v_index):
        # tmp_inp = input_uv.permute(0,2,3,1)
        # vt_3d = tmp_inp[:, new_vts.T[0], new_vts.T[1]]
        # opt_v_3d = vt_3d[:, resmaple_vvt] #.permute(0,2,1)
        self.save_for_backward(bary_weights, v_index)
        opt_v_3d = input_uv[:,:, new_vts.T[0], new_vts.T[1]][:,:, resmaple_vvt] 
        return opt_v_3d

    @staticmethod
    def backward(self, grad_output):
        bary_weights, v_index = self.saved_tensors
        im = grad_output.permute(0,2,1)[:,v_index,:].to(grad_output.device) #grad_output.permute(0,2,1)[:, :, v_index].to(grad_output.device)
        bw = bary_weights[:, :, None, :].to(grad_output.device)
        grad_input = torch.matmul(bw, im).squeeze(dim=3).to(grad_output.device).permute(0,3,1,2)
        return grad_input, None, None, None, None

def resample_mesh_func(input_uv, new_vts, resmaple_vvt, bary_weights, v_index):
    return resample_mesh.apply(input_uv, new_vts, resmaple_vvt, bary_weights, v_index)

def compute_similarity_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = np.sum(X1**2)
    K = X1.dot(X2.T)
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))
    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale*(R.dot(mu1))
    S1_hat = scale*R.dot(S1) + t
    if transposed:
        S1_hat = S1_hat.T
    return S1_hat

def compute_scale(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = np.sum(X1**2)
    K = X1.dot(X2.T)
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))
    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale*(R.dot(mu1))
    S1_hat = scale * S1 + t
    if transposed:
        S1_hat = S1_hat.T
    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def compute_scale_batch(S1, S2):
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_scale(S1[i], S2[i])
    return S1_hat

def pa_error(S1, S2, reduction='mean'):
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    return re

def eval_3DPCK(pred_pose, gt_pose):
    # Parents of joints in MuPoTS joint set
    _JOINT_PARENTS = np.array([1, 2, 12, 12, 3, 4, 7, 6, 12, 12, 9, 10, 12, 12]) 
    # The order in which joints are scaled, from the hip to outer limbs
    _TRAVERSAL_ORDER = np.array([12, 13, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5]) 
    rescaled_pred_poses = pred_pose.copy()
    for ind in _TRAVERSAL_ORDER:
        parent = _JOINT_PARENTS[ind]
        gt_bone_length = np.linalg.norm(gt_pose[:, ind] - gt_pose[:, parent], axis=1)  # (nPoses,)
        pred_bone = pred_pose[:, ind] - pred_pose[:, parent]  # (nPoses, 3)
        pred_bone = pred_bone * gt_bone_length[:, np.newaxis] / \
                    (np.linalg.norm(pred_bone, axis=1, keepdims=True) + 1e-8)
        rescaled_pred_poses[:, ind] = rescaled_pred_poses[:, parent] + pred_bone
    errors = np.linalg.norm(gt_pose -  rescaled_pred_poses, axis=2)
    return errors

def error_3DPCK(pred_pose, gt_pose):
    #rescaled_pred_pose = compute_similarity_transform_batch(pred_pose, gt_pose)
    rescaled_pred_pose = compute_scale_batch(pred_pose, gt_pose)
    errors = np.linalg.norm(gt_pose -  rescaled_pred_pose, axis=2)
    return errors

def convert_to_coco_pts(lsp_pts):
    # convert lsp14 to coco17. headtop -> nose
    kp_map = [13, 13, 13, 13, 13, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
    kps = np.array(lsp_pts, dtype = np.float)[kp_map].copy()
    if lsp_pts.shape[1] > 2:
        tmp = np.nonzero(lsp_pts[12:][:,2])
        kps[:5, :2] = [-1,-1] if not len(tmp[0]) else lsp_pts[12:][tmp][:,:2].mean(axis=0) 
    else:
        kps[:5, :2] = (lsp_pts[12,:2] + lsp_pts[13,:2])/2. 
    return kps

def convert_to_coco_pts_3d(lsp_pts):
    kp_map = [13, 13, 13, 13, 13, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
    kps = np.array(lsp_pts, dtype = np.float)[kp_map].copy()
    kps[:5,:] = (lsp_pts[12,:] + lsp_pts[13,:])/2. 
    return kps

def cal_bonelength(joint_3d, type='H36M'):
    # according to kinematic tree
    if type == 'lsp':
        parents = [1,2,12,12,3,4,7,8,12,12,9,10,13]
    elif type == 'H36M':
        parents = [7,0,1,2,0,4,5,8,10,9,8,8,11,12,8,14,15]
        flip = [0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13]
    # tmp = torch.abs(joint_3d - joint_3d[:, parents])
    bone_length = torch.norm(torch.abs(joint_3d - joint_3d[:, parents]), dim=2)
    flip_bone = bone_length[:,flip] 
    return bone_length, flip_bone

def est_trans(mesh, J3d, J2d, image, focal=5000.):
    f = focal
    cy = image.shape[0] / 2.
    cx = image.shape[1] / 2.
    wp_cam_intri  = np.array([[f,0,cx], [0,f,cy], [0,0,1]])
    init_extri = np.eye(4)
    j_conf = J2d[:,2]
    gt_cam_t = estimate_translation_np(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, focal_length=focal)
    init_extri = np.eye(4)
    init_extri[:3,3] = gt_cam_t
    # mesh_proj = surface_project(mesh, init_extri, wp_cam_intri)
    return init_extri[:3,:3], gt_cam_t, wp_cam_intri

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w , h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


