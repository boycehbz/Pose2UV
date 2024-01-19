#coding:utf-8

import cv2
import numpy as np
import math
from copy import deepcopy
import time
# from global_vars import *
#import matplotlib.pyplot as plt

#相机的旋转和平移，xyz的角度，xyz的平移
rotation    = np.array([0, 0, 0])
translation = np.array([0, 0, -10])


def grobal_trans(g):
    tem_g = g
    grobal1 = tem_g[:9].reshape(3,3)
    trans = tem_g[9:].reshape(3,1)
    grobal1 = np.c_[grobal1,trans]
    tem = np.array([0.,0.,0.,1.])
    grobal1 = np.vstack((grobal1,tem))

    return grobal1

#参数： obj（txt）文件路径, 相机旋转， 相机平移， 是否显示。
def get_projection(obj, g, inter, viz, index, image):
    im = cv2.imread(image)
    #im = np.zeros((im.shape[0],im.shape[1]))
    #世界坐标原点
    point_zero = np.array([0,0,0,1])
    point_one = np.array([1,1,1,1])

    inter_ = np.insert(inter,3,values=0.,axis=1)

    outer = g

    f = open(obj,'r')
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if words[0] == 'v':
            t = np.array([float(words[1]),-float(words[3]),float(words[2]),1.])
            out_point = np.dot(outer, t.T)
            dis = out_point[2]
            out_point = np.dot(inter_, out_point)/dis

            if 0<=int(out_point[1])<im.shape[0] and 0<=int(out_point[0])<im.shape[1] :
                im[int(out_point[1]),int(out_point[0])]=[0,255,255]

    if viz:
        road = base_dir + 'projection/' + str(index) +'.jpg'
        cv2.imwrite(road,im)

def get_cam_pose(pose_file):
    line_num = 0
    grobal = []
    f = open(pose_file,'r')
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if line_num != 0 and len(words)>1:
            grobal.append(words)
        else:
            frame_num = int(words[0])
            line_num +=1
            
    grobal=np.array(grobal,dtype=float)
    return frame_num, grobal

def surface_projection(vertices, faces, joint, exter, intri, image, op):
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

def joint_projection(joint, extri, intri, image, viz=False):

    im = deepcopy(image)

    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(extri, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = out_point.transpose(1,0)

    if viz:
        viz_point = out_point.copy().astype(np.int32)
        for i in range(len(viz_point)):
            im = cv2.circle(im, tuple(viz_point[i]), 2, (0,0,255),-1)


        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh",0)
        cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("mesh",0,0)
        cv2.imshow('mesh',im/255.)
        cv2.waitKey()

    return out_point, im


if __name__ == "__main__":
    def load_IKE_campose(file):
        campose = []
        intra = []
        campose_ = []
        intra_ = []
        f = open(file,'r')
        for line in f:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if len(words) == 3:
                intra_.append([float(words[0]),float(words[1]),float(words[2])])
            elif len(words) == 4:
                campose_.append([float(words[0]),float(words[1]),float(words[2]),float(words[3])])
            else:
                pass

        index = 0
        intra_t = []
        for i in intra_:
            index+=1
            intra_t.append(i)
            if index == 3:
                index = 0
                intra.append(intra_t)
                intra_t = []

        index = 0
        campose_t = []
        for i in campose_:
            index+=1
            campose_t.append(i)
            if index == 3:
                index = 0
                campose_t.append([0.,0.,0.,1.])
                campose.append(campose_t)
                campose_t = []
        
        return np.array(campose), np.array(intra)

    campose, intri = load_IKE_campose('/home/hbz/vclo_process/smplify_public/camera_para.txt')
    d3p = np.array([[8.5778,70.3073,11.2463,1.0],
    [10,10,0,1.0],
    [-8.0,-0.26,0,1.],
    [-22.9,2.79876,-7.0932,1.]])
    d2p = np.array([[100,100]])

    g = np.linalg.inv(campose[0])
    
    intr = intri[0]

    joint_projection(d2p, d3p, g, intr, True, 0, 0, '/home/hbz/vclo_process/smplify_public/0000/Camera00/00000.jpg')