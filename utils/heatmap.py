import numpy as np
import math
import cv2

# generate the ground truth heatmap in the center
def heatmap_stand(width, height, sigma):
    heatmap = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            exponent = 0.5 * (pow(i - height / 2, 2.0) + pow(j - width / 2, 2.0)) * pow(1 / sigma, 2.0)
            if exponent > 4.6052: continue
            heatmap[i,j] = min(math.exp(-exponent), 1.0)
    
   # viz_map = np.zeros((height, width, 3), dtype=np.float32) + 0.2
#    # for i in range(3):
#     viz_map[:,:,0] = heatmap 
#     cv2.imwrite('tt.jpg', viz_map*255)
    #cv2.waitKey()
    return heatmap

def gen_heatmap(label, heatmap_st, label_format='coco_13'):
    
    if label_format == 'lsp':
        num = 14
    elif label_format == 'coco':
        num = 17
    elif label_format == 'coco_13':
        num = 13
    # generate the heatmap
    total_valid_joint = []
    total_heatmap = []
    for heatmap_i in heatmap_st:

        # empty list
        heatmap=np.zeros((num, heatmap_i.shape[0], heatmap_i.shape[1]), dtype=np.float32)
        valid_joint = []

        # image ratio
        ratio_x = 256/heatmap_i.shape[1]
        ratio_y = 256/heatmap_i.shape[0]

        # keypoints heatmap
        start_idx = 0
        for i in range(label.shape[0]):
            # has_joint = False
            
            # fix heatmap generation bug
            if (label[i]<0).any(): continue
            offset_x = label[i,0]/ratio_x - heatmap_i.shape[1]/2
            offset_y = label[i,1]/ratio_y - heatmap_i.shape[0]/2
            M = np.float32([[1, 0, offset_x],[0, 1, offset_y]])
            dst = cv2.warpAffine(heatmap_i, M, heatmap_i.shape, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            heatmap[i] = cv2.max(heatmap[i], dst) * label[i,2]
            # cv2.imshow('2', heatmap[i])
            # cv2.waitKey()
            # has_joint = True
            # if vis_hand_count==0: 
            #     valid_joint.append(int(i))
            # else:
            #     if has_joint: valid_joint.append(int(i))
        total_heatmap.append(heatmap)
        # # region map
        # start_idx = 14 ### after the joint heatmap
        # valid_joint.append(int(start_idx))
        # valid_joint.append(int(start_idx+1))
        # valid_joint.append(int(start_idx+2))
        # for hand_idx in range(hand_count):
        #     if (label[:,2+hand_idx*2:2+(hand_idx+1)*2]<0).any(): continue
        #     rect_size = label[:,3+hand_idx*2] - label[:,2+hand_idx*2]
        #     center = label[:,2+hand_idx*2] + rect_size*0.5
        #     # center 
        #     offset_x = center[0]/ratio_x - heatmap_i.shape[1] / 2
        #     offset_y = center[1]/ratio_y - heatmap_i.shape[0] / 2
        #     M = np.float32([[1, 0, offset_x],[0, 1, offset_y]])
        #     dst = cv2.warpAffine(heatmap_i, M, heatmap_i.shape, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        #     heatmap[start_idx] = cv2.max(heatmap[start_idx], dst)

        #     # rect
        #     ratio_w = rect_size[0]/TRAIN_IMAGE_WIDTH; ratio_h = rect_size[1]/TRAIN_IMAGE_HEIGHT
        #     center_x = center[0]/ratio_x; center_y = center[1]/ratio_y
        #     for y in range(max(int(center_y-4),0), min(int(center_y+5), heatmap_i.shape[0])):
        #         for x in range(max(int(center_x-4),0), min(int(center_x+5), heatmap_i.shape[1])):
        #             heatmap[start_idx+1, y, x] = max(heatmap[start_idx+1, y, x],ratio_w)
        #             heatmap[start_idx+2, y, x] = max(heatmap[start_idx+2, y, x],ratio_h)

        # total_heatmap.append(heatmap)
        # total_valid_joint.append(np.array(valid_joint))

    return total_heatmap
# import numpy as np
# import math
# import cv2

# # generate the ground truth heatmap in the center
# def heatmap_stand(width, height, sigma):
#     heatmap = np.zeros((height, width), dtype=np.float32)
#     for i in range(height):
#         for j in range(width):
#             exponent = 0.5 * (pow(i - height / 2, 2.0) + pow(j - width / 2, 2.0)) * pow(1 / sigma, 2.0)
#             if exponent > 4.6052: continue
#             heatmap[i,j] = min(math.exp(-exponent), 1.0)
    
#    # viz_map = np.zeros((height, width, 3), dtype=np.float32) + 0.2
# #    # for i in range(3):
# #     viz_map[:,:,0] = heatmap 
# #     cv2.imwrite('tt.jpg', viz_map*255)
#     #cv2.waitKey()
#     return heatmap

# def gen_heatmap(label, heatmap_st, label_format='lsp'):
    
#     if label_format == 'lsp':
#         num = 14
#     elif label_format == 'coco':
#         num = 17
#     elif label_format == 'coco_13':
#         num = 13
#     # generate the heatmap
#     total_valid_joint = []
#     total_heatmap = []
#     for heatmap_i in heatmap_st:

#         # empty list
#         heatmap=np.zeros((num, heatmap_i.shape[0], heatmap_i.shape[1]), dtype=np.float32)
#         valid_joint = []

#         # image ratio
#         ratio_x = 256/heatmap_i.shape[1]
#         ratio_y = 256/heatmap_i.shape[0]

#         # keypoints heatmap
#         start_idx = 0
#         for i in range(label.shape[0]):
#             # has_joint = False
            
#             # fix heatmap generation bug
#             if (label[i]<0).any(): continue
#             offset_x = label[i,0]/ratio_x - heatmap_i.shape[1]/2
#             offset_y = label[i,1]/ratio_y - heatmap_i.shape[0]/2
#             M = np.float32([[1, 0, offset_x],[0, 1, offset_y]])
#             dst = cv2.warpAffine(heatmap_i, M, heatmap_i.shape, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
#             heatmap[i] = cv2.max(heatmap[i], dst)
#             # cv2.imshow(heatmap[i])
#             # cv2.waitKey()
#             # has_joint = True
#             # if vis_hand_count==0: 
#             #     valid_joint.append(int(i))
#             # else:
#             #     if has_joint: valid_joint.append(int(i))
#         total_heatmap.append(heatmap)
#         # # region map
#         # start_idx = 14 ### after the joint heatmap
#         # valid_joint.append(int(start_idx))
#         # valid_joint.append(int(start_idx+1))
#         # valid_joint.append(int(start_idx+2))
#         # for hand_idx in range(hand_count):
#         #     if (label[:,2+hand_idx*2:2+(hand_idx+1)*2]<0).any(): continue
#         #     rect_size = label[:,3+hand_idx*2] - label[:,2+hand_idx*2]
#         #     center = label[:,2+hand_idx*2] + rect_size*0.5
#         #     # center 
#         #     offset_x = center[0]/ratio_x - heatmap_i.shape[1] / 2
#         #     offset_y = center[1]/ratio_y - heatmap_i.shape[0] / 2
#         #     M = np.float32([[1, 0, offset_x],[0, 1, offset_y]])
#         #     dst = cv2.warpAffine(heatmap_i, M, heatmap_i.shape, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
#         #     heatmap[start_idx] = cv2.max(heatmap[start_idx], dst)

#         #     # rect
#         #     ratio_w = rect_size[0]/TRAIN_IMAGE_WIDTH; ratio_h = rect_size[1]/TRAIN_IMAGE_HEIGHT
#         #     center_x = center[0]/ratio_x; center_y = center[1]/ratio_y
#         #     for y in range(max(int(center_y-4),0), min(int(center_y+5), heatmap_i.shape[0])):
#         #         for x in range(max(int(center_x-4),0), min(int(center_x+5), heatmap_i.shape[1])):
#         #             heatmap[start_idx+1, y, x] = max(heatmap[start_idx+1, y, x],ratio_w)
#         #             heatmap[start_idx+2, y, x] = max(heatmap[start_idx+2, y, x],ratio_h)

#         # total_heatmap.append(heatmap)
#         # total_valid_joint.append(np.array(valid_joint))

#     return total_heatmap