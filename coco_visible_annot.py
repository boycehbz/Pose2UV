import sys
sys.path.append('./')
from utils.FileLoaders import *
import pycocotools.mask as maskUtils
import cv2
from utils.imutils import vis_img
import numpy as np
from tqdm import tqdm

coco_annot = load_json('data/person_keypoints_val2017.json')

def annToMask(segm, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    def _annToRLE(segm, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = segm
        return rle

    rle = _annToRLE(segm, height, width)
    mask = maskUtils.decode(rle)
    return mask

for ant in tqdm(coco_annot['annotations'], total=len(coco_annot['annotations'])):

    img_path = os.path.join('/media/buzhenhuang/SSD/Human-Training-v3.12/COCO_PoseSeg/images/val2017', '%012d.jpg' %ant['image_id'])
    img = cv2.imread(img_path)

    mask = ant['segmentation']
    mask = annToMask(mask, img.shape[0], img.shape[1])

    keyps = np.array(ant['keypoints']).reshape(-1, 3)

    if keyps.max() == 0:
        continue
    
    # img_all = img.copy()
    # for kp in keyps:
    #     if kp[2] == 0:
    #         continue
    #     img_all = cv2.circle(img_all, tuple(kp[:2]), 5, (0,0,255), -1)

    for i in range(len(keyps)):
        if mask[int(keyps[i][1]), int(keyps[i][0])] == 0:
            keyps[i] = 0.

    # img_vis = img.copy()
    # for kp in keyps:
    #     if kp[2] == 0:
    #         continue
    #     img_vis = cv2.circle(img_vis, tuple(kp[:2]), 5, (0,255,255), -1)

    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)*255
    # img = np.concatenate((img_all, img_vis, mask), axis=1)

    keyps = keyps.reshape(-1,).tolist()

    ant['keypoints'] = keyps

    # vis_img('img', img)

save_json('data/person_visible_keypoints_val2017.json', coco_annot)
