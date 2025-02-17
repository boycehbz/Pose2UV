import sys
import time
import os
from utils.logger import Logger, savefig
import yaml
from utils.uv_map_generator import UV_Map_Generator
from utils.smpl_torch_batch import SMPLModel
import torch
from loss_func import *
import torch.nn as nn
import torch.optim as optim
import cv2
from utils.imutils import *
from datasets.MPDataLoader import MPData
from datasets.MPEvalLoader import MPeval
from datasets.eval_dataset import MP_Eval_Data
from datasets.DemoDataLoader import DemoData
from datasets.poseseg_data import PoseSegData
from datasets.coco_v12 import COCODataset
from utils.render import Renderer
from utils.renderer_pyrd import Renderer_inp
from utils.cyclic_scheduler import CyclicLRWithRestarts
from utils.inference import get_final_preds

def seed_worker(worker_seed=7):
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    # Set a constant random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def init(note='MP', dtype=torch.float32, output='output', **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join(output, note)
    out_dir = os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the log for the current experiment
    logger = Logger(os.path.join(out_dir, 'log.txt'), title="multi-person")
    logger.set_names([note])
    logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    if not kwargs.get('eval'):
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Test Loss'])
    else:
        logger.set_names(['Name', 'MPJPE', 'PA-MPJPE', 'ABS-PCK', 'PCK', 'MPVPE', 'Accel', ])

    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./data/SMPL_NEUTRAL.pkl', 
                        data_type=dtype,
                    )
    
    # load UV generator
    generator = UV_Map_Generator(
        UV_height=256,
        UV_pickle='./data/param.pkl'
    )

    # load virtual occlusion
    if kwargs.get('virtual_mask'):
        occlusion_folder = os.path.join(kwargs.get('data_folder'), 'occlusion/images')
        occlusions = [os.path.join(occlusion_folder, k) for k in os.listdir(occlusion_folder)]
    else:
        occlusions = None

    return out_dir, logger, model_smpl, generator, occlusions

class DatasetLoader():
    def __init__(self, trainset=None, testset=None, smpl_model=None, data_folder='./data', generator=None, occlusions=None, use_mask=False, use_gt=False, poseseg=False, task=None, visible_only=False, **kwargs):
        self.data_folder = data_folder
        self.data_folder_2d = kwargs.get('data_folder2D')
        self.trainset = trainset.split(' ')
        self.testset = testset.split(' ')
        self.use_gt = use_gt
        self.poseseg = poseseg
        self.model = smpl_model
        self.generator = generator
        self.task = task
        self.visible_only = visible_only

        self.use_mask = use_mask
        self.use_dis = kwargs.get('use_dis')
        self.occlusions = occlusions
    
    def load_evalset(self):
        eval_dataset = []
        for i in range(len(self.testset)):
            eval_dataset.append(MP_Eval_Data(False, self.use_mask, self.data_folder, self.model, self.generator, self.occlusions, self.poseseg, self.testset[i]))
        
        return eval_dataset

    def load_poseseg_evalset(self):
        eval_dataset = []
        for i in range(len(self.testset)):
            eval_dataset.append(MPeval(False, self.use_mask, self.data_folder, self.model, self.generator, self.occlusions, self.poseseg, self.testset[i], self.use_gt))
        
        return eval_dataset

    def load_trainset(self):
        train_dataset = []
        for i in range(len(self.trainset)):
            if self.task == 'poseseg':
                # train_dataset.append(PoseSegData(True, self.data_folder, self.trainset[i], self.model, self.occlusions, self.generator))
                train_dataset.append(COCODataset(True, self.data_folder, self.trainset[i], self.model, self.occlusions, self.generator, self.visible_only))
            else:
                train_dataset.append(MPData(True, self.use_mask, self.data_folder, self.model, self.generator, self.occlusions, self.poseseg, self.trainset[i]))

        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        return train_dataset
    
    def load_testset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            if self.task == 'poseseg':
                # test_dataset.append(PoseSegData(False, self.data_folder, self.testset[i], self.model, self.occlusions, self.generator))
                test_dataset.append(COCODataset(False, self.data_folder, self.testset[i], self.model, self.occlusions, self.generator, self.visible_only))
            else:
                test_dataset.append(MPData(False, self.use_mask, self.data_folder, self.model, self.generator, self.occlusions, self.poseseg, self.testset[i]))

        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset

    def load_demo_data(self):

        dataset = DemoData(False, self.use_mask, self.data_folder, self.model, self.generator, self.occlusions, self.poseseg)

        return dataset

class ModelLoader():
    def __init__(self, model=None, lr=0.0001, device=torch.device('cpu'), pretrain=False, pretrain_dir='', out_dir='', smpl=None, generator=None, batchsize=32, pretrain_poseseg=False, uv_mask=None, test_loss='MPJPE', **kwargs):
        self.smpl = smpl
        self.generator = generator
        self.batchsize = batchsize
        self.output = out_dir
            
        self.J_regressor_halpe = np.load('data/J_regressor_halpe.npy')

        self.test_loss = test_loss
        if self.test_loss in ['PCK', 'mAP']:
            self.best_loss = -1
        else:
            self.best_loss = 999999999

        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)(self.generator, pretrain_poseseg)
        self.device = device
        # if uv_mask:
        self.uv_mask = cv2.imread('./data/MASK.png')
        if self.uv_mask.max() > 1:
            self.uv_mask = self.uv_mask / 255.

        print('load model: %s' %self.model_type)

        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        # load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            premodel_dict = torch.load(pretrain_dir)['model']
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print("load pretrain model")

        # Calculate model size
        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count: %.2fM' % (model_params / 1e6))

        self.optimizer = optim.Adam(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = None

    def load_scheduler(self, epoch_size):
        self.scheduler = CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=self.batchsize, epoch_size=epoch_size, restart_period=10, t_mult=2, policy="cosine", verbose=True)

    def save_best_model(self, testing_loss, epoch, task):
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        if self.test_loss in ['PCK', 'mAP']:
            if self.best_loss < testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)
        else:
            if self.best_loss > testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)

    def save_model(self, epoch, task):
        # save trained model
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        model_name = os.path.join(output, '%s_epoch%03d.pkl' %(task, epoch))
        torch.save(self.model, model_name)
        print('save model to %s' % model_name)
        # save discriminator
        if self.use_disc:
            model_name = os.path.join(output, '%s_epoch%03d.pkl' %('discriminator', epoch))
            torch.save(self.discriminator, model_name)
            print('save discriminator to %s' % model_name)

    def get_max_pred(self, heatmaps):
        num_joints = heatmaps.shape[0] # heatmap.shape : (17, 64, 48)
        width = heatmaps.shape[2] # 
        heatmaps_reshaped = heatmaps.reshape((num_joints, -1)) # reshape to Two-dimensional tensor, -1:H*W
        idx = np.argmax(heatmaps_reshaped, 1) # index：(max probability) from second dim
        maxvals = np.max(heatmaps_reshaped, 1) # value: max confidence from second dim

        maxvals = maxvals.reshape((num_joints, 1)) # reshape to two-dimensional tensor by num_joint
        idx = idx.reshape((num_joints, 1)) # reshape to two-dimensional tensor by num_joint

        preds = np.tile(idx, (1, 2)).astype(np.float32) # create two-dimensional arry (keypoint coordinates) by duplicating idx

        preds[:, 0] = (preds[:, 0]) % width # caculate x coordinate 
        preds[:, 1] = np.floor((preds[:, 1]) / width) # caculate y coordinate

        # create mask to filters out keypoint predictions with low confidence
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2)) # compare maxvals with (0.0), > True/ < False and create two-dimensional arry by duplicating the result
        pred_mask = pred_mask.astype(np.float32) # convert boolean to float

        preds *= pred_mask
        return preds, maxvals

    def heatmap_to_coord(self, hms, bbox, hms_flip=None, offset=None, scale=None, **kwargs):
        if hms_flip is not None:
            hms = (hms + hms_flip) / 2
        if not isinstance(hms,np.ndarray): # tensor to numpy
            hms = hms.cpu().data.numpy()
        coords, maxvals = self.get_max_pred(hms) 
            
        hm_h = hms.shape[1]
        hm_w = hms.shape[2]

        # post-processing: fine-tune the prediction coordinates to the direction of neighboring pixels with higher heatmap values
        for p in range(coords.shape[0]):
            hm = hms[p] # joint p's heatmap 
            px = int(round(float(coords[p][0]))) # coordinate x of joint p 
            py = int(round(float(coords[p][1]))) # coordinate y of joint p 
            if 1 < px < hm_w - 1 and 1 < py < hm_h - 1: # check coords are within the valid range
                diff = np.array((hm[py][px + 1] - hm[py][px - 1], 
                                hm[py + 1][px] - hm[py - 1][px])) # diff[0]:difference in the x direction diff[1]:difference in the y direction
                coords[p] += np.sign(diff) * .25 #sign: (+1/0/-1) ，0.25是 scale


        coords[:, :2] = coords[:, :2] - offset
        coords[:, :2] = coords[:, :2] / scale
        preds = coords
        return preds, maxvals


    def save_coco_results(self, kpt_json, data, pred):
        c = data['center'].detach().cpu().numpy()
        s = data['scale1'].detach().cpu().numpy()
        bboxes = data['bbox'].detach().cpu().numpy()

        preds, maxvals = get_final_preds(pred['preheat'][-2].clone().cpu().numpy(), c, s)

        for i, (keypoints, pose_scores, bbox) in enumerate(zip(preds, maxvals, bboxes)):

            keypoints = np.concatenate((keypoints, pose_scores), axis=1)

            data_json = {}
            data_json['bbox'] = bbox.tolist()
            data_json['image_id'] = int(data['img_id'][i])
            data_json['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
            data_json['category_id'] = 1
            data_json['keypoints'] = keypoints.reshape(-1).tolist()

            viz = False
            if viz:
                from utils.imutils import draw_keyp
                img = cv2.imread(data['img_path'][i])
                img = draw_keyp(img, keypoints, format='coco17')
                vis_img("img", img)
            
            kpt_json.append(data_json)

        return kpt_json


        # pred_heatmaps = pred['preheat'][-1].detach().cpu().numpy()
        # offsets = data['offset'].detach().cpu().numpy()
        # scales = data['scale'].detach().cpu().numpy()
        # bboxes = data['bbox'].detach().cpu().numpy()

        # for i, (heatmap, offset, scale, bbox) in enumerate(zip(pred_heatmaps, offsets, scales, bboxes)):
        #     hm_size = [heatmap.shape[1], heatmap.shape[2]]
        #     pose_coords, pose_scores = self.heatmap_to_coord(
        #         heatmap, bbox, hm_shape=hm_size, norm_type=None, hms_flip = None, offset=offset,scale=scale, )
        #     keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            
        #     data_json = {}
        #     data_json['bbox'] = bbox.tolist()
        #     data_json['image_id'] = int(data['img_id'][i])
        #     data_json['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
        #     data_json['category_id'] = 1
        #     data_json['keypoints'] = keypoints.reshape(-1).tolist()

        #     viz = False
        #     if viz:
        #         from utils.imutils import draw_keyp
        #         img = cv2.imread(data['img_path'][i])
        #         img = draw_keyp(img, keypoints, format='coco17')
        #         vis_img("img", img)
            
        #     kpt_json.append(data_json)

        # return kpt_json

    def save_poseseg_results(self, results, iter, batchsize=10):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        imgs = results['imgs'].transpose(0, 2, 3, 1)
        imgs = ((imgs * std) + mean)[...,::-1]
        input_heats = results['input_heats'].transpose(0, 2, 3, 1)
        pred_heats = results['pred_heats'].transpose(0, 2, 3, 1)
        gt_heats = results['gt_heats'].transpose(0, 2, 3, 1)
        pred_masks = results['pred_masks'] * 255
        gt_masks = results['gt_masks'] * 255

        for index, (img, input_heat, pred_heat, gt_heat, pred_mask, gt_mask) in enumerate(zip(imgs, input_heats, pred_heats, gt_heats, pred_masks, gt_masks)):
            img = img * 255
            img_h, img_w = img.shape[:2]

            input_heat = np.max(input_heat, axis=2)
            input_heat = convert_color(input_heat*255)
            input_heat = cv2.addWeighted(img.astype(np.uint8), 0.5, input_heat.astype(np.uint8),0.5,0)
            input_heat = cv2.putText(input_heat, 'input_heat', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,191,105), 1)

            pred_heat = np.max(pred_heat, axis=2)
            pred_heat = convert_color(pred_heat*255)
            pred_heat = cv2.addWeighted(img.astype(np.uint8), 0.5, pred_heat.astype(np.uint8),0.5,0)
            pred_heat = cv2.putText(pred_heat, 'pred_heat', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,191,105), 1)

            pred_mask = pred_mask.reshape(img_h, img_w)
            pred_mask = convert_color(pred_mask)
            pred_mask = cv2.addWeighted(img.astype(np.uint8), 0.5, pred_mask.astype(np.uint8),0.5,0)
            # mask_name = "%05d_pred_mask.jpg" % (iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, mask_name), pred_mask)
            
            gt_heat = np.max(gt_heat, axis=2)
            gt_heat = convert_color(gt_heat*255)
            gt_heat = cv2.addWeighted(img.astype(np.uint8), 0.5, gt_heat.astype(np.uint8),0.5,0)
            gt_heat = cv2.putText(gt_heat, 'gt_heat', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,191,105), 1)

            gt_mask = gt_mask.reshape(img_h, img_w)
            gt_mask = convert_color(gt_mask)
            gt_mask = cv2.addWeighted(img.astype(np.uint8), 0.5, gt_mask.astype(np.uint8),0.5,0)
            # mask_name = "%05d_gt_mask.jpg" % (iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, mask_name), gt_mask)

            img = np.concatenate((img, input_heat, pred_heat, gt_heat, pred_mask, gt_mask), axis=1)
            img_name = "%05d_img.jpg" % (iter * batchsize + index)
            cv2.imwrite(os.path.join(output, img_name), img)

    def save_results_smpl(self, results, iter, batchsize): 
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)
        for item in results:
            opt = results[item]
            
            for index, img in enumerate(opt):
                if item != 'pred' and item != 'uv_gt':
                    img_name = "%05d_%s.jpg" % (iter * batchsize + index, item)
                    img = img.transpose(1, 2, 0)  # H*W*C
                    img = img * 255

                # save mesh
                if item == 'pred' or item == 'uv_gt':
                    pose = torch.Tensor(img[:72]).unsqueeze(0)
                    shape = torch.Tensor(img[72:]).unsqueeze(0)
                    _trans = torch.zeros(1, 3)
                    opt_mesh, _ = self.smpl(shape, pose, _trans)
                    self.smpl.write_obj(
                        opt_mesh[0].cpu().numpy(), os.path.join(output, '%05d_%s_mesh.obj' %(iter * batchsize + index, item) )
                    )
            
                # save img
                if item == 'heatmap':
                    merge_heatmap = np.max(img/255, axis=2)
                    gtt = convert_color(merge_heatmap*255)
                    dst_image = results['rgb_img'][index].transpose(1, 2, 0) * 255
                    img = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)

                if item != 'pred' and item != 'uv_gt':
                    cv2.imwrite(os.path.join(output, img_name), img)

    def viz_input(self, input_ht=None, output_ht=None, rgb_img=None, pred=None, mask=None):
        input_ht = input_ht.detach().data.cpu().numpy().astype(np.float32)
        output_ht = output_ht.detach().data.cpu().numpy().astype(np.float32)
        rgb_img = rgb_img.detach().data.cpu().numpy().astype(np.float32)
        preds = pred.detach().data.cpu().numpy().astype(np.float32)
        mask = mask.detach().data.cpu().numpy().astype(np.float32)

        for in_ht, out_ht, img, pred, ms in zip(input_ht, output_ht, rgb_img, preds, mask):
            img = img.transpose(1,2,0)
            dst_image = img * 255
            pred = pred.transpose(1,2,0)
            pred = pred * self.uv_mask
            ms = np.clip(ms[0], 0, 1)

            in_ht = np.max(in_ht, axis=0)
            in_ht = convert_color(in_ht*255)
            in_ht = cv2.addWeighted(in_ht,0.5, dst_image.astype(np.uint8),0.5,0)

            out_ht = np.max(out_ht, axis=0)
            out_ht = convert_color(out_ht*255)
            out_ht = cv2.addWeighted(out_ht,0.5, dst_image.astype(np.uint8),0.5,0)

            cv2.imshow("in_ht",in_ht)
            cv2.imshow("out_ht",out_ht)
            cv2.imshow("pred",(pred+0.5))
            cv2.imshow("rgb_img",img)
            cv2.imshow("mask",ms)
            cv2.waitKey()

    def save_results(self, results, iter, batchsize):
        """
        object order: 
        """

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        results['gt_meshes'] = self.generator.resample_np(results['uv_gt'])
        results['pred_meshes'] = self.generator.resample_np(results['uv'])

        results['rgb_img'] = results['rgb_img'].transpose(0, 2, 3, 1)
        results['rgb_img'] = ((results['rgb_img'] * std) + mean)[...,::-1]

        heatmaps = results['heatmap'].transpose(0, 2, 3, 1)
        joint2ds = np.zeros((heatmaps.shape[0], heatmaps.shape[-1], 3))
        confidence = np.max(heatmaps, axis=(1, 2))
        confidence[np.where(confidence < 0.3)] = 0

        joint2ds[:, :, -1] = confidence
        for index, (joint2d, heatmap) in enumerate(zip(joint2ds, heatmaps)):
            for j in range(heatmap[0].shape[-1]):
                if joint2d[j][2] < 0.3:
                    continue
                joint2d[j][:2] = np.mean(np.where(heatmap[:, :, j] == joint2d[j][2]), axis=1)[::-1]

        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        for index, (img, heatmap, mask, uv, uv_gt, pred_mesh, gt_mesh) in enumerate(zip(results['rgb_img'], results['heatmap'], results['mask'], results['uv'], results['uv_gt'], results['pred_meshes'], results['gt_meshes'])):
            img = img * 255
            img_h, img_w = img.shape[:2]

            heatmap = np.max(heatmap.transpose(1,2,0), axis=2)
            heatmap = convert_color(heatmap*255)
            heatmap = cv2.addWeighted(img.astype(np.uint8), 0.5, heatmap.astype(np.uint8),0.5,0)
            heatmap = cv2.putText(heatmap, 'heatmap', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,191,105), 1)

            mask = mask.reshape(img_h, img_w)
            mask = convert_color(mask*255)
            mask = cv2.addWeighted(img.astype(np.uint8), 0.5, mask.astype(np.uint8),0.5,0)
            mask = cv2.putText(mask, 'mask', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,191,105), 1)


            uv = uv.transpose(1, 2, 0)  # H*W*C
            uv = uv * self.uv_mask
            uv = (uv + 0.5) * 255
            uv = cv2.putText(uv, 'uv', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,191,105), 1)

            uv_gt = uv_gt.transpose(1, 2, 0)  # H*W*C
            uv_gt = uv_gt * self.uv_mask
            uv_gt = (uv_gt + 0.5) * 255
            uv_gt = cv2.putText(uv_gt, 'uv_gt', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,191,105), 1)

            img = np.concatenate((img, heatmap, mask, uv, uv_gt), axis=1)

            self.smpl.write_obj(
                pred_mesh, os.path.join(output, '%05d_pred_mesh.obj' %(iter * batchsize + index) )
            )

            self.smpl.write_obj(
                gt_mesh, os.path.join(output, '%05d_gt_mesh.obj' %(iter * batchsize + index) )
            )

            img_name = "%05d_img.jpg" % (iter * batchsize + index)
            cv2.imwrite(os.path.join(output, img_name), img)

        # for item in results:
        #     opt = results[item]
            
        #     for index, img in enumerate(opt):
        #         img_name = "%05d_%s.jpg" % (iter * batchsize + index, item)
                
        #         # save mesh
        #         if item in ['gt_meshes', 'pred_meshes']:
        #             resampled_mesh = img
        #             self.smpl.write_obj(
        #                 resampled_mesh, os.path.join(output, '%05d_%s_mesh.obj' %(iter * batchsize + index, item) )
        #             )
        #             joint3ds = np.matmul(self.J_regressor_halpe, resampled_mesh)

        #             img_render = results['rgb_img'][index] * 255
        #             joint3d = joint3ds[:17]
        #             joint2d = joint2ds[index]
        #             rot, trans, intri = est_trans(resampled_mesh, joint3d, joint2d, img_render, focal=5000)
        #             render_out = self.render(resampled_mesh, self.smpl.faces, rot.copy(), trans.copy(), intri.copy(),
        #                                      img_render.copy(), color=[1, 1, 0.9])
        #             # self.render.vis_img('render', render_out)

        #             render_name = "%05d_%s_render.jpg" % (iter * batchsize + index, item)
        #             cv2.imwrite(os.path.join(output, render_name), render_out)

        #         # save img
        #         elif item in ['uv', 'uv_gt']:
        #             img = img.transpose(1, 2, 0)  # H*W*C
        #             img = img * self.uv_mask
        #             img = (img + 0.5) * 255
        #             cv2.imwrite(os.path.join(output, img_name), img)
        #         elif item == 'heatmap' or item == 'preheat':
        #             img = img.transpose(1, 2, 0)  # H*W*C
        #             merge_heatmap = np.max(img, axis=2)
        #             gtt = convert_color(merge_heatmap*255)
        #             dst_image = results['rgb_img'][index] * 255
        #             img = cv2.addWeighted(gtt,0.5, dst_image.astype(np.uint8),0.5,0)
        #             cv2.imwrite(os.path.join(output, img_name), img)
        #         elif item in ['mask']:
        #             img = img.transpose(1, 2, 0)  # H*W*C
        #             img = img * 255
        #             cv2.imwrite(os.path.join(output, img_name), img)
                    
    def save_demo_results(self, results, iter, img_path):
        """
        object order: 
        """
        results['pred_meshes'] = self.generator.resample_np(results['uv'])

        heatmaps = results['heatmap'].transpose(0, 2, 3, 1)
        joint2ds = np.zeros((heatmaps.shape[0], heatmaps.shape[-1], 3))
        confidence = np.max(heatmaps, axis=(1, 2))
        confidence[np.where(confidence < 0.3)] = 0

        img = results['img']

        joint2ds[:, :, -1] = confidence
        for index, (joint2d, heatmap, scale, center) in enumerate(zip(joint2ds, heatmaps, results['scales'], results['centers'])):
            for j in range(heatmap[0].shape[-1]):
                if joint2d[j][2] < 0.3:
                    continue
                joint2d[j][:2] = np.mean(np.where(heatmap[:, :, j] == joint2d[j][2]), axis=1)[::-1]
                joint2d[j][:2] = np.array(trans(joint2d[j][:2], center, scale, (256,256), invert=1))-1
                # joint2d[j][:2] = joint2d[j][:2] / scale

        # for j2ds in joint2ds:
        #     for j2d in j2ds[:,:2].astype(np.int):
        #         img = cv2.circle(img, tuple(j2d), 5, (0,0,255), -1)
        # vis_img('img', img)


        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        abs_meshes = []
        for mesh, j2ds in zip(results['pred_meshes'], joint2ds):

            joint3ds = np.matmul(self.J_regressor_halpe, mesh)

            img_render = img
            joint3d = joint3ds[:17]
            joint2d = j2ds
            if (joint2d[:,2] > 0).sum() < 2:
                continue
            rot, transl, intri = est_trans(mesh, joint3d, joint2d, img_render, focal=1000)

            abs_meshes.append(mesh + transl)


        render = Renderer_inp(focal_length=1000, img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl.faces)

        rendered = render.render_front_view(abs_meshes, img.copy())
        render.delete()
        # vis_img('img', rendered)

        render_name = "%s" % (img_path.replace('\\', '_').replace('/', '_'))
        cv2.imwrite(os.path.join(output, render_name), rendered)
    
    def save_pose(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)
        img = results['img']
        pre_heat = results['pre_heat']
        gt_heat = results['gt_heat']
        save_mask = False
        if results['pre_mask'] is not None:
            pre_mask = results['pre_mask']
            gt_mask = results['gt_mask']
            save_mask = True
        index = 0
        for t in range(batchsize):
            rgb = img[t].transpose(1, 2, 0) * 255
            merge_heatmap = np.max(pre_heat[t], axis=0)
            pre = convert_color(merge_heatmap*255)
            merge_heatmap = np.max(gt_heat[t], axis=0)
            gt = convert_color(merge_heatmap*255)
            gt_img = cv2.addWeighted(rgb.astype(np.uint8), 0.5, gt.astype(np.uint8),0.5,0)
            pre_img = cv2.addWeighted(rgb.astype(np.uint8), 0.5, pre.astype(np.uint8),0.5,0)
            pre_img_name = "%05d_%s.jpg" % (iter * batchsize + index, 'pre_pose')
            gt_img_name = "%05d_%s.jpg" % (iter * batchsize + index, 'gt_pose')
            cv2.imwrite(os.path.join(output, pre_img_name), pre_img)
            cv2.imwrite(os.path.join(output, gt_img_name), gt_img)
            if save_mask:
                pre_mask_name = "%05d_%s.jpg" % (iter * batchsize + index, 'pre_mask')
                gt_mask_name = "%05d_%s.jpg" % (iter * batchsize + index, 'gt_mask')
                mask_pre = pre_mask[t].transpose(1, 2, 0) * 255
                mask_gt = gt_mask[t].transpose(1, 2, 0) * 255
                cv2.imwrite(os.path.join(output, pre_mask_name), mask_pre)
                cv2.imwrite(os.path.join(output, gt_mask_name), mask_gt)
            index += 1

    def save_seg(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)
        img = results['img']
        save_mask = False
        if results['pre_mask'] is not None:
            pre_mask = results['pre_mask']
            gt_mask = results['gt_mask']
            save_mask = True
        index = 0
        for t in range(batchsize):
            rgb = img[t].transpose(1, 2, 0) * 255
            if save_mask:
                pre_mask_name = "%05d_%s.jpg" % (iter * batchsize + index, 'pre_mask')
                gt_mask_name = "%05d_%s.jpg" % (iter * batchsize + index, 'gt_mask')
                rgb_name = "%05d_%s.jpg" % (iter * batchsize + index, 'rgb')
                mask_pre = pre_mask[t].transpose(1, 2, 0) * 255
                mask_gt = gt_mask[t].transpose(1, 2, 0) * 255
                cv2.imwrite(os.path.join(output, pre_mask_name), mask_pre)
                cv2.imwrite(os.path.join(output, gt_mask_name), mask_gt)
                cv2.imwrite(os.path.join(output, rgb_name), rgb)
            index += 1

class LossLoader():
    def __init__(self, smpl, train_loss='L1', test_loss='L1', generator=None, device=torch.device('cpu'), uv_mask=False, batchsize=1, visible_only=False, **kwargs):
        self.train_loss_type = train_loss.split(' ')
        self.test_loss_type = test_loss.split(' ')
        self.smpl = smpl
        self.device = device
        self.train_loss = {}
        self.use_mask = uv_mask
        self.generator = generator
        for loss in self.train_loss_type:
            if loss == 'weight_L1':
                self.train_loss.update(w_L1=weight_L1(self.device))
            if loss == 'SMPL_Loss':
                self.train_loss.update(SMPL_Loss=SMPL_Loss(self.device, self.smpl, self.generator))
            if loss == 'Surface_smooth_Loss':
                self.train_loss.update(Surface_smooth_Loss=Surface_smooth_Loss(self.device, self.smpl.faces))
            if loss == 'L1':
                self.train_loss.update(L1=L1(self.device))
            if loss == 'partloss':
                self.train_loss.update(partloss=part_loss(generator).to(self.device))

            if loss == 'L2':
                self.train_loss.update(L2=L2(self.device))
            if loss == 'LPloss':
                self.train_loss.update(LPloss=LPloss(self.device))
            if loss == 'vtloss':
                self.train_loss.update(vtloss=vtcloss(generator).to(self.device))
            if loss == 'boneloss':            
                self.train_loss.update(boneloss=boneloss(generator).to(self.device))
            if loss == 'shapeloss':            
                self.train_loss.update(shapeloss=shapeloss(generator).to(self.device))
            if loss == 'ocheat_loss':            
                self.train_loss.update(ocheat_loss=nn.MSELoss(size_average=False).to(self.device))
            if loss == 'vaeloss':            
                self.train_loss.update(vaeloss=vaeloss('pretrain_model/vae.pkl', 64).to(self.device))
            if loss == 'MASK_L1':            
                self.train_loss.update(MASK_L1=MASK_L1(self.device))
            if loss == 'MASK_L2':            
                self.train_loss.update(MASK_L2=MASK_L2(self.device))
            if loss == 'POSE_L1':            
                self.train_loss.update(POSE_L1=POSE_L1(self.device))
            if loss == 'POSE_L2':            
                self.train_loss.update(POSE_L2=POSE_L2(self.device, visible_only))

        self.test_loss = {}
        for loss in self.test_loss_type:
            if loss == 'L1':
                self.test_loss.update(L1=L1(self.device))
            if loss == 'MPJPE':
                self.test_loss.update(MPJPE=MPJPE(generator, self.device))
            if loss == 'PA_MPJPE':
                self.test_loss.update(PA_MPJPE=MPJPE(generator, self.device))
            if loss == 'MASK_L1':
                self.test_loss.update(MASK_L1=MASK_L1(self.device))
            if loss == 'POSE_L1':
                self.test_loss.update(POSE_L1=POSE_L1(self.device))
            if loss == 'MASK_L2':
                self.test_loss.update(MASK_L2=MASK_L2(self.device))
            if loss == 'POSE_L2':
                self.test_loss.update(POSE_L2=POSE_L2(self.device))

        self.uv_mask = cv2.imread('./data/MASK.png')
        uv_mask = uv_to_torch_noModifyChannel(self.uv_mask).unsqueeze(0)
        self.uv_mask = uv_mask.to(device)


    def calcul_trainloss(self, pred, data):
        loss_dict = {}

        data['uv_flag'] = data['uv_flag'].squeeze(-1)
        data['pose_flag'] = data['pose_flag'].squeeze(-1)
        data['mask_flag'] = data['mask_flag'].squeeze(-1)

        if self.use_mask and 'pred_uv' in pred.keys():
            pred['pred_uv'] = pred['pred_uv'] * self.uv_mask

        for ltype in self.train_loss:
            if ltype == 'w_L1':
                loss_dict.update(w_L1=self.train_loss['w_L1'](pred['pred_uv'], data['gt_uv'], data['uv_flag']))
            elif ltype == 'SMPL_Loss':
                SMPL_loss = self.train_loss['SMPL_Loss'](pred['pred_verts'], data['verts'], data['uv_flag'])
                loss_dict = {**loss_dict, **SMPL_loss}
            elif ltype == 'Surface_smooth_Loss':
                Surface_smooth_Loss = self.train_loss['Surface_smooth_Loss'](pred['pred_verts'])
                loss_dict = {**loss_dict, **Surface_smooth_Loss}
            elif ltype == 'MASK_L1':
                loss_dict.update(MASK_L1=self.train_loss['MASK_L1'](pred['premask'], data['mask'], data['mask_flag']))
            elif ltype == 'POSE_L1':
                loss_dict.update(POSE_L1=self.train_loss['POSE_L1'](pred['preheat'], data['partialheat'], data['pose_flag']))
            elif ltype == 'MASK_L2':
                loss_dict.update(MASK_L2=self.train_loss['MASK_L2'](pred['premask'], data['mask'], data['mask_flag']))
            elif ltype == 'POSE_L2':
                loss_dict.update(POSE_L2=self.train_loss['POSE_L2'](pred['preheat'], data['gt_heat'], data['pose_flag'], data['vis']))
            else:
                pass

        loss = 0
        for k in loss_dict:
            loss_temp = loss_dict[k] * 60.
            loss += loss_temp
            loss_dict[k] = round(float(loss_temp.detach().cpu().numpy()), 3)
        return loss, loss_dict

    def calcul_testloss(self, pred, data):

        data['uv_flag'] = data['uv_flag'].squeeze(-1)
        data['pose_flag'] = data['pose_flag'].squeeze(-1)
        data['mask_flag'] = data['mask_flag'].squeeze(-1)

        if self.use_mask and 'pred_uv' in pred.keys():
            pred['pred_uv'] = pred['pred_uv'] * self.uv_mask

        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'MPJPE':
                loss_dict.update(MPJPE=self.test_loss['MPJPE'](pred['pred_joints'], data['gt_3d']))
            elif ltype == 'PA_MPJPE':
                loss_dict.update(PA_MPJPE=self.test_loss['PA_MPJPE'].pa_mpjpe(pred['pred_joints'], data['gt_3d']))
            elif ltype == 'MASK_L1':
                loss_dict.update(MASK_L1=self.test_loss['MASK_L1'](pred['premask'], data['mask'], data['mask_flag']))
            elif ltype == 'POSE_L1':
                loss_dict.update(POSE_L1=self.test_loss['POSE_L1'](pred['preheat'], data['partialheat'], data['pose_flag']))
            elif ltype == 'MASK_L2':
                loss_dict.update(MASK_L2=self.test_loss['MASK_L2'](pred['premask'], data['mask'], data['mask_flag']))
            elif ltype == 'POSE_L2':
                loss_dict.update(POSE_L2=self.test_loss['POSE_L2'](pred['preheat'], data['partialheat'], data['pose_flag']))
            else:
                print('The specified loss: %s does not exist' % ltype)
                pass

        loss = 0
        for k in loss_dict:
            loss += loss_dict[k]
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 3)
        return loss, loss_dict
