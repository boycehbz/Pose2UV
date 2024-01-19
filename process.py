import torch
import sys
from torch.autograd import Variable
from utils.imutils import *
import numpy as np
import os
import cv2
import sys
import gc
import torch.nn.functional as F
from tqdm import tqdm

def to_device(data, device):
    temp = {}
    if 'mask' in data.keys():
        temp['mask'] = [item.to(device) for item in data['mask']]
    if 'fullheat' in data.keys():
        temp['fullheat'] = [item.to(device) for item in data['fullheat']]
    if 'partialheat' in data.keys():
        temp['partialheat'] = [item.to(device) for item in data['partialheat']]

    data = {k:v.to(device).float() for k, v in data.items() if k not in ['mask', 'fullheat', 'partialheat']}

    data = {**temp, **data}

    return data

def viz_poseseg(pred_hm=None, gt_hm=None, pred_ms=None, gt_ms=None, img=None):
    pred_hm = pred_hm.detach().data.cpu().numpy().astype(np.float32)
    gt_hm = gt_hm.detach().data.cpu().numpy().astype(np.float32)
    pred_ms = pred_ms.detach().data.cpu().numpy().astype(np.float32)
    gt_ms = gt_ms.detach().data.cpu().numpy().astype(np.float32)
    img = img.detach().data.cpu().numpy().astype(np.float32)
    for phm, ghm, pms, gms, im in zip(pred_hm, gt_hm, pred_ms, gt_ms, img):
        im = im.transpose((1,2,0)) 
        pms = pms[0]  
        gms = gms[0]  

        for p_kp, g_kp in zip(phm, ghm):
            scale = p_kp.shape[0] / g_kp.shape[0] 
            g_kp = cv2.resize(g_kp, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # if p_kp.max() > 0.3:
            #     p_kp = np.mean(np.where(p_kp == np.max(p_kp)), axis=1).astype(np.int64)
            #     im = cv2.circle(im, (p_kp[1], p_kp[0]), 2, (0,0,255),-1)

            # if g_kp.max() > 0.3:
            #     g_kp = cv2.resize(g_kp, (256,256),interpolation=cv2.INTER_CUBIC)
            #     g_kp = np.mean(np.where(g_kp == np.max(g_kp)), axis=1).astype(np.int64)
            #     im = cv2.circle(im, (g_kp[1], g_kp[0]), 2, (0,255,0),-1)

        cv2.imshow("img", im)
        # cv2.imshow("p_mask",pms/255)
        cv2.imshow("g_mask",gms)
        cv2.waitKey()


def viz_masks(m0, m1, m2, m3, mask):
    m_0 = m0.detach().data.cpu().numpy().astype(np.float32)
    m_1 = m1.detach().data.cpu().numpy().astype(np.float32)
    m_2 = m2.detach().data.cpu().numpy().astype(np.float32)
    m_3 = m3.detach().data.cpu().numpy().astype(np.float32)
    mask_viz = mask.detach().data.cpu().numpy().astype(np.float32)
    for m0, m1, m2, m3, mask in zip(m_0, m_1, m_2, m_3, mask_viz):

        m0 = m0.transpose(1,2,0)
        m1 = m1.transpose(1,2,0)
        m2 = m2.transpose(1,2,0)
        m3 = m3.transpose(1,2,0)
        mask = mask.transpose(1,2,0)

        cv2.imshow("m0",m0)
        cv2.imshow("m1",m1)
        cv2.imshow("m2",m2)
        cv2.imshow("m3",m3)
        cv2.imshow("mask",mask)
        cv2.waitKey()

def poseseg_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['img'].size(0)
        data = to_device(data, device)

        pred = model.model(data)

        loss, loss_dict = loss_func.calcul_trainloss(pred, data)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # optimize
        model.optimizer.step()
        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), loss_dict)
        train_loss += loss_batch
    return train_loss/len_data

def poseseg_test(model, loss_func, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['img'].size(0)
            data = to_device(data, device)
            
            # forward
            pred = model.model(data)

            # calculate loss
            loss, loss_dict = loss_func.calcul_testloss(pred, data)

            # save results
            if i < 1:
                results = {}
                results.update(imgs=data['img'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_heats=pred['preheat'][-1].detach().cpu().numpy().astype(np.float32))
                results.update(pred_masks=pred['premask'][-1].detach().cpu().numpy().astype(np.float32))
                results.update(gt_heats=data['partialheat'][-1].detach().cpu().numpy().astype(np.float32))
                results.update(gt_masks=data['mask'][-1].detach().cpu().numpy().astype(np.float32))
                model.save_poseseg_results(results, i, batchsize)
                
            loss_batch = loss.detach()
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), loss_dict)
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def posenet_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['img'].size(0)
        if torch.cuda.is_available():
            hmgt = [Variable(item).to(device) for item in data['heatmaps']]
            img = Variable(data['img']).to(device)
            crop = Variable(data['crop']).to(device)
        else:
            print('CUDA error')
            sys.exit(0)

        # forward
        output = model.model(crop)

        # calculate loss
        loss = loss_func.calcul_heatmaploss(output, hmgt)
        # visualize
        if viz:
            # viz_poseseg(pred_hm=output[3], gt_hm=hmgt[2], pred_ms=output[9][:,14,:,:], gt_ms=data['mask'], img=img)

            test = output[3].detach().cpu().numpy().astype(np.float32)
            test_img = img.detach().cpu().numpy().astype(np.float32)
            gt = hmgt[2].detach().cpu().numpy().astype(np.float32)
            test_img = test_img[0].transpose((1,2,0))
            vis_img("img", test_img)
            for t in range(14):
                temp = convert_color(test[0][t]*255)
                gtt = convert_color(gt[0][t]*255)
                vis_img("hm", temp)
                vis_img("gt", gtt)

        # backward
        model.optimizer.zero_grad()
        loss.backward()
        # optimize
        model.optimizer.step()

        loss_batch = loss.detach() / batchsize
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch))
        train_loss += loss_batch

    return train_loss/len_data

def posenet_test(model, loss_func, loader, viz=False, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                hmgt = [Variable(item).to(device) for item in data['heatmaps']]
                img = Variable(data['img']).to(device)
                crop = Variable(data['crop']).to(device)
            else:
                print('CUDA error')
                sys.exit(0)

            # forward
            output = model.model(crop)

            # calculate loss
            loss = loss_func.calcul_heatmaploss(output, hmgt)
            # visualize
            if viz:
                # viz_poseseg(pred_hm=output[8], gt_hm=hmgt[2], pred_ms=output[9][:,14,:,:], gt_ms=data['mask'], img=img)

                test = output[3].detach().cpu().numpy().astype(np.float32)
                test_img = crop.detach().cpu().numpy().astype(np.float32)
                gt = hmgt[2].detach().cpu().numpy().astype(np.float32)
                test_img = test_img[0].transpose((1,2,0))
                vis_img("img", test_img)
                for t in range(14):
                    temp = convert_color(test[0][t]*255)
                    gtt = convert_color(gt[0][t]*255)
                    vis_img("hm", temp)
                    vis_img("gt", gtt)

                #viz_masks(m0, m1, m2, m3, mask, mask1)
            # save results
            if i < 0:
                results = {}
                results.update(img=img.detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)
                
            loss_batch = loss.detach() / batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch))
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def segnet_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['img'].size(0)
        if torch.cuda.is_available():
            msgt = [Variable(item).to(device) for item in data['masks']]
            full_hm = [Variable(item).to(device) for item in data['full_heatmaps']]
            img = Variable(data['img']).to(device)
            # oc_index = Variable(data['oc_index']).to(device)
        else:
            print('CUDA error')
            sys.exit(0)

        # forward
        output = model.model(img, full_hm) #img,crop

        # calculate loss
        loss = loss_func.calcul_segloss(output, msgt)

        # backward
        model.optimizer.zero_grad()
        loss.backward()
        # optimize
        model.optimizer.step()
        loss_batch = loss.detach() / batchsize
        print('epoch：%d/%d, batch：%d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch))
        train_loss += loss_batch
    return train_loss/len_data

def segnet_test(model, loss_func, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                msgt = [Variable(item).to(device) for item in data['masks']]
                full_hm = [Variable(item).to(device) for item in data['full_heatmaps']]
                img = Variable(data['img']).to(device)
            else:
                print('CUDA error')
                sys.exit(0)

            # forward
            output = model.model(img, full_hm)

            # calculate loss
            loss = loss_func.calcul_segloss(output, msgt)

            # save results
            if i < 5:
                results = {}
                results.update(img=img.detach().cpu().numpy().astype(np.float32))
                results.update(pre_mask=output[4].detach().cpu().numpy().astype(np.float32))
                results.update(gt_mask=data['mask'].detach().cpu().numpy().astype(np.float32))
                model.save_seg(results, i, batchsize)
            loss_batch = loss.detach() / batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch))
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all


def segnet_uv_vae_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['img'].size(0)
        data = to_device(data, device)

        output = model.model(data)

        loss, loss_dict = loss_func.calcul_trainloss(output, data)

        # visualize
        if viz:
            model.viz_input(input_ht=data['fullheat'][-1], output_ht=output['heatmap'], rgb_img=data['img'], pred=output['pred_uv'], mask=output['pred_mask'][-1])
        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # optimize
        model.optimizer.step()
        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), loss_dict)
        train_loss += loss_batch
    return train_loss/len_data

def segnet_uv_vae_test(model, loss_func, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['img'].size(0)
            data = to_device(data, device)

            # forward
            output = model.model(data)
            
            # calculate loss
            loss, loss_dict = loss_func.calcul_testloss(output, data)

            # save results
            if i < 4:
                results = {}
                results.update(mask=output['mask'].detach().cpu().numpy().astype(np.float32))
                results.update(heatmap=output['heatmap'].detach().cpu().numpy().astype(np.float32))
                results.update(uv=output['pred_uv'].detach().cpu().numpy().astype(np.float32))
                results.update(uv_gt=data['gt_uv'].detach().cpu().numpy().astype(np.float32))
                results.update(rgb_img=data['img'].detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)
                
            loss_batch = loss.detach()
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), loss_dict)
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def demo(model, yolox_predictor, alpha_predictor, loader, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, img_path in tqdm(enumerate(loader), total=len(loader)):
            img = cv2.imread(img_path)

            det, _ = yolox_predictor.predict(img_path, viz=False)
            poses = alpha_predictor.predict(img, det['bbox'])

            # alpha_predictor.visualize(img, poses, viz=False)

            data = loader.prepare(img, det['bbox'], poses, device)

            # forward
            output = model.model(data)

            # save results
            results = {}
            results.update(scales=data['scale'].astype(np.float32))
            results.update(offsets=data['offset'].astype(np.float32))
            results.update(heatmap=output['heatmap'].detach().cpu().numpy().astype(np.float32))
            results.update(uv=output['pred_uv'].detach().cpu().numpy().astype(np.float32))
            results.update(img=img)
            model.save_demo_results(results, i, img_path)
                

def EvalModel(model, evaltool, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'evaluation' + '-' * 10)
    abs_errors, errors, error_pas, abs_pcks, pcks, imnames, joints, joints_2ds, vertex_errors = [], [], [], [], [], [], [], [], []
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                rgb_img = Variable(data['img']).to(device)
                full_hm = [Variable(item).to(device) for item in data['fullheat']]
            else:
                print('CUDA error')
                sys.exit(0)
            
            # forward
            output = model.model(rgb_img, full_hm)
            abs_error, error, error_pa, abs_pck, pck, imname, joint, joints_2d, vertex_error = evaltool(output, data)
            abs_errors += abs_error
            errors += error
            error_pas += error_pa
            abs_pcks += abs_pck
            pcks += pck
            imnames += imname
            joints += joint
            joints_2ds += joints_2d
            vertex_errors += vertex_error

            # # save results
            # if i < 4:
            #     results = {}
            #     results.update(mask=output['mask'].detach().cpu().numpy().astype(np.float32))
            #     results.update(heatmap=output['heatmap'].detach().cpu().numpy().astype(np.float32))
            #     results.update(pred=output['decoded'].detach().cpu().numpy().astype(np.float32))
            #     results.update(uv_gt=uv_gt.detach().cpu().numpy().astype(np.float32))
            #     results.update(rgb_img=rgb_img.detach().cpu().numpy().astype(np.float32))
            #     model.save_results(results, i, batchsize)
        
        abs_error = np.mean(np.array(abs_errors))
        error = np.mean(np.array(errors))
        error_pa = np.mean(np.array(error_pas))
        abs_pck = np.mean(np.array(abs_pcks))
        pck = np.mean(np.array(pcks))
        vertex_error = np.mean(np.array(vertex_errors))
        return abs_error, error, error_pa, abs_pck, pck, imnames, joints, joints_2ds, vertex_error

def EvalPoseSeg(model, evaltool, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'evaluation' + '-' * 10)
    seg_results, alpha_mpjpes, pred_mpjpes= [], [], []
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                rgb_img = Variable(data['img']).to(device)
                full_hm = [Variable(item).to(device) for item in data['fullheat']]
            else:
                print('CUDA error')
                sys.exit(0)
            
            # forward
            output = model.model(rgb_img, full_hm)
            seg_result, alpha_mpjpe, pred_mpjpe = evaltool.eval_poseseg(output, data)
            seg_results += seg_result
            alpha_mpjpes += alpha_mpjpe
            pred_mpjpes += pred_mpjpe

            # if i > 1:
            #     break
            # save results
            if i < 4:
                results = {}
                results.update(premask=output['premask'][-1].detach().cpu().numpy().astype(np.float32))
                results.update(preheat=output['preheat'][-1].detach().cpu().numpy().astype(np.float32))
                results.update(heatmap=output['heatmap'].detach().cpu().numpy().astype(np.float32))
                results.update(rgb_img=rgb_img.detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)
        
        alpha_mpjpe = np.mean(np.array(alpha_mpjpes))
        pred_mpjpe = np.mean(np.array(pred_mpjpes))

        return seg_results, alpha_mpjpe, pred_mpjpe

