
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.imutils import *

# from model.resnest_ed import resnest_ed
from model.UVPoser import UVPoser
partmask = np.loadtxt('./data/part_mask.txt', dtype=np.float32)

#### weight2 -
weightA = partmask.copy()
np.putmask(weightA, weightA == 1, 3.0)#2
np.putmask(weightA, weightA == 2, 1.0)
np.putmask(weightA, weightA == 0, 1.0)
np.putmask(weightA, weightA == 3, 4.0)
np.putmask(weightA, weightA == 4, 4.0)
np.putmask(weightA, weightA == 5, 12.0)#8
np.putmask(weightA, weightA == 6, 12.0)
weightAt1 = np.expand_dims(weightA, axis=0)
weightAt2 = np.repeat(weightAt1, [3], axis=0)
weightAt3 = np.expand_dims(weightAt2, axis=0)

weightB = partmask.copy()
np.putmask(weightB, weightB == 1, 1.0)
np.putmask(weightB, weightB == 2, 2.0)
np.putmask(weightB, weightB == 0, 2.0)
np.putmask(weightB, weightB == 3, 2.0)
np.putmask(weightB, weightB == 4, 2.0)
np.putmask(weightB, weightB == 5, 0.5)
np.putmask(weightB, weightB == 6, 0.5)
weightBt1 = np.expand_dims(weightB, axis=0)
weightBt2 = np.repeat(weightBt1, [3], axis=0)
weightBt3 = np.expand_dims(weightBt2, axis=0)


class weight_L1(nn.Module):
    def __init__(self, device):
        super(weight_L1, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss(reduction='none')
        self.weight = torch.from_numpy(weightAt3).to(self.device)

    def forward(self, x, y, flag):
        diff = self.L1Loss(x, y)
        diff = diff[flag == 1]
        diff = self.weight * diff
        diff = torch.mean(diff, [1,2,3])
        diff = torch.mean(diff)
        return diff


class Surface_smooth_Loss(nn.Module):
    def __init__(self, device, faces):
        super(Surface_smooth_Loss, self).__init__()
        self.device = device
        self.faces = faces
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.weight = 1.0

    def forward(self, pred_verts):
        loss_dict = {}

        v1 = pred_verts[:,self.faces[:,0].tolist()]
        v2 = pred_verts[:,self.faces[:,1].tolist()]
        v3 = pred_verts[:,self.faces[:,2].tolist()]

        loss = 0.
        loss += self.criterion_vert(v1, v2)
        loss += self.criterion_vert(v1, v3)
        loss += self.criterion_vert(v3, v2)

        loss_dict['smooth_loss'] = loss * self.weight

        return loss_dict


class SMPL_Loss(nn.Module):
    def __init__(self, device, smpl, generator):
        super(SMPL_Loss, self).__init__()
        self.device = device
        self.smpl = smpl
        self.generator = generator
        self.regressor = torch.tensor(np.load('data/J_regressor_halpe.npy').astype(np.float32)).to(self.device)
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss().to(self.device)
        self.joint_weight = 1.0
        self.verts_weight = 1.0

    def forward(self, pred_verts, gt_verts, flag):
        loss_dict = {}

        pred_verts = pred_verts[flag==1]
        gt_verts = gt_verts[flag==1]

        pred_vertices_with_shape = pred_verts
        gt_vertices_with_shape = gt_verts

        pred_joints = torch.matmul(self.regressor, pred_vertices_with_shape)
        gt_joints = torch.matmul(self.regressor, gt_vertices_with_shape)

        pred_pelvis = pred_joints[:,19,:][:,None,:].detach()
        gt_pelvis = gt_joints[:,19,:][:,None,:].detach()

        pred_joints = pred_joints - pred_pelvis
        gt_joints = gt_joints - gt_pelvis

        pred_vertices_with_shape = pred_vertices_with_shape - pred_pelvis
        gt_vertices_with_shape = gt_vertices_with_shape - gt_pelvis

        if len(gt_vertices_with_shape) > 0:
            vert_loss = self.criterion_vert(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            vert_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['vert_loss'] = vert_loss * self.verts_weight

        if len(gt_joints) > 0:
            joint_loss = self.criterion_joint(pred_joints, gt_joints)
        else:
            joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['joint_loss'] = joint_loss * self.joint_weight

        return loss_dict

class POSE_L1(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(POSE_L1, self).__init__()
        self.dtype = dtype
        self.device = device
        self.L1Loss = nn.L1Loss()
        self.weight = 100.

    def forward(self, pred, gt, flag):

        loss = 0.
        if (flag==1).max() == True:
            loss += self.L1Loss(pred[0][flag==1], gt[0][flag==1])
            loss += self.L1Loss(pred[1][flag==1], gt[0][flag==1])
            loss += self.L1Loss(pred[2][flag==1], gt[1][flag==1])
            loss += self.L1Loss(pred[3][flag==1], gt[2][flag==1])
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        return loss * self.weight

class POSE_L2(nn.Module):
    def __init__(self, device, visible_only=False, dtype=torch.float32):
        super(POSE_L2, self).__init__()
        self.dtype = dtype
        self.device = device
        self.L2Loss = nn.MSELoss()
        self.weight = 100.
        self.visible_only = visible_only

    def forward(self, pred, gt, flag, vis):

        vis = vis[flag==1][:,:,None,None]

        if self.visible_only:
            vis = torch.ones_like(vis)

        loss = 0.
        if (flag==1).max() == True:
            loss += self.L2Loss(pred[-2][flag==1]*vis, gt[0][flag==1]*vis)
            # loss += self.L2Loss(pred[1][flag==1]*vis, gt[0][flag==1]*vis)
            # loss += self.L2Loss(pred[2][flag==1]*vis, gt[1][flag==1]*vis)
            # loss += self.L2Loss(pred[3][flag==1]*vis, gt[2][flag==1]*vis)
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        return loss * self.weight

class MASK_L2(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(MASK_L2, self).__init__()
        self.dtype = dtype
        self.device = device
        self.L2Loss = nn.MSELoss()
        self.weight = 1.

    def forward(self, pred, gt, flag):

        loss = 0.
        if (flag==1).max() == True:
            loss += self.L2Loss(pred[-2][:,0][flag==1], gt[1][flag==1])
            # loss += self.L2Loss(pred[1][flag==1], gt[0][flag==1])
            # loss += self.L2Loss(pred[2][flag==1], gt[1][flag==1])
            # loss += self.L2Loss(pred[3][flag==1], gt[2][flag==1])
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        return loss * self.weight

class MASK_L1(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(MASK_L1, self).__init__()
        self.dtype = dtype
        self.device = device
        self.L1Loss = nn.L1Loss()
        self.weight = 10.

    def forward(self, pred, gt, flag):

        loss = 0.
        if (flag==1).max() == True:
            loss += self.L1Loss(pred[0][flag==1], gt[0][flag==1])
            loss += self.L1Loss(pred[1][flag==1], gt[0][flag==1])
            loss += self.L1Loss(pred[2][flag==1], gt[1][flag==1])
            loss += self.L1Loss(pred[3][flag==1], gt[2][flag==1])
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        return loss * self.weight

class Edge_Loss(nn.Module):
    def __init__(self, device, smpl, generator):
        super(Edge_Loss, self).__init__()
        self.device = device
        self.smpl = smpl
        self.generator = generator
        self.regressor = torch.tensor(np.load('data/J_regressor_halpe.npy').astype(np.float32)).to(self.device)
        self.criterion_edge = nn.L1Loss().to(self.device)
        self.edge_weight = 1.0
        self.faces1 = smpl.faces[:,0].tolist()
        self.faces2 = smpl.faces[:,1].tolist()
        self.faces3 = smpl.faces[:,2].tolist()

    def forward(self, pred_verts, gt_verts, flag):
        loss_dict = {}

        pred_verts = pred_verts[flag==1]
        gt_verts = gt_verts[flag==1]

        pred_edge1 = torch.sum(torch.abs(pred_verts[:,self.faces1] - pred_verts[:,self.faces2]), dim=2)
        pred_edge2 = torch.sum(torch.abs(pred_verts[:,self.faces1] - pred_verts[:,self.faces3]), dim=2)
        pred_edge3 = torch.sum(torch.abs(pred_verts[:,self.faces2] - pred_verts[:,self.faces3]), dim=2)
        gt_edge1 = torch.sum(torch.abs(gt_verts[:,self.faces1] - gt_verts[:,self.faces2]), dim=2)
        gt_edge2 = torch.sum(torch.abs(gt_verts[:,self.faces1] - gt_verts[:,self.faces3]), dim=2)
        gt_edge3 = torch.sum(torch.abs(gt_verts[:,self.faces2] - gt_verts[:,self.faces3]), dim=2)

        edge_loss = self.criterion_edge(pred_edge1, gt_edge1)
        edge_loss += self.criterion_edge(pred_edge2, gt_edge2)
        edge_loss += self.criterion_edge(pred_edge3, gt_edge3)
        edge_loss = edge_loss

        loss_dict['edge_loss'] = edge_loss * self.edge_weight

        return loss_dict

class MPJPE(nn.Module):
    def __init__(self, generator, device, dtype=torch.float32):
        super(MPJPE, self).__init__()
        self.dtype = dtype
        self.generator = generator
        self.device = device
        self.J_regressor_halpe = torch.tensor(np.load('data/J_regressor_halpe.npy').astype(np.float32)).to(self.device)

        self.halpe2lsp = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 18, 17]

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}

        # from utils.gui_3d import Gui_3d
        # gui = Gui_3d()
        conf = gt_joints[:, self.halpe2lsp, -1]

        pred_joints = pred_joints[:, self.halpe2lsp]
        gt_joints = gt_joints[:, self.halpe2lsp, :3]

        # use lsp format directly

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp').detach()
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp').detach()

        # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='lsp')
        diff = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000

        return diff

    def pa_mpjpe(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:, self.halpe2lsp, -1]

        pred_joints = pred_joints[:, self.halpe2lsp]
        gt_joints = gt_joints[:, self.halpe2lsp, :3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000

        return diff

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0, 2, 1)
            S2 = S2.permute(0, 2, 1)
            transposed = True
        assert (S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0, 2, 1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0], 1, 1)
        t1 = U.bmm(V.permute(0, 2, 1))
        t2 = torch.det(t1)
        Z[:, -1, -1] = Z[:, -1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0, 2, 1)

        return S1_hat

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]

        return joints - pelvis[:, None, :].repeat(1, 14, 1)


class vtcloss(nn.Module):
    def __init__(self, generator):
        super(vtcloss, self).__init__()
        self.generator  = generator
        self.func = torch.nn.L1Loss(size_average=False)
    def forward(self, pre, gt):
        batch_size = pre.size(0)
        pre = self.generator.resample_t(pre).to(pre.device)
        gt = self.generator.resample_t(gt).to(pre.device)
        loss = self.func(pre, gt) / batch_size
        # bary_weights = torch.FloatTensor(self.generator.bary_weights).to(pre.device)
        # v_index = torch.LongTensor(self.generator.v_index).to(pre.device)
        # new_vts = torch.LongTensor(self.generator.refine_vts).to(pre.device)
        # resmaple_vvt = torch.LongTensor(self.generator.resample_v_to_vt).to(pre.device)
        # pre = resample_mesh_func(pre, new_vts, resmaple_vvt, bary_weights, v_index)
        # gt = resample_mesh_func(gt, new_vts, resmaple_vvt, bary_weights, v_index)
        #### for debug
        # gt = gt_v.permute(0,2,1)
        # tmp = gt[0].detach().cpu().numpy()
        # smpl = SMPLModel(
        #                 device=torch.device('cpu'),
        #                 model_path='./data/model_lsp.pkl', 
        #             )
        # smpl.write_obj(tmp, 'test_newfunc.obj')
        # inp = (pre, new_vts, resmaple_vvt, bary_weights, v_index)
        # from torch.autograd import gradcheck
        # test = gradcheck(resample_mesh, inp, eps=1e-6, atol=1e-4)
        # print(test)
        
        #(resample_mesh_func(pre, new_vts, resmaple_vvt, bary_weights, v_index), resample_mesh_func(gt, new_vts, resmaple_vvt, bary_weights, v_index)).to(pre.device)
        return loss

class vaeloss(nn.Module):
    def __init__(self, pretrain_dir, latentD):
        super(vaeloss, self).__init__()
        self.latentD = latentD
        self.vae = UVPoser(512, self.latentD, 1) #resnest_ed(512,self.latentD,1)
        model_dict = self.vae.state_dict()
        seg_model_dict = torch.load(pretrain_dir).state_dict()
        model_dict.update(seg_model_dict)
        self.vae.load_state_dict(model_dict)
        # fixed model parameters
        for param in self.vae.parameters():
            param.requires_grad = False

    def forward(self, pred, gt=None):
        batch_size = pred.size(0)
        device = pred.device
        dtype = pred.dtype

        # # KL between ground-truth and prediction
        # if gt is not None:
        #     gt_qz = self.vae.encode_qz(gt)
        # else:
        #     gt_qz = torch.distributions.normal.Normal(
        #             loc=torch.tensor(np.zeros([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype),
        #             scale=torch.tensor(np.ones([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype))
        
        # pred_qz = self.vae.encode_qz(pred)
        # loss = torch.sum(torch.distributions.kl.kl_divergence(gt_qz, pred_qz))

        # L2 of sampled z
        mean = self.vae.encode_qz(pred)
        # sampled = pred_qz.sample()
        # sampled.requires_grad = True
        loss = torch.sum(torch.sqrt(torch.sum(mean**2, dim=1))) * 20 / batch_size
        return loss

class Bone_Loss(nn.Module):
    def __init__(self, generator, device):
        super(Bone_Loss, self).__init__()
        self.generator = generator
        self.device = device
        self.bone_weight = 1.0
        self.L1Loss = torch.nn.L1Loss() # L1Loss MSELoss
        self.J_regressor = torch.from_numpy(np.load('data/J_regressor_h36m.npy')).float().to(self.device)

    def forward(self, verts):
        loss_dict = {}
        joints = torch.tensordot(verts, self.J_regressor, dims=([1], [1])).permute(0, 2, 1)
        pre_bone, flip_bone = cal_bonelength(joints)
        loss = self.L1Loss(pre_bone, flip_bone)

        loss_dict['Bone_Loss'] = loss * self.bone_weight
        return loss_dict

class shapeloss(nn.Module):
    def __init__(self, generator):
        super(shapeloss, self).__init__()
        self.generator  = generator
        self.L1Loss = torch.nn.L1Loss(size_average=False) # L1Loss MSELoss
   
    def forward(self, pre):
        batch_size = pre.size(0)
        anchor_1 = pre[:,:, self.generator.set1[:,0], self.generator.set1[:,1]].permute(0,2,1)
        anchor_2 = pre[:,:, self.generator.set2[:,0], self.generator.set2[:,1]].permute(0,2,1)
        anchor_len = torch.norm(torch.abs(anchor_1 - anchor_2), dim=2)
        flip = [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11]
        anchor_flip = anchor_len[:,flip]
     
        # mean_bone = self.mean_bone.repeat(pre_bone.size(0),1,1).to(pre.device)
        # loss = self.L1Loss(pre_bone, mean_bone)
        loss = self.L1Loss(anchor_len, anchor_flip) / batch_size
        return loss



class L2(nn.Module):
    def __init__(self, device):
        super(L2, self).__init__()
        self.device = device

    def forward(self, x, y, flag):
        diff = (x - y) ** 2
        diff = torch.sum(diff, [1,2,3])
        diff = diff * flag
        # diff = torch.sum(diff)
        diff = torch.mean(diff)
        return diff


class L1(nn.Module):
    def __init__(self, device):
        super(L1, self).__init__()
        self.device = device

    def forward(self, x, y, flag):
        diff = torch.abs(x - y)
        diff = torch.sum(diff, [1,2,3])
        diff = diff * flag
        # diff = torch.sum(diff)
        diff = torch.mean(diff)
        return diff

class LPloss(nn.Module):
    def __init__(self, device):
        super(LPloss, self).__init__()
        self.device = device

    def forward(self, x, y):
        batch_size = x.size(0)
        weight = torch.from_numpy(np.repeat(weightBt3, x.shape[0], axis=0)).to(self.device)
        return torch.sum(weight[:, :, :x.shape[2], :x.shape[3]] * torch.abs(x - y)) / batch_size
        # return torch.mean(weight[:, :, :x.shape[2], :x.shape[3]] * torch.abs(x - y))
        
class part_loss(nn.Module):
    def __init__(self, generator):
        super(part_loss, self).__init__()
        # self.genertaor  = generator
        self.t2 = torch.Tensor(generator.part2).long()
        self.t3 = torch.Tensor(generator.part3).long()
        self.t4 = torch.Tensor(generator.part4).long()
        self.func = torch.nn.L1Loss(size_average=False)
        
    def forward(self, x, y):
        mean2 = torch.mean(y[:, :, self.t2[:, :, 0], self.t2[:, :, 1]], dim=3, keepdim=True)
        mean3 = torch.mean(y[:, :, self.t3[:, :, 0], self.t3[:, :, 1]], dim=3, keepdim=True)
        mean4 = torch.mean(y[:, :, self.t4[:, :, 0], self.t4[:, :, 1]], dim=3, keepdim=True)
        mean2 = torch.cat([mean2, mean2], dim=3)
        mean3 = torch.cat([mean3, mean3, mean3], dim=3)
        mean4 = torch.cat([mean4, mean4, mean4, mean4], dim=3)
        loss1 = self.func(x[:, :, self.t2[:, :, 0], self.t2[:, :, 1]], mean2)
        loss2 = self.func(x[:, :, self.t3[:, :, 0], self.t3[:, :, 1]], mean3)
        loss3 = self.func(x[:, :, self.t4[:, :, 0], self.t4[:, :, 1]], mean4) 
        return (loss1+loss2+loss3) / x.size(0)
