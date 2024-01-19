import torch
import torch.nn as nn
import numpy as np
import cv2
from pycocotools import mask as maskUtils
from utils.imutils import vis_img

class HumanEval(nn.Module):
    def __init__(self, name, generator=None, smpl=None, dtype=torch.float32, **kwargs):
        super(HumanEval, self).__init__()
        self.generator = generator
        self.smpl = smpl
        self.dtype = dtype
        self.name = name
        self.dataset_scale = self.dataset_mapping(self.name)
        self.J_regressor_H36 = np.load('data/J_regressor_h36m.npy').astype(np.float32)
        self.J_regressor_LSP = self.smpl.joint_regressor.clone().transpose(1,0).detach().numpy()
        self.J_regressor_SMPL = self.smpl.J_regressor.clone().detach().numpy()

        self.eval_handler_mapper = dict(
            VCLMP=self.LSPEvalHandler,
            h36m_valid_protocol1=self.LSPEvalHandler,
            h36m_valid_protocol2=self.LSPEvalHandler,
            MPI3DPW=self.SMPLEvalHandler,
            Panoptic_haggling1=self.PanopticEvalHandler,
            Panoptic_mafia2=self.PanopticEvalHandler,
            Panoptic_pizza1=self.PanopticEvalHandler,
            Panoptic_ultimatum1=self.PanopticEvalHandler,
            Panoptic_Eval=self.PanopticEvalHandler,
            MuPoTS_origin=self.MuPoTSEvalHandler,
        )

    def dataset_mapping(self, name):
        if name == 'VCLMP':
            return 105
        else:
            return 1

    def estimate_translation_from_intri(self, S, joints_2d, joints_conf, fx=5000., fy=5000., cx=128., cy=128.):
        num_joints = S.shape[0]
        # focal length
        f = np.array([fx, fy])
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

        # test
        A += np.eye(A.shape[0]) * 1e-6

        # solution
        trans = np.linalg.solve(A, b)
        return trans

    def cal_trans(self, J3ds, J2ds, intris):
        trans = np.zeros((J3ds.shape[0], 3))
        for i, (J3d, J2d, intri) in enumerate(zip(J3ds, J2ds, intris)):
            fx = intri[0][0]
            fy = intri[1][1]
            cx = intri[0][2]
            cy = intri[1][2]
            j_conf = J2d[:,2] 
            trans[i] = self.estimate_translation_from_intri(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, fx=fx, fy=fy)
        return trans

    def get_abs_meshes(self, pre_meshes, joints_2ds, intri):
        lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        pre_meshes = ((pre_meshes + 0.5) * 2. * self.dataset_scale)
        # get predicted 3D joints and estimate translation
        joints = np.matmul(self.J_regressor_LSP, pre_meshes)
        # we use 12 joints to calculate translation
        transl = self.cal_trans(joints[:,lsp14_to_lsp13], joints_2ds, intri)

        abs_mesh = pre_meshes + transl[:,np.newaxis,:]
        return abs_mesh

    def compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale*(R.dot(mu1))

        # 7. Error:
        S1_hat = scale*R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat


    def align_by_pelvis(self, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]
        if get_pelvis:
            return joints - np.expand_dims(pelvis, axis=0), pelvis
        else:
            return joints - np.expand_dims(pelvis, axis=0)

    def align_mesh_by_pelvis(self, mesh, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2
            pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]
        if get_pelvis:
            return mesh - np.expand_dims(pelvis, axis=0), pelvis
        else:
            return mesh - np.expand_dims(pelvis, axis=0)

    def compute_errors(self, gt3ds, preds, format='lsp', confs=None):
        """
        Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
        Evaluates on the 14 common joints.
        Inputs:
        - gt3ds: N x 14 x 3
        - preds: N x 14 x 3
        """
        if confs is None:
            confs = np.ones((gt3ds.shape[:2]))
        abs_errors, errors, errors_pa, abs_pck, pck = [], [], [], [], []
        for i, (gt3d, pred, conf) in enumerate(zip(gt3ds, preds, confs)):
            gt3d = gt3d.reshape(-1, 3)

            # Get abs error.
            joint_error = np.sqrt(np.sum((gt3d - pred)**2, axis=1)) * conf
            abs_errors.append(np.mean(joint_error))

            # Get abs pck.
            abs_pck.append(np.mean(joint_error < 150) * 100)

            # Root align.
            gt3d = self.align_by_pelvis(gt3d, format=format)
            pred3d = self.align_by_pelvis(pred, format=format)

            joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1)) * conf
            errors.append(np.mean(joint_error))

            # Get pck
            pck.append(np.mean(joint_error < 150) * 100)

            # Get PA error.
            pred3d_sym = self.compute_similarity_transform(pred3d, gt3d)
            pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1)) * conf
            errors_pa.append(np.mean(pa_error))

        return abs_errors, errors, errors_pa, abs_pck, pck

    def LSPEvalHandler(self, premesh, gt_joint):
        joints = np.matmul(self.J_regressor_LSP, premesh)

        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='lsp')
        return abs_error, error, error_pa, abs_pck, pck

    def SMPLMeshEvalHandler(self, premeshes, gt_meshes):
        premeshes = premeshes * 1000
        gt_meshes = gt_meshes * 1000

        joints = np.matmul(self.J_regressor_LSP, premeshes)
        gt_joints = np.matmul(self.J_regressor_LSP, gt_meshes)

        vertex_errors = []

        for i, (premesh, gt_mesh, joint, gt_joint) in enumerate(zip(premeshes, gt_meshes, joints, gt_joints)):
            # Root align.
            premesh = self.align_mesh_by_pelvis(premesh, joint, format='lsp')
            gt_mesh = self.align_mesh_by_pelvis(gt_mesh, gt_joint, format='lsp')

            vertex_error = np.sqrt(np.sum((premesh - gt_mesh)**2, axis=1))
            vertex_errors.append(np.mean(vertex_error))

        return vertex_errors

    def PanopticEvalHandler(self, premesh, gt_joint):
        joints = np.matmul(self.J_regressor_H36, premesh)
        conf = gt_joint[:,:,-1].copy()
        gt_joint = gt_joint[:,:,:3]
        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='h36m', confs=conf)
        return abs_error, error, error_pa, abs_pck, pck

    def MuPoTSEvalHandler(self, premesh, gt_joint):
        h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
        joints = np.matmul(self.J_regressor_H36, premesh)
        # gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        joints = joints[:,h36m_to_MPI]
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='mpi')

        return abs_error, error, error_pa, abs_pck, pck, joints

    def SMPLEvalHandler(self, premesh, gt_joint):
        joints = np.matmul(self.J_regressor_SMPL, premesh)

        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='smpl')
        return abs_error, error, error_pa, abs_pck, pck

    def forward(self, output, data):
        results = {}
        # data processing
        intris = data['intri'].to(self.dtype).detach().cpu().numpy()
        gt_joint = data['gt_3d'].to(self.dtype).detach().cpu().numpy()
        transs = data['trans'].to(self.dtype).cpu().numpy()
        scales = data['scale'].cpu().numpy()

        # get predicted 2D joints
        heatmaps = output['preheat'][4].cpu().numpy()
        joints_2ds = np.zeros((heatmaps.shape[0], heatmaps.shape[1], 3))
        confidence = np.max(heatmaps, axis=(2,3))
        confidence[np.where(confidence < 0.7)] = 0
        joints_2ds[:,:,2] = confidence
        for ind, (heatmap, trans, scale) in enumerate(zip(heatmaps, transs, scales)):
            for j, hm in enumerate(heatmap):
                if joints_2ds[ind][j][2] < 0.7:
                    continue
                joints_2ds[ind][j][:2] = np.mean(np.where(hm==joints_2ds[ind][j][2]), axis=1)[::-1]
            joints_2ds[ind,:,0] -= trans[0]
            joints_2ds[ind,:,1] -= trans[1]
            joints_2ds[ind,:,:2] /= scale

        # img_dir = data['raw_img'][0]
        # images = cv2.imread(img_dir)
        # img = images.copy()
        # for joint in joints_2ds[0]:
        #     img = cv2.circle(img, tuple(joint[:2].astype(np.int)), 10, (0,0,255), -1)
        #     vis_img('img', img)

        # get predicted meshes
        pre_meshes = self.generator.resample_np(output['decoded'].detach().cpu().numpy())
        meshes = self.get_abs_meshes(pre_meshes, joints_2ds, intris)
        
        if self.name == 'MuPoTS_origin':
            abs_error, error, error_pa, abs_pck, pck, joints = self.eval_handler_mapper[self.name](meshes, gt_joint)
            imnames = data['raw_img']
            joints_2ds = np.matmul(intris, joints.transpose((0,2,1)))
            joints_2ds = (joints_2ds[:,:2,:] / joints_2ds[:,-1:,:]).transpose((0,2,1))
            joints = joints.tolist()
            joints_2ds = joints_2ds.tolist()
        else:
            abs_error, error, error_pa, abs_pck, pck = self.eval_handler_mapper[self.name](meshes, gt_joint)
            imnames = [None] * len(abs_error)
            joints = [None] * len(abs_error)
            joints_2ds = [None] * len(abs_error)

        # calculate vertex error
        if data['gt_mesh'].size(1) < 6890:
            vertex_error = [None] * len(abs_error)
        else:
            meshes = meshes / self.dataset_scale
            vertex_error = self.SMPLMeshEvalHandler(meshes, data['gt_mesh'].detach().cpu().numpy())

        return abs_error, error, error_pa, abs_pck, pck, imnames, joints, joints_2ds, vertex_error

    def eval_poseseg(self, output, data):
        results = {}
        # data processing
        transs = data['trans'].to(self.dtype).cpu().numpy()
        scales = data['scale'].cpu().numpy()
        img_shapes = data['img_shape'].cpu().numpy()
        img_ids = data['image_id'].cpu().numpy().tolist()

        # get predicted 2D joints
        heatmaps = output['preheat'][4].cpu().numpy()
        masks = output['premask'][4].cpu().numpy()
        joints_2ds = np.zeros((heatmaps.shape[0], heatmaps.shape[1], 3))
        confidence = np.max(heatmaps, axis=(2,3))
        scores = np.mean(confidence, axis=1)
        confidence[np.where(confidence < 0.3)] = 0
        joints_2ds[:,:,2] = confidence
        seg_results = []
        for ind, (heatmap, trans, scale, mask, shape, im_id) in enumerate(zip(heatmaps, transs, scales, masks, img_shapes, img_ids)):
            for j, hm in enumerate(heatmap):
                if joints_2ds[ind][j][2] < 0.3:
                    continue
                joints_2ds[ind][j][:2] = np.mean(np.where(hm==joints_2ds[ind][j][2]), axis=1)[::-1]
            mask = mask[0]
            M = np.float32([[1, 0, -trans[0]], [0, 1, -trans[1]]])
            mask = cv2.warpAffine(mask, M, (int(shape[1]*scale), int(shape[0]*scale)), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0))
            mask = cv2.resize(mask, (int(shape[1]), int(shape[0])))
            mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
            if mask.max() < 0.01:
                index = (np.array([0]), np.array([0]))
            else:
                index = np.where(mask>0)
            # lt = np.array([index[1].min(), index[0].min()])
            # rb = np.array([index[1].max(), index[0].max()])
            joints_2ds[ind,:,0] -= trans[0]
            joints_2ds[ind,:,1] -= trans[1]
            joints_2ds[ind,:,:2] /= scale

            result = {
                "image_id": im_id,
                "category_id": 1,
                "bbox": [float(index[1].min()), float(index[0].min()), float(index[1].max()) - float(index[1].min()), float(index[0].max()) - float(index[0].min())],
                "score": float(scores[ind]),
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            seg_results.append(result)

            # from utils.imutils import convert_color
            # img_dir = data['raw_img'][ind]
            # images = cv2.imread(img_dir)
            # img = images.copy()
            # for joint in joints_2ds[ind]:
            #     if joint[2] > 0.3:
            #         img = cv2.circle(img, tuple(joint[:2].astype(np.int)), 10, (0,0,255), -1)
            # # img = cv2.rectangle(img, tuple(lt), tuple(rb), (0,255,255), 5)
            # ms = convert_color(mask)
            # img = cv2.addWeighted(ms, 0.5, img.astype(np.uint8),0.5,0)
            # vis_img('img', img)
        
        alpha_kp2d = data['input_kp2d'].numpy()
        gt_kp2d = data['gt_kp2d'].numpy()
        alpha_mpjpe_2d = []
        pred_mpjpe_2d = []
        for ind, (alpha, gt, pred) in enumerate(zip(alpha_kp2d, gt_kp2d, joints_2ds)):
            if alpha.max() < 1 or gt.max() < 1:
                continue
            confidence = gt[:,2]
            gt = gt[:,:2]
            alpha_conf = alpha[:,2]
            alpha_conf[np.where(alpha_conf>0.5)] = 1
            alpha_conf[np.where(alpha_conf<=0.5)] = 0
            alpha = alpha[:,:2]
            pred_conf = pred[:,2]
            pred_conf[np.where(pred_conf>0.2)] = 1
            pred_conf[np.where(pred_conf<=0.2)] = 0
            pred = pred[:,:2]
            # # set threshold to select visible pose for Alphapose
            # alpha_error = np.sqrt(np.sum((alpha - gt)**2, axis=1)) * confidence * pred_conf #* alpha_conf
            # # set threshold to select visible pose for Pred
            # pred_error = np.sqrt(np.sum((pred - gt)**2, axis=1)) * confidence * pred_conf
            # set threshold to select visible pose for Alphapose
            alpha_error = np.sqrt(np.sum((alpha * np.expand_dims(alpha_conf,1).repeat(2,axis=1) - gt)**2, axis=1))
            # set threshold to select visible pose for Pred
            pred_error = np.sqrt(np.sum((pred * np.expand_dims(pred_conf,1).repeat(2,axis=1) - gt)**2, axis=1))
            alpha_error = np.mean(alpha_error)
            pred_error = np.mean(pred_error)
            alpha_mpjpe_2d.append(alpha_error)
            pred_mpjpe_2d.append(pred_error)

        return seg_results, alpha_mpjpe_2d, pred_mpjpe_2d