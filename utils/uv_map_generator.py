import numpy as np
import pickle
import torch

class UV_Map_Generator():
    def __init__(self, UV_height, UV_width=-1, 
        UV_pickle='param.pickle'):
        self.h = UV_height
        self.w = self.h if UV_width < 0 else UV_width
        
        ### Load UV texcoords and face mapping info
        with open(UV_pickle, 'rb') as f:
            tmp = pickle.load(f)
        for k in tmp.keys():
            setattr(self, k, tmp[k])

        t1 = [4432,4459,4559,4579,947,973,1073,1093,6280,2820,5026,5016,2818,6281,1577,1601]
        t2 = [4472,4438,4939,4571,986,952,1465,1085,5364,4852,5157,5195,1903,789,1687,1725]

        # t1 = [4433,4459,4559,4579,944,972,1073,1114,6280,2820,5411,5016,2818,6281,1950,1569]
        # t2 = [4472,4441,4936,4571,986,952,1463,1085,5364,4267,5400,5413,1903,791,1689,1725]
        self.tmp1 = [[list(self.v_to_vt[m+1])[0]] for m in t1]
        self.tmp2 = [[list(self.v_to_vt[m+1])[0]] for m in t2]

        self.set1 = (np.squeeze(self.texcoords[np.array(self.tmp1)],axis=1)*255).astype(np.int)
        self.set2 = (np.squeeze(self.texcoords[np.array(self.tmp2)],axis=1)*255).astype(np.int)
        # uv[self.set1[:,0],self.set1[:,1],:]
        # print('ok')
        self.vt_to_v_index = np.array([
            self.vt_to_v[i] for i in range(self.texcoords.shape[0])
        ])

    def UV_interp(self, rgbs):
        face_num = self.vt_faces.shape[0]
        vt_num = self.texcoords.shape[0]
        assert(vt_num == rgbs.shape[0])
       # uvs = self.vts #self.texcoords * np.array([[self.h - 1, self.w - 1]])
        triangle_rgbs = rgbs[self.vt_faces][self.face_id]
        bw = self.bary_weights[:,:,np.newaxis,:]
        im = np.matmul(bw, triangle_rgbs).squeeze(axis=2)
        return im

    def get_UV_map(self, verts):
        vmin = np.min(verts, axis=0) 
        vmax = np.max(verts, axis=0)
        #box = (vmax-vmin).max() #2019.11.9 vmax.max()
        box = 2 # define 2 meters bounding-box  @buzhenhuang 21/04/2020
        verts = (verts - vmin) / box - 0.5
        rgbs = verts[self.vt_to_v_index]
        uv = self.UV_interp(rgbs)
        return uv, vmin, vmax

    def get_UV_t(self, verts):
        #### input verts： [N, 6890, 3]
        #### output uv map： [N, 256, 256, 3] 
        # normalized to [-0.5,0.5] backgruond 0
        vmin = torch.min(verts, axis=1)[0] 
        box = 2 
        verts = (verts - vmin) / box - 0.5
        bary_weights_t = torch.FloatTensor(self.bary_weights).to(verts.device)
        v_index_t = torch.LongTensor(self.v_index).to(verts.device)
        if verts.dim() == 2:
            verts = verts.unsqueeze(0)
        im = verts[:, v_index_t, :]
        bw = bary_weights_t[:, :, None, :]
        uv = torch.matmul(bw, im).squeeze(dim=3)
        return uv

    def resample_t(self, input_uv):
        # uv = UV_map.to(device)#torch.from_numpy(UV_map).to(device)
        new_vts = self.refine_vts
        resmaple_vvt = self.resample_v_to_vt
        input_uv = input_uv.permute(0,2,3,1)
        vt_3d = input_uv[:, new_vts.T[0], new_vts.T[1]]
        opt_v_3d = vt_3d[:, resmaple_vvt]
        return opt_v_3d

    def resample_np(self, input_uv):
        # uv = UV_map.to(device)#torch.from_numpy(UV_map).to(device)
        new_vts = self.refine_vts
        resmaple_vvt = self.resample_v_to_vt
        input_uv = input_uv.transpose((0,2,3,1))
        vt_3d = input_uv[:, new_vts.T[0], new_vts.T[1]]
        opt_v_3d = vt_3d[:, resmaple_vvt]
        return opt_v_3d


