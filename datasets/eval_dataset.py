import torch
from datasets.MPDataLoader import MPData
import numpy as np
from tqdm import tqdm

class MP_Eval_Data(MPData):
    def __init__(self, train=True, use_mask=False, data_folder='', smpl=None, uv_generator=None, occlusions=None, poseseg=False, name='', use_gt=False):
        super(MP_Eval_Data, self).__init__(train=train, use_mask=use_mask, data_folder=data_folder, smpl=smpl, uv_generator=uv_generator, occlusions=occlusions, poseseg=poseseg, name=name, use_gt=use_gt)

        params = self.load_pkl(self.dataset)

        self.seq_ids, self.frame_ids, self.genders = [], [], []
        for s_id, seq in enumerate(tqdm(params, total=len(params))):
            if len(seq) < 1:
                continue
            for f_id, frame in enumerate(seq):
                # print("frame:", frame)
                for key in frame.keys():
                    if key in ['img_path', 'h_w']:
                        continue
                    self.seq_ids.append(s_id)
                    self.frame_ids.append(f_id)
                    if 'gender' in frame[key].keys():
                        self.genders.append(frame[key]['gender'])
                    else:
                        self.genders.append(np.array(-1, dtype=np.float32))

    def __getitem__(self, index):
        data = self.create_UV_maps(index)

        seq_ind = self.seq_ids[index]
        ind = self.frame_ids[index]

        data['seq_id'] = seq_ind
        data['frame_id'] = ind
        data['gender'] = np.array(self.genders[index], dtype=self.np_type)
        return data

    def __len__(self):
        return self.len


