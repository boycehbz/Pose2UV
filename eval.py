import sys
import torch
import os
from cmd_parser import parse_config
from modules import init, DatasetLoader, ModelLoader
from process import EvalModel
from utils.eval_utils import HumanEval
###########global parameters#########
sys.argv = ['','--config=cfg_files\\eval.yaml']

def main(**args):
    # global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'),type='cuda')
    viz = args.get('viz')

    # init project setting
    out_dir, logger, smpl, generator, occlusions = init(dtype=dtype, **args)

    # load model
    model = ModelLoader(device=device, output=out_dir, smpl=smpl, generator=generator, **args)

    # create data loader
    dataset = DatasetLoader(smpl_model=smpl, generator=generator,
     occlusions=occlusions, **args)
    eval_dataset = dataset.load_evalset()
    for i, (name, dataset) in enumerate(zip(dataset.testset, eval_dataset)):
        eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True
        )
        evaltool = HumanEval(name, generator=generator, smpl=smpl, dtype=dtype, **args)
        abs_error, error, error_pa, abs_pck, pck, imnames, joints, joints_2ds, vertex_error = EvalModel(model, evaltool, eval_loader, viz=viz, device=device)

        logger.append([name, abs_error, error, error_pa, abs_pck, pck, vertex_error])
        if name == 'MuPoTS_origin':
            import numpy as np
            import scipy.io as scio
            result = []
            result_2d = []
            name_last = None
            f_result = []
            f_result_2d = []
            for i, (imname, joint, joint_2d) in enumerate(zip(imnames, joints, joints_2ds)):
                if imname != name_last:
                    if name_last is not None:
                        while(len(f_result_2d) < 3 and len(f_result) < 3):
                            f_result.append(np.zeros_like(f_result[0]))
                            f_result_2d.append(np.zeros_like(f_result_2d[0]))
                        result.append(f_result)
                        result_2d.append(f_result_2d)
                        f_result = []
                        f_result_2d = []
                    name_last = imname
                f_result.append(np.array(joint))
                f_result_2d.append(np.array(joint_2d))

            # The last one
            while(len(f_result_2d) < 3 and len(f_result) < 3):
                f_result.append(np.zeros_like(f_result[0]))
                f_result_2d.append(np.zeros_like(f_result_2d[0]))
            result.append(f_result)
            result_2d.append(f_result_2d)

            result = np.array(result).reshape(-1, 3, 17, 3)
            result_2d = np.array(result_2d).reshape(-1, 3, 17, 2)
            scio.savemat(os.path.join(out_dir, 'mupots.mat'), {'result': result, 'result_2d': result_2d})


    logger.close()

if __name__ == "__main__":

    args = parse_config()
    main(**args)





