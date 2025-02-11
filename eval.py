import sys
import torch
import os
from cmd_parser import parse_config
from modules import init, DatasetLoader, ModelLoader
from process import EvalModel
from utils.eval_utils import HumanEval
###########global parameters#########
sys.argv = ['','--config=cfg_files/eval.yaml']

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
    model = ModelLoader(device=device, out_dir=out_dir, smpl=smpl, generator=generator, **args)

    # create data loader
    dataset = DatasetLoader(smpl_model=smpl, generator=generator,
     occlusions=occlusions, **args)
    eval_dataset = dataset.load_evalset()

    # Load handle function with the task name
    task = args.get('task')
    exec('from process import %s_eval' %task)

    for i, (name, dataset) in enumerate(zip(dataset.testset, eval_dataset)):
        eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=False
        )
        # abs_error, error, error_pa, abs_pck, pck, imnames, joints, joints_2ds, vertex_error = EvalModel(model, evaltool, eval_loader, viz=viz, device=device)

        pred, gt = eval('%s_eval' %task)(model, eval_loader, device=device)

        evaluator = HumanEval(name)
        evaluator(pred, gt)
        vertex_error, error, error_pa, abs_pck, pck, accel = evaluator.report()


        logger.append([name, error, error_pa, abs_pck, pck, vertex_error, accel])


    logger.close()

if __name__ == "__main__":

    args = parse_config()
    main(**args)





