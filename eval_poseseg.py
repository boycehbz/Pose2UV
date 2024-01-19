from pickle import load
import sys

from torch._C import set_flush_denormal
from utils.imutils import vis_img
import torch
import os
from cmd_parser import parse_config
from modules import init, DatasetLoader, ModelLoader
from process import EvalPoseSeg
from utils.eval_utils import HumanEval
from utils.coco import CocoDataset
from pycocotools.cocoeval import COCOeval
###########global parameters#########
sys.argv = ['','--config=cfg_files\\eval_poseseg.yaml']

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
        seg_results, alpha_mpjpe, pred_mpjpe = EvalPoseSeg(model, evaltool, eval_loader, viz=viz, device=device)

        logger.append([name, alpha_mpjpe, pred_mpjpe, 0, 0, 0, 0])

        if name == 'COCO2017':
            import pickle
            def save_pkl(path, result):
                """"
                save pkl file
                """
                folder = os.path.dirname(path)
                if not os.path.exists(folder):
                    os.makedirs(folder)

                with open(path, 'wb') as result_file:
                    pickle.dump(result, result_file, protocol=2)

            save_pkl(os.path.join(out_dir, 'COCO2017val_results.pkl'), seg_results)

            def load_pkl(path):
                """"
                load pkl file
                """
                param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
                return param

            # seg_results = load_pkl('COCO2017val_results.pkl')

            dataset_val = CocoDataset()
            coco = dataset_val.load_coco('E:/HumanData-Source/COCO2017', "val", year=2017, return_coco=True)
            dataset_val.prepare()
            coco_results = coco.loadRes(seg_results)
            
            # Evaluate
            cocoEval = COCOeval(coco, coco_results, 'segm')
            cocoEval.params.catIds = [1]
            # cocoEval.params.imgIds = coco_image_ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            print('AlphaPose Error: %.02f  Pred Error: %.02f'  %(alpha_mpjpe, pred_mpjpe))

    logger.close()

if __name__ == "__main__":

    args = parse_config()
    main(**args)





