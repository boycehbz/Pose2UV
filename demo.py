'''
 @FileName    : demo.py
 @EditTime    : 2024-01-15 14:17:21
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import torch
from cmd_parser import parse_config
from modules import init, DatasetLoader, ModelLoader
from yolox.yolox import Predictor
from alphapose_module.alphapose_core import AlphaPose_Predictor

###########Load config file in debug mode#########
# import sys
# sys.argv = ['','--config=cfg_files/demo.yaml']

def main(**args):
    # Global setting
    dtype = torch.float32
    device = torch.device(index=args.get('gpu_index'), type='cuda')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl, generator, occlusions = init(dtype=dtype, **args)

    # human detection
    yolox_model_dir = R'data/bytetrack_x_mot17.pth.tar'
    yolox_thres = 0.3
    yolox_predictor = Predictor(yolox_model_dir, yolox_thres)

    # 2D pose estimation
    alpha_config = R'alphapose_module/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
    alpha_checkpoint = R'data/halpe26_fast_res50_256x192.pth'
    alpha_thres = 0.1
    alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)

    # load Pose2UV model
    model = ModelLoader(device=device, out_dir=out_dir, smpl=smpl, generator=generator, **args)

    # load data
    dataset = DatasetLoader(smpl_model=smpl, generator=generator, occlusions=occlusions, **args)
    test_dataset = dataset.load_demo_data()

    task = args.pop('task')
    exec('from process import %s' %task)

    eval('%s' %task)(model, yolox_predictor, alpha_predictor, test_dataset, device=device)


if __name__ == "__main__":
    args = parse_config()
    main(**args)





