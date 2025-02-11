'''
 @FileName    : main.py
 @EditTime    : 2024-01-13 14:32:48
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''
import sys
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from modules import init, DatasetLoader, ModelLoader, LossLoader, set_seed, seed_worker

###########global parameters#########
# sys.argv = ['','--config=cfg_files/poseseg.yaml'] #train/test/poseseg

def main(**args):
    seed = 7
    g = set_seed(seed)

    # global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'),type='cuda')
    viz = args.get('viz')
    mode = args.get('mode')

    # init project setting
    out_dir, logger, smpl, generator, occlusions = init(dtype=dtype, **args)

    # load loss function
    loss = LossLoader(smpl, device=device, generator=generator, **args)

    # load model
    model = ModelLoader(device=device, out_dir=out_dir, smpl=smpl, generator=generator, **args)

    # create data loader
    dataset = DatasetLoader(smpl_model=smpl, generator=generator, occlusions=occlusions, **args)
    if mode == 'train':
        train_dataset = dataset.load_trainset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if args.get('use_sch'):
            model.load_scheduler(train_dataset.cumulative_sizes[-1])
    test_dataset = dataset.load_testset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    task = args.pop('task')
    exec('from process import %s_train' %task)
    exec('from process import %s_test' %task)

    for epoch in range(num_epoch):
        # training modes
        if mode == 'train':
            training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, viz=viz, device=device)

            # if (epoch) % 1 == 0:
            #     model.save_model(epoch, task)

            if (epoch) % 1 == 0:
                testing_loss = eval('%s_test' %task)(model, loss,test_loader, epoch, viz=viz, device=device)
            else:
                testing_loss = 9e10

            # save trained model
            model.save_best_model(testing_loss, epoch, task)

        # testing mode
        elif epoch == 0 and mode == 'test':
            training_loss = -1.
            testing_loss = eval('%s_test' %task)(model, loss, test_loader, epoch, viz=viz, device=device)


        lr = model.optimizer.state_dict()['param_groups'][0]['lr']
        logger.append([int(epoch + 1), lr, training_loss, testing_loss])
        


if __name__ == "__main__":

    args = parse_config()
    main(**args)





