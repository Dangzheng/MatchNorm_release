# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from utils.model import BPNet
from utils.utils import NLL_Loss, init_logger, load_checkpoint
from utils.train import train_iteration, validate
from utils.data import BOP_Dataset
from torch.utils.data import DataLoader
from utils.comp import metric_func_modelnet, bop_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BPNet Training From Scratch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_frequency", type=int, default=100)
    parser.add_argument("--mode", type=str, default='ori', choices=['ori', 'ckpt'],
                        help='nas: NAS searched arch. origin: Original net arch. ckpt: load model from checkpoint.')
    parser.add_argument("--exp_name", type=str, default='test', help='experiment name')

    group_bpnet = parser.add_argument_group('bpnet', 'Hyper parameters for BPNet')
    group_bpnet.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch')
    group_bpnet.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch')
    group_bpnet.add_argument('--iters', type=int, default=30005, metavar='N',
                        help='number of episode to train ')
    group_bpnet.add_argument('--log_intv', type=int, default=1000, metavar='N',
                        help='number of iterations to write log ')
    group_bpnet.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    group_bpnet.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    group_bpnet.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    group_bpnet.add_argument('--num_subsampled_points', type=int, default=768, metavar='N',
                        help='Subsampled number of points to use')
    group_bpnet.add_argument('--rot_factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    group_bpnet.add_argument('--th', type=float, default=5.e-2, metavar='N',
                        help='Threshold for identify the inlier')
    group_bpnet.add_argument('--val', type=float, default=1, metavar='N',
                        help='Weight for the ourlier bin')
    group_bpnet.add_argument('--bi_layer', type=str, default='ot',
                        help='Set the Sinkhorn layer to be default')
    group_bop_benchmark = parser.add_argument_group('bop_benchmark', 'Dataset info for BOP benchmark')
    group_bop_benchmark.add_argument('--bop_dataset', type=str, default='tudl', choices=['lm', 'lmo', 'ycbv', 'tudl'],
                        help='Choose the dataset of bop benchmark')
    args = parser.parse_args()

    if not os.path.exists('ckpt/' + args.exp_name): os.makedirs('ckpt/' + args.exp_name)
    logger = init_logger(args.exp_name)
    for k in list(vars(args).keys()): logger.info('%s: %s' % (k, vars(args)[k]))

    if args.workers > 0:
         torch.multiprocessing.set_start_method('spawn')
         logger.info('Using multi workers.')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


    model = BPNet()
    
    if args.mode == 'ckpt':
        ckpt_path = f'./ckpt/{args.exp_name}/ckpt_best.pth.tar'
        print(f'load checkpoint: {ckpt_path}')
        model, _, _, _ = load_checkpoint(model, None, ckpt_path)

    model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(0, torch.cuda.device_count())))

    criterion = NLL_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    comp_tr = metric_func_modelnet
    comp_val = bop_benchmark


    train_loader = DataLoader(BOP_Dataset(args=args, partition='train', dataset=args.bop_dataset, mask_type='gt', format='bopdataset', pose_type='read_file'),
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            # num_workers=0,
                            shuffle=True)

    val_loader = DataLoader(BOP_Dataset(args=args, partition='test', dataset=args.bop_dataset, mask_type='gt', format='bopdataset', pose_type='read_file'),
                            batch_size=args.test_batch_size,
                            num_workers=0,
                            shuffle=False)

    if args.mode == 'ori':
        train_iteration(args, model, criterion, optimizer, train_loader, val_loader, comp_tr, comp_val, logger=logger)
    elif args.mode == 'ckpt':
        validate(0, model, val_loader, args, comp_val, logger=logger, dataset_name=args.bop_dataset)
    else:
        raise(f'Not implement mode: {args.mode}')