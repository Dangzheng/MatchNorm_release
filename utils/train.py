# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import os
import torch
from termcolor import colored
from utils.utils import dump_checkpoint, mAP_report, AverageMeterGroup, load_checkpoint, save_best
     
def validate(epoch, model, loader, args, comp_func, logger=None, dataset_name=None):
    model.eval()
    meters = AverageMeterGroup()
    
    # file_name = f'/home/dz/2022_code/bop_res/zheng-bpnet_{dataset_name}-test.csv'
    file_name = f'./bop_res/zheng-bpnet_{dataset_name}-test.csv'
    if os.path.exists(file_name): os.remove(file_name)

    with torch.no_grad():
        for step, (src, tgt, R_gt, t_gt, _, info_dict) in enumerate(loader):
            pred = model(src.cuda(), tgt.cuda())
            metrics = comp_func(src.cuda(), tgt.cuda(), R_gt, t_gt, pred, info_dict, file_name=file_name)
            meters.update(metrics)
            if step % args.log_frequency == 0 or step + 1 == len(loader):
                logger.info(colored(f'\nEpoch [{epoch + 1}/{args.epochs}] Val Step [{step + 1}/{len(loader)}] ' 
                            f'\nmAP_R:[{meters.mAP_R5.avg:.2f}, {meters.mAP_R10.avg:.2f}, {meters.mAP_R15.avg:.2f}, '
                            f'{meters.mAP_R20.avg:.2f}, {meters.mAP_R25.avg:.2f}, {meters.mAP_R30.avg:.2f}] '
                            f'(avg:{meters.mAP_R_mean.avg:.2f})'
                            f'\nmAP_t:[{meters.mAP_t5.avg:.2f}, {meters.mAP_t10.avg:.2f}, {meters.mAP_t15.avg:.2f}, '
                            f'{meters.mAP_t20.avg:.2f}, {meters.mAP_t25.avg:.2f}, {meters.mAP_t30.avg:.2f}] '
                            f'(avg:{meters.mAP_t_mean.avg:.2f})'
                            f'\nADD:[{meters.add.avg:.2f}] ADDS:[{meters.adds.avg:.2f}]'
                            f'\nInlier: pred:[{meters.inlier_pred.avg:.2f}], true:[{meters.inlier_true.avg:.2f}]'
                            , 'green' )
                            )

    logger.info(colored(f'\nEpoch [{epoch + 1}] validation ' 
                f'\nmAP_R:[{meters.mAP_R5.avg:.2f}, {meters.mAP_R10.avg:.2f}, {meters.mAP_R15.avg:.2f}, '
                f'{meters.mAP_R20.avg:.2f}, {meters.mAP_R25.avg:.2f}, {meters.mAP_R30.avg:.2f}] '
                f'(avg:{meters.mAP_R_mean.avg:.2f}) '
                f'\nmAP_t:[{meters.mAP_t5.avg:.2f}, {meters.mAP_t10.avg:.2f}, {meters.mAP_t15.avg:.2f}, '
                f'{meters.mAP_t20.avg:.2f}, {meters.mAP_t25.avg:.2f}, {meters.mAP_t30.avg:.2f}] '
                f'(avg:{meters.mAP_t_mean.avg:.2f}) '
                f'\nADD:[{meters.add.avg:.2f}] ADDS:[{meters.adds.avg:.2f}]'
                f'\nInlier: pred:[{meters.inlier_pred.avg:.2f}], true:[{meters.inlier_true.avg:.2f}]'
                , 'green')
                )