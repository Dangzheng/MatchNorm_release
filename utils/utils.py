# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from collections import OrderedDict
import logging
from logging import handlers
import os

class NLL_Loss(nn.Module):
    def __init__(self):
        super(NLL_Loss, self).__init__()

    def forward(self, input, scores):
        return torch.mean(torch.sum(-(input + 1.e-6).log() * scores,  [1, 2])/torch.sum(scores, [1, 2]))

def init_logger(exp_name):
    file_name = 'ckpt/' + exp_name + '/log.txt'
    logger = logging.getLogger('bpnet.scratch')
    logger.handlers.clear()
    format_str = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S") # format
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=file_name, when='D', interval=10, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger

def load_checkpoint(model, optimizer=None, path=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iters = checkpoint['iters']
    else:
        iters = None # for the best model.
    try:
        best_score = checkpoint['best_score']
        return model, optimizer, iters, best_score
    except:
        return model, optimizer, iters, None

def dump_checkpoint(model, optimizer, iters, best_score, checkpoint_dir, logger):
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    path = os.path.join(checkpoint_dir, f"iter_{iters}.pth.tar")
    torch.save({
        'iters': iters,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    torch.save({
        'iters': iters,
        'model_state_dict': state_dict,
        'best_score': best_score,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_dir, "ckpt_latest.pth.tar"))
    logger.info("Saving model to %s", path)

def save_best(model, checkpoint_dir, logger):
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    path = os.path.join(checkpoint_dir, "ckpt_best.pth.tar")
    torch.save({
        'model_state_dict': state_dict,
    }, path)
    logger.info("Saving model to %s", path)

def mAP_report(meters):
    result = f'\nmAP_R:[{meters.mAP_R5.avg:.2f}, {meters.mAP_R10.avg:.2f}, {meters.mAP_R15.avg:.2f}, ' + \
            f'{meters.mAP_R20.avg:.2f}, {meters.mAP_R25.avg:.2f}, {meters.mAP_R30.avg:.2f}] '+ \
            f'(avg:{meters.mAP_R_mean.avg:.2f}) '+ \
            f'\nmAP_t:[{meters.mAP_t5.avg:.2f}, {meters.mAP_t10.avg:.2f}, {meters.mAP_t15.avg:.2f}, '+ \
            f'{meters.mAP_t20.avg:.2f}, {meters.mAP_t25.avg:.2f}, {meters.mAP_t30.avg:.2f}] '+ \
            f'(avg:{meters.mAP_t_mean.avg:.2f}) '
    return result

class AverageMeterGroup:
    """
    Average meter group for multiple average meters.
    """

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        """
        Update the meter group with a dict of metrics.
        Non-exist average meters will be automatically created.
        """
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        """
        Return a summary string of group data.
        """
        return "  ".join(v.summary() for v in self.meters.values())

class AverageMeter:
    """
    Computes and stores the average and current value.

    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.

        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

def build_bijective_label(source, target, R_ab, translation_ab, th, val, bi_layer='ot'):
    '''
    args: dict
    source: N x 3
    target: N x 3
    R_ab: 3 x 3
    translation_ab: 3 x 1
    '''
    # building the label
    src_trans = R_ab.dot(source.T) + translation_ab
    # row -> src, col -> tgt.
    src_trans = np.expand_dims(src_trans.T, axis=1)
    tgt = np.expand_dims(target, axis=0)
    score = np.linalg.norm(src_trans.repeat(target.shape[0], 1) - tgt.repeat(src_trans.shape[0], 0), axis=-1)

    row, col = np.where(score < th)

    # forward and backward consistency.
    label_col = np.zeros((score.shape[0] + 1, score.shape[1] + 1))
    label_row = np.zeros((score.shape[0] + 1, score.shape[1] + 1))
    label = np.zeros((score.shape[0] + 1, score.shape[1] + 1))  # for outlier bin
    idx = np.argmin(score, 0)
    label_col[(idx[col], col)] = 1.
    idx = np.argmin(score, 1)
    label_row[(row, idx[row])] = 1.
    label = label_col * label_row

    add_outlier_label = True if bi_layer == 'ot' else False

    if add_outlier_label is True:
        row_id = np.where(label.sum(1)==0)
        col_id = np.where(label.sum(0)==0)
        label[-1, col_id] = val
        label[row_id, -1] = val
        label[-1, -1] = 0
    else:
        label = label[:-1, :-1]
    return label

def verts2pt3d(verts):
    '''
    Convert the pointcloud coordinate to the pytorch3d compatiable format.
    '''
    trans = torch.zeros((3,3), device='cpu')
    trans[0,0] = -1
    trans[1,1] = -1 
    trans[2,2] = 1
    return trans @ verts

def voxelization_o3d(source, target, voxel_size=0.035):
    pcd_src, pcd_tgt = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(source)
    pcd_tgt.points = o3d.utility.Vector3dVector(target)
    
    pcd_src = pcd_src.voxel_down_sample(voxel_size=voxel_size)
    pcd_tgt = pcd_tgt.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_src.points), np.asarray(pcd_tgt.points)

def voxelization(pts1, pts2, th, src_points, tgt_points):
    source, target = voxelization_o3d(pts1, pts2, voxel_size=th * (np.sqrt(2) / 2))
    if source.shape[0] > src_points:
        id_1 = np.random.choice(source.shape[0], src_points, replace=False)
        source = source[id_1]
    else:
        source, target = voxelization_o3d(pts1, pts2, voxel_size=th * (np.sqrt(2) / 2) * 0.5)
        id_1 = np.random.choice(source.shape[0], src_points, replace=source.shape[0] < src_points)
        source = source[id_1]

    try:
        if target.shape[0] < tgt_points:
            id_2 = np.random.choice(pts2.shape[0], tgt_points, replace=pts2.shape[0] < tgt_points)
            target = pts2[id_2]
        else:
            id_2 = np.random.choice(target.shape[0], tgt_points, replace=target.shape[0] < tgt_points)
            target = target[id_2]
    except:
        target = np.random.rand(tgt_points, 3)
        print('no points!')

    np.random.shuffle(source)
    np.random.shuffle(target)
    return source, target