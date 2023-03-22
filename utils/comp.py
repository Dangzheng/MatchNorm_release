# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import open3d as o3d
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error
import os, csv
from copy import deepcopy as dd

def icp(src, tgt, current_transformation=None, method='point2point'):
    # random initialize the rotation, using the translation of two point cloud sets' center as initionzation.
    src_o3d, tgt_o3d = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(src.numpy().T)
    tgt_o3d.points = o3d.utility.Vector3dVector(tgt.numpy().T)
    if method == 'point2point':
        result = o3d.registration.registration_icp(
                    src_o3d, tgt_o3d, 20., current_transformation,
                    o3d.registration.TransformationEstimationPointToPoint()) 
        
    elif method == 'point2plane':
        radius = 5
        max_nn = 100
        src_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        tgt_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        result = o3d.registration.registration_icp(
                    src_o3d, tgt_o3d, radius, current_transformation,
                    o3d.registration.TransformationEstimationPointToPlane())
    trans = torch.from_numpy(np.array(result.transformation)).float()
    return trans[:3, :3], trans[:3, 3][:, None]

def calc_pose(src, tgt):
    reflect = torch.eye(3)
    reflect[2, 2] = -1
    reflect = reflect.to(src)
    src_centered = src - src.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)

    H = torch.matmul(src_centered, tgt_centered.transpose(1, 0).contiguous())
    u, s, v = torch.svd(H)
    r = torch.matmul(v, u.transpose(1, 0).contiguous())
    r_det = torch.det(r)
    if r_det < 0:
        v = torch.matmul(v, reflect)
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
    t = torch.matmul(-r, src.mean(dim=1, keepdim=True)) + tgt.mean(dim=1, keepdim=True)
    return r, t

def degree_err(R_gt, R_pred, eps=1e-16):
    err = np.arccos(
      np.clip((np.trace(R_pred[:3, :3].T @ R_gt[:3, :3]) - 1) / 2, -1 + eps,
              1 - eps)) * 180. / np.pi
    return err

def trans_err(t_gt, t_pred):
    return torch.sqrt(torch.sum((t_gt - t_pred)**2))

def mse_period(R_gt, R_pred):
    # remove the period issue.
    mse_0 = ((R_gt - R_pred)**2)
    mse_1 = ((R_gt - (R_pred + 360))**2)
    mse_2 = ((R_gt - (R_pred - 360))**2)
    mse = np.stack([mse_0, mse_1, mse_2])
    mse = mse.min(0).mean()
    return mse

def mae_period(R_gt, R_pred):
    # remove the period issue.
    mae_0 = np.absolute(R_gt - R_pred)
    mae_1 = np.absolute(R_gt - (R_pred + 360))
    mae_2 = np.absolute(R_gt - (R_pred - 360))
    mae = np.stack([mae_0, mae_1, mae_2])
    mae = mae.min(0).mean()
    return mae

def mAP_th(err, bins=[0., 5., 10., 15., 20., 25., 30.]):
    '''
    different threshold, summarize it and draw the pic.

    '''
    total_num = len(err)
    hist, _ = np.histogram(err, bins=bins)
    mAP = [np.sum(hist[:i+1]) for i in range(hist.shape[0])]
    mAP = np.array(mAP) / total_num
    return mAP


def metric_func_modelnet(source, target, rotation, translation, scores, info_dict=None):
    mse_r, mse_t = [], []
    mae_r, mae_t = [], []
    degree_r = []
    trans_th = [0., 1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1, 5.e-1] # modelnet40

    for _, (score, src, tgt, R, T, scene_id, im_id, obj_id) in enumerate(zip(
        scores, 
        source.cpu(), 
        target.cpu(), 
        rotation, 
        translation,
        info_dict['scene_id'],
        info_dict['im_id'],
        info_dict['obj_id'],)):
        score[-1, -1] = np.inf
        val, row = torch.max(score, 0)
        col = torch.arange(score.shape[1])
        mask = (row != score.shape[0] - 1).type(torch.bool)
        src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
        r, t = calc_pose(src_m, tgt_m)
        R_, T_, r_, t_ = R, T, r, t

        tmp = Rotation.from_matrix(R.numpy())
        R = tmp.as_euler('zyx', degrees=True)
        tmp = Rotation.from_matrix(r.numpy())
        r = tmp.as_euler('zyx', degrees=True)

        try:
            degree_r.append(degree_err(R_, r_))
            mse_r.append(mse_period(R, r))
            mse_t.append(mean_squared_error(T, t))
            mae_r.append(mae_period(R, r))
            mae_t.append(np.absolute((T - t)).mean().item())
        except:
            # if this sample cannot produce reasonable result.
            degree_r.append(360.)
            mse_r.append(360.)
            mse_t.append(360.)
            mae_r.append(360.)
            mae_t.append(360.)

    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(mse_t, trans_th)
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(mse_t, trans_th).mean(),
        'add': 1234,
        'adds': 1234, # notice this is just a placeholder.
    }
    return metrics

def count_true_inlier(src, tgt, R, t, th=0.05):
    inlier_pred = src.shape[1]
    dist = torch.sqrt(((tgt - (R@src + t))**2).sum(0))
    inlier_true = (dist < th).sum()
    return inlier_pred, inlier_true

def bop_benchmark(source, target, rotation, translation, scores, info_dict=None, file_name='/home/dz/2022_code/bop_res/zheng-bpnet_datasetname-test.csv'):
    trans_mse = []
    trans_d = []
    degree_r = []
    add_list = []
    inlier_pred = []
    inlier_true = []

    trans_th = [0., .5, 1., 2., 5., 10., 15.] # bop_dataset in mm
    trans_th = [item * 10. for item in trans_th] # in cm
    indicator = os.path.exists(file_name)
    csvfile = open(file_name, 'a+', newline='')
    fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not indicator: writer.writeheader()

    for _, (score, src, tgt, R, T, scene_id, im_id, obj_id, model_scale, diameter) in enumerate(zip(
            scores.cpu(), 
            source.cpu(), 
            target.cpu(), 
            rotation, 
            translation, 
            info_dict['scene_id'],
            info_dict['im_id'],
            info_dict['obj_id'],
            info_dict['model_scale'],
            info_dict['diameter'],
            )):
        score[-1, -1] = np.inf
        val, row = torch.max(score, 0)
        col = torch.arange(score.shape[1])
        mask1 = (row != score.shape[0] - 1).type(torch.bool)
        mask2 = (val > 0.02).type(torch.bool)
        mask = mask1 & mask2
        src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
        r, t = calc_pose(src_m, tgt_m)
        
        # calculate inlier number
        in_pred, in_true = count_true_inlier(dd(src_m), dd(tgt_m), R, T)
        inlier_pred.append(in_pred)
        inlier_true.append(in_true)

        src /= model_scale
        tgt /= model_scale
        T /= model_scale
        t /= model_scale
        A = np.eye(4)
        A[:3, :3] = r
        A[:3, 3] = t.squeeze()
        r_icp, t_icp = icp(src, tgt, A)

        writer.writerow({'scene_id': scene_id.numpy(),
                        'im_id': im_id.numpy(),
                        'obj_id': obj_id.numpy(), 
                        'score': 1.,
                        'R': ' '.join(map(str, r_icp.flatten().tolist())),
                        't': ' '.join(map(str, t_icp.flatten().tolist())),
                        'time': 1.
                        })

        try:
            degree_r.append(degree_err(R, r_icp))
            trans_d.append(trans_err(T, t_icp))
        except:
            degree_r.append(360.)

        # ADD 
        src_gt = R @ src + T
        src_eval  = r_icp @ src + t_icp
        # src_eval  = r @ src + t
        add = torch.sum((src_gt - src_eval)**2, 0).sqrt().mean()
        th = .1 * diameter
        add_list.append(add < th)
        trans_mse.append(add)

    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(trans_d, trans_th)
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(trans_d, trans_th).mean(),
        'add': np.asarray(add_list).sum()/len(add_list),
        'adds': 1234, # notice this is just a placeholder.
        'inlier_pred': np.asarray(inlier_pred).sum()/len(inlier_pred),
        'inlier_true': np.asarray(inlier_true).sum()/len(inlier_true),
    }
    return metrics