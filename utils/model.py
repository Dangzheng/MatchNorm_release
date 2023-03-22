# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from utils.transformer import Transformer

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
    
def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature

class MatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = None

    def forward(self, x, v_min=-1, v_max=1):
        reshape = False
        if x.dim() == 4:
            x = x.contiguous()
            reshape = True
            bs, c, num_pt, num_gr = x.shape
            x = x.view([bs, -1, num_pt])

        if self.scale is None:
            max, _ = x.max(2)
            min, _ = x.min(2)
            self.scale, _ = torch.max(torch.maximum(max, torch.abs(min)), 1)
            self.scale = self.scale[:, None, None]
            # intv, _ = (max - min).max(1)
            # x = (x - max[:, :, None] - min[:, :, None])/intv[:, None, None]*(v_max - v_min) + v_min
            x /= self.scale
        else:
            x /= self.scale
            self.scale = None
            # from bpnet_utils.visual import plot_point_cloud
            # plot_point_cloud(x[2].cpu().numpy().T, x[2].cpu().numpy().T, np.eye(4))
            # import ipdb; ipdb.set_trace()
        x = x - x.mean(dim=2, keepdim=True)

        if reshape: x = x.view([bs, c, num_pt, num_gr])
        return x

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False)
        self.bn4 = nn.BatchNorm2d(256, track_running_stats=False)
        self.bn5 = nn.BatchNorm2d(emb_dims, track_running_stats=False)
    def forward(self, x, mn_list):
        batch_size, num_dims, num_points = x.size()
        mn1, mn2, mn3, mn4, mn5, mn6 = mn_list[0], mn_list[1], mn_list[2], mn_list[3], mn_list[4], mn_list[5]
        x = mn1(x)
        x = get_graph_feature(x)
        x = F.relu(self.bn1(mn2(self.conv1(x))))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(mn3(self.conv2(x))))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(mn4(self.conv3(x))))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(mn5(self.conv4(x))))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(mn6(self.conv5(x)))).view(batch_size, -1, num_points)
        return x

class BPNet(nn.Module):
    def __init__(self, emb_net=DGCNN, pointer_net=Transformer):
        super(BPNet, self).__init__()
        self.emb_nn = emb_net()
        self.mn_list = [MatchNorm(), MatchNorm(), MatchNorm(), MatchNorm(), MatchNorm(), MatchNorm()]
        self.pointer = pointer_net()
        self.outlier_bin = torch.nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.relu = nn.ReLU(inplace=True)
        self.lamb = 0.5

    @staticmethod
    def optimal_transport_layer(score, outlier_bin, lamb=np.nan, num_iters=50):
        bs, w, h = score.size()
        score *= (1/lamb)
        # add outlier bin to score map
        score_ = torch.cat([score, outlier_bin.repeat(bs, w, 1)], dim=2)
        score_ = torch.cat([score_, outlier_bin.repeat(bs, 1, h + 1)], dim=1)

        # init the a, b, f and g.
        a, b = torch.ones(w + 1).to(score), torch.ones(h + 1).to(score)
        a[-1], b[-1] = w, h
        log_a, log_b = a.log().repeat(bs, 1), b.log().repeat(bs, 1)
        f, g = torch.zeros_like(log_a), torch.zeros_like(log_b)

        for _ in range(num_iters):
            f = lamb * log_a - torch.logsumexp(score_ + g[:, None, :], dim=2)
            g = lamb * log_b - torch.logsumexp(score_ + f[..., None], dim=1)
        out = score_ + f[..., None] + g[:, None, :]
        return out.exp()

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        '''
        In the original bpnet, the input is scaled and voxelization_downsampled data,
        here we have to do the voxelization first and then doing the scaling and reduce_mean
        in the new normlization functions.'''
        src_embedding = self.emb_nn(src, self.mn_list)
        tgt_embedding = self.emb_nn(tgt, self.mn_list)
        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        src_embedding = self.relu(src_embedding)
        tgt_embedding = self.relu(tgt_embedding)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores_ot = self.optimal_transport_layer(scores, self.outlier_bin, self.lamb)
        return scores_ot

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
