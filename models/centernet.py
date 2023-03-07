import torch as t
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18

from models.convlayer import ConvLayer
from models.decovlayer import DeCovLayer
from models.spp import SPP
from tools import get_loss


class CenterNet(nn.Module):

    def __init__(self, classes_num, topk):
        super(CenterNet, self).__init__()

        self.classes_num = classes_num
        self.topk = topk

        self.backbone = resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            *list(self.backbone.children())
        )
        self.smooth = nn.Sequential(
            SPP(),
            ConvLayer(512*4, 256, kernel_size=1, padding=0),
            ConvLayer(256, 512, kernel_size=3, padding=1),
        )
        
        self.deconv5 = DeCovLayer(512, 256, kernel_size=4, stride=2)
        self.deconv4 = DeCovLayer(256, 256, kernel_size=4, stride=2)
        self.deconv3 = DeCovLayer(256, 256, kernel_size=4, stride=2)

        self.cls_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        self.twth_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1)
        )

    def decode(self, pred):

        output = t.zeros_like(pred)
        grid_y, grid_x = t.meshgrid([
            t.arange(128, device=pred.device),
            t.arange(128, device=pred.device)
            ])
        grid_cell = t.stack([grid_x,grid_y], dim=1).float().view(1, 128*128, 2)
        pred[:,:,:2] = 4 * (t.sigmoid(pred[:,:,:2]),+ grid_cell)
        pred[:,:,2:] = 4 * (t.exp(pred[:,:,2:]))

        # 坐标转换 [cx, xy, w, h] -> [xmin, ymin, xmax, ymax]
        output[:,:,0] = pred[:,:,0] - pred[:,:,2] / 2
        output[:,:,1] = pred[:,:,1] - pred[:,:,3] / 2
        output[:,:,2] = pred[:,:,0] - pred[:,:,2] / 2
        output[:,:,3] = pred[:,:,1] - pred[:,:,3] / 2

        return output

    def gather_feat(self, feat, ind):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        return feat.gather(1, ind)

    # 选取 topk 个满足要求的点
    def get_topk(self, scores):
        B, C, H, W = scores.size()
        topk_scores, topk_inds = t.topk(scores.view(B, C, -1), self.topk)
        topk_inds = topk_inds % (H *W)
        topk_score, topk_ind = t.topk(topk_scores.view(B, -1), topk_ind)
        topk_inds = self.gather_feat(topk_inds.view(B, -1, -1), self.topk).view(B, self.topk)
        top_clses = t.floor_divide(topk_ind, self.topk).int()
        return topk_score, topk_inds, top_clses
    
    def generate(self, x):
        pass

    def forward(self,x):
        pass