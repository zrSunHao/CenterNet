import torch as t
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18,ResNet18_Weights

from models.convlayer import ConvLayer
from models.decovlayer import DeCovLayer
from models.spp import SPP
from tools import get_loss,decode,get_topk


class CenterNet(nn.Module):

    def __init__(self, classes_num, topk):
        super(CenterNet, self).__init__()

        self.classes_num = classes_num      # 类别数
        self.topk = topk                    # 选择前 k 个符合要求的点

        '''
        特征提取的主干网，resnet
        输入： [B, 3, 512, 512]
        输出:  [B, 512, 16, 16]
        '''
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )
        
        '''
        使用了空间金字塔，实现局部特征与全局特征的相互融合
        提高了感受野
        输入:  [B, 512, 16, 16] 
        输出:  [B, 512, 16, 16] 
        '''
        self.smooth = nn.Sequential(
            SPP(),  # 输出:  [batch_size, 512*4, 16, 16] 
            ConvLayer(512*4, 256, kernel_size=1, padding=0),
            ConvLayer(256, 512, kernel_size=3, padding=1),
        )
        
        # 输入:  [B, 512, 16, 16]  输出:  [B, 256, 32, 32]
        self.deconv5 = DeCovLayer(512, 256, kernel_size=4, stride=2)
        # 输入:  [B, 256, 32, 32]  输出:  [B, 256, 64, 64]
        self.deconv4 = DeCovLayer(256, 256, kernel_size=4, stride=2)
        # 输入:  [B, 256, 64, 64]  输出:  [B, 256, 128, 128]
        self.deconv3 = DeCovLayer(256, 256, kernel_size=4, stride=2)

        # 输入:  [B, 256, 128, 128]  输出:  [B, 80, 128, 128]
        self.cls_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, self.classes_num, kernel_size=1)
        )
        # 输入:  [B, 256, 128, 128]  输出:  [B, 2, 128, 128]
        self.txty_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        # 输入:  [B, 256, 128, 128]  输出:  [B, 2, 128, 128]
        self.twth_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1)
        )

    '''
    x:  [B, 3, 512, 512]
    target: [B, 128 * 128, 85]
        80个类
        2个偏移量
        2个尺寸
        1个所属类别?
    '''
    def forward(self, x, target):
        c5 = self.backbone(x)
        B = c5.size(0)
        p5 = self.smooth(c5)
        p4 = self.deconv5(p5)
        p3 = self.deconv4(p4)
        p2 = self.deconv3(p3)
        
        cls_pred = self.cls_pred(p2)    # 输出:  [B, 80, 128, 128]
        txty_pred = self.txty_pred(p2)  # 输出:  [B, 2,  128, 128]
        twth_pred = self.twth_pred(p2)  # 输出:  [B, 2,  128, 128]

        # 热力图 [B, classes_num, 128, 128] ===> [B, 128 * 128, classes_num]
        cls_pred = cls_pred.permute(0, 2, 3, 1) \
                           .contiguous() \
                           .view(B, -1, self.classes_num)
        # 中心点偏移 [B, 2, 128, 128]  ===> [B, 128 * 128, 2]
        txty_pred = txty_pred.permute(0, 2, 3, 1) \
                             .contiguous() \
                             .view(B, -1, 2)
        # 物体尺度 [B, 2, 128, 128]  ===> [B, 128 * 128, 2]
        twth_pred = twth_pred.permute(0, 2, 3, 1) \
                             .contiguous() \
                             .view(B, -1, 2)

        # 计算损失函数
        total_loss = get_loss(
            pre_cls = cls_pred, 
            pre_txty = txty_pred, 
            pre_twth = twth_pred, 
            label = target, 
            classes_num = self.classes_num
        )
        return total_loss


    # 生成候选框，并进行可视化
    def generate(self, x):
        c5 = self.backbone(x)       # resnet18 提取特征
        B = c5.size(0)              # batch_size
        p5 = self.smooth(c5)        # 
        p4 = self.deconv5(p5)       #
        p3 = self.deconv4(p4)
        p2 = self.deconv3(p3)

        cls_pred = self.cls_pred(p2)        # 类别热力图，预测结果
        cls_pred = t.sigmoid(cls_pred)
        txty_pred = self.txty_pred(p2)      # 偏移量预测结果
        twth_pred = self.twth_pred(p2)      # 尺度预测结果

        # 寻找 8-近邻极大值点，其中 keypoints 为极大值点的位置
        # cls_pred 为对应的极大值点
        hmax = nn.functional.max_pool2d(cls_pred, 
                                        kernel_size = 5, 
                                        padding = 2,
                                        stride = 1)
        keep = (hmax == cls_pred).float()
        cls_pred = keep

        txtytwth_pred = t.cat([txty_pred, twth_pred], dim = 1) \
                            .permute(0, 2, 3, 1) \
                            .contiguous() \
                            .view(B, -1, 4)
        
        scale = np.array([[[512, 512, 512, 512]]])
        scale_t = t.tensor(scale.copy(), device=txtytwth_pred.device).float()
        pre = decode(txtytwth_pred/scale_t)
        pre = pre / scale_t
        bbox_pred = t.clamp(pre[0], 0., 1.)
        
        # 得到 topk 取值， top_score：置信度，top_ink：index，topk_clses：类别
        topk_score, topk_ind, top_clses = get_topk(cls_pred)
        topk_bbox_pred = bbox_pred[topk_ind[0]]

        topk_bbox_pred = topk_bbox_pred.cpu().numpy()
        topk_score = topk_score[0].cpu().numpy()
        top_clses = top_clses[0].cpu().numpy()
        return topk_bbox_pred, topk_score, top_clses

    