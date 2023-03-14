import torch as t
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights

from models.convlayer import ConvLayer
from models.decovlayer import DeCovLayer
from models.spp import SPP


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
    def forward(self, x):
        c5 = self.backbone(x)
        p5 = self.smooth(c5)
        p4 = self.deconv5(p5)
        p3 = self.deconv4(p4)
        p2 = self.deconv3(p3)
        
        cls_pred = self.cls_pred(p2)    # 输出:  [B, 80, 128, 128]
        txty_pred = self.txty_pred(p2)  # 输出:  [B, 2,  128, 128]
        twth_pred = self.twth_pred(p2)  # 输出:  [B, 2,  128, 128]

        return cls_pred, txty_pred, twth_pred


    