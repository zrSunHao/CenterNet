import torch as t
import torch.nn as nn

'''
空间金字塔
此处的作用：
    增强特征图的表达能力
原理：
    通过组合不同尺度的特征图信息，
    实现局部特征与全局特征的相互融合，
    以此提高感受野
输入:  [batch_size, 512, 16, 16]
输出:  [batch_size, 512*4, 16, 16]
'''
class SPP(nn.Module):

    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        # 输入: [batch_size, 512, 16, 16]  输出: [batch_size, 512, 16, 16]
        x_1 = nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        # 输入: [batch_size, 512, 16, 16]  输出: [batch_size, 512, 16, 16]
        x_2 = nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        # 输入: [batch_size, 512, 16, 16]  输出: [batch_size, 512, 16, 16]
        x_3 = nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        # 大小为: [batch_size, 512*4, 16, 16]
        output = t.cat([x, x_1, x_2, x_3], dim=1)
        return output
