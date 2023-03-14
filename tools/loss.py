import torch as t
import torch.nn as nn

'''
关键点，热力图的损失函数：
    监督网络对中心点学习的效果
    是对 Focal Loss 改进的
'''
class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss,self).__init__()
    
    '''
    inputs:  [B, 128 * 128, classes_num]
    targets: [B, 128 * 128, classes_num]   80个类
    '''
    def forward(self, inputs, targets):
        # 激活函数，映射到 (0,1) 之间
        inputs = t.sigmoid(inputs)
        # [B, 128 * 128, classes_num]
        center_id = (targets == 1.0).float()
        # [B, 128 * 128, classes_num]
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0 -inputs)**2 * t.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**4 * (inputs)**2 * t.log(1.0 - inputs + 1e-14)

        return center_loss + other_loss


'''
损失函数
参数：
    pre_cls:    [B, 128 * 128, classes_num]
    pre_txty:   [B, 128 * 128, 2]
    pre_twth:   [B, 128 * 128, 2]
    label:      [B, 128 * 128, 85]
    classes_num: 80
'''
def get_loss(pre_cls, pre_txty, pre_twth, label, classes_num):
    cls_loss_function = FocalLoss()
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')

    '''
    获取标记框 gt
    gt_cls:       [B, 128 * 128, classes_num]
    gt_txtytwth:  [B, 128 * 128, 4]
    gt_box_scale_weight:[B, 128 * 128, 1]
    '''
    gt_cls = label[:, :, :classes_num].float()
    gt_txtytwth = label[:, :, classes_num:-1].float()
    # 关键点的位置为 1，其余点为 0
    gt_box_scale_weight = label[:, :, -1]

    # 中心点热力图损失 L_k
    batch_size = pre_cls.size(0)
    cls_loss = cls_loss_function(pre_cls, gt_cls)
    cls_loss = t.sum(cls_loss) / batch_size

    # 中心点偏移量损失 L_off
    txty_loss = txty_loss_function(pre_txty, gt_txtytwth[:, :, :2])
    # 合并标号为2的维度 [B, 128 * 128, 4] ===> [B, 128 * 128]
    txty_loss = t.sum(txty_loss, dim=2)
    # [B, 128 * 128] * [B, 128 * 128] ===> [B, 128 * 128]
    txty_loss = txty_loss * gt_box_scale_weight
    # [B, 128 * 128] ===> 1
    txty_loss = t.sum(txty_loss)
    txty_loss = txty_loss / batch_size

    # 物体尺度损失 L_size
    twth_loss = twth_loss_function(pre_twth, gt_txtytwth[:, :, 2:])
    # 合并标号为2的维度 [B, 128 * 128, 2] ===> [B, 128 * 128]
    twth_loss = t.sum(twth_loss, dim=2)
    # [B, 128 * 128] * [B, 128 * 128] ===> [B, 128 * 128]
    twth_loss = twth_loss * gt_box_scale_weight
    # [B, 128 * 128] ===> 1
    twth_loss = t.sum(twth_loss)
    twth_loss = twth_loss / batch_size

    # 总损失
    total_loss = cls_loss + txty_loss + twth_loss
    return total_loss
