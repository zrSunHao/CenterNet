import torch as t
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss,self).__init__()
    
    def forward(self, inputs, targets):
        inputs = t.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0 -inputs)**2 * t.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**4 * (inputs)**2 * t.log(1.0 - inputs + 1e-14)
        return center_loss + other_loss


def get_loss(pre_cls, pre_txty, pre_twth, label, classes_num):
    cls_loss_function = FocalLoss()
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')

    # 获取标记框 gt
    gt_cls = label[:, :, :classes_num].float()
    gt_txtytwth = label[:, :, classes_num:-1].float()
    gt_box_scale_weight = label[:, :, -1]

    # 中心点热力图损失 L_k
    batch_size = pre_cls.size(0)
    cls_loss = cls_loss_function(pre_cls, gt_cls)
    cls_loss = t.sum(cls_loss) / batch_size

    # 中心点偏移量损失 L_off
    txty_loss = txty_loss_function(pre_txty, gt_txtytwth[:, :, :2])
    txty_loss = t.sum(txty_loss, 2)
    txty_loss = t.sum(txty_loss * gt_box_scale_weight)
    txty_loss = txty_loss / batch_size

    # 物体尺度损失 L_size
    twth_loss = twth_loss_function(pre_twth, gt_txtytwth[:, :, 2:])
    twth_loss = t.sum(twth_loss, 2)
    twth_loss = t.sum(twth_loss * gt_box_scale_weight)
    twth_loss = twth_loss / batch_size

    # 总损失
    total_loss = cls_loss + txty_loss + twth_loss
    return total_loss
