import torch as t
import torch.nn as nn
import numpy as np

'''
输入：
    txtytwth_pred  [B, 128 * 128, 4]
    最后的维度为：[cx, cy, w, h]
当前部分代码仅支持一张图片检测，即：
    B(batch_size) = 1
作用：
    模型的预测输出 ===> 直观的检测框信息
输出：
    txtytwth_pred  [B, 128 * 128, 4]
    最后的维度为：[xmin, ymin, xmax, ymax]
'''
def decode_lxlyrxry(pred):

    output = t.zeros_like(pred)   # [B, 128 * 128, 4]
    # [128 * 128], [128 * 128]
    grid_y, grid_x = t.meshgrid([
        t.arange(128, device=pred.device),
        t.arange(128, device=pred.device)
        ],indexing = 'ij')
    '''
    加载每个点的坐标时上
    stack [128 * 128] [128 * 128] ===> [128, 2, 128] ===> [1, 128*128, 2]
    grid_cell:
        第 1 行：[0,0],[1,0],[2,0]······[125,0],[126,0],[127,0]
        第 2 行：[0,1],[1,1],[2,1]······[125,1],[126,1],[127,1]
        第 3 行：[0,3],[1,3],[2,3]······[125,3],[126,3],[127,3]
        ······
        第 128 行：[0,127],[1,127],[2,127]······[125,127],[126,127],[127,127]
    '''
    grid_cell = t.stack([grid_x, grid_y], dim=-1).float().view(1, 128*128, 2)
    
    pred[:,:,:2] = t.sigmoid(pred[:,:,:2])
    # 偏置需加位置坐标值 grid_cell，后面计算中心点时可得真正的坐标
    pred[:,:,:2] = 4 *(pred[:,:,:2] + grid_cell)
    # TODO 与 encoder generate_txtytwth 函数中的 tw、th 的求值运算互反
    pred[:,:,2:] = 4 * (t.exp(pred[:,:,2:]))

    # 坐标转换 [cx, xy, w, h] -> [xmin, ymin, xmax, ymax]
    output[:,:,0] = pred[:,:,0] - pred[:,:,2] / 2 # xmin = cx - w/2
    output[:,:,1] = pred[:,:,1] - pred[:,:,3] / 2 # ymin = cx - h/2
    output[:,:,2] = pred[:,:,0] + pred[:,:,2]     # xmax = xmin + w
    output[:,:,3] = pred[:,:,1] + pred[:,:,3]     # ymax = ymin + h

    return output


'''
在 dim 维度上，按照 indexs 所给的坐标选择元素
参数：
    feat: [B, 80 * topk, 1]  temp_cls_inds
    ind:  [B, topk]          img_topk_inds
返回：
    [B, topk, 1]
'''
def gather_feat(feat, ind):
    
    B = ind.size(0)     # B
    topk = ind.size(1)  # topk
    dim = feat.size(2)  # 1
    
    # [B, topk] ==> [B, topk, 1]
    ind = ind.unsqueeze(2)
    # [B, topk, 1] ===> [B, topk, 1]
    ind = ind.expand(B, topk, dim)
    # [B, 80 * topk, 1] [B, topk, 1] ===> [B, topk, 1]
    return feat.gather(1, ind)


'''
选取 topk 个满足要求的点
参数
    topk:符合要求点的个数
    cls_pred: 各类别关键点[B, 80, 128, 128]，关键点的值为 1
返回
    img_topk_scores: topk个关键点每个的置信度[B, topk]
    topk_inds: 图 128 * 128 一维张量上的 topk 个关键点的索引 [B, topk]
    top_clses: topk个关键点每个的类别 [B, topk] 
'''
def get_topk(topk, scores):
    B, C, H, W = scores.size()
    '''
    【每张图片】的【每个类别】各取置信度最高的 topk 个点
    '''
    # [B, 80, 128 * 128]
    cls_scores = scores.view(B, C, -1)
    # [B, 80, topk] [B, 80, topk]
    cls_topk_scores, cls_topk_inds = t.topk(cls_scores, topk)
    # 索引取余，防止超出范围
    cls_topk_inds = cls_topk_inds % (H * W)

    '''
    【每张图片】各取置信度最高的 topk 个点
    '''
    # [B, 80, topk] ===> [B, 80 * topk]
    temp_cls_scores = cls_topk_scores.view(B, -1)
    # [B, topk] [B, topk]
    img_topk_scores, img_topk_inds = t.topk(temp_cls_scores, topk)

    '''
    获取每张图 topk 个点的索引，在 [80, 128 * 128] 的尺度上
    在每张图片的 128 * 128 一维张量上的 topk 个点的索引
    '''
    # [B, 80, topk] ===> [B, 80 * topk, 1]
    temp_cls_inds = cls_topk_inds.view(B, -1, 1)
    # [B, 80 * topk, 1] [B, topk] ===> [B, topk, 1] ===> [B, topk]
    topk_inds = gather_feat(temp_cls_inds, img_topk_inds).view(B, topk)
    '''
    索引/topk 向下取整 [B, topk] ===> [B, topk]
    除完之后，可得topk个关键点每个的类别
    '''
    top_clses = t.floor_divide(img_topk_inds, topk).int()
    return img_topk_scores, topk_inds, top_clses


'''
生成候选框，并进行可视化
当前部分代码仅支持一张图片检测，即：
    B(batch_size) = 1
输入：
    cls_pred:  [B, 80, 128, 128]
    txty_pred: [B, 2,  128, 128]
    twth_pred: [B, 2,  128, 128]
输出：
    topk_bbox_pred:  关键点边框坐标 [100, 4]
    topk_scores:  topk 个关键点置信度 [topk]
    top_clses:  topk 个关键点所属类别 [topk]
'''
def decode_bbox(cls_pred, txty_pred, twth_pred, topk):
    B = cls_pred.size(0)              # batch_size
    # 类别热力图, 预测结果, 输出:  [B, 80, 128, 128]
    cls_pred = t.sigmoid(cls_pred)

    '''
    寻找 8-近邻极大值点，其中 keypoints 为极大值点的位置
    cls_pred 为对应的极大值点
    '''
    hmax = nn.functional.max_pool2d(cls_pred, 
                                    kernel_size = 5, 
                                    padding = 2,
                                    stride = 1)
    # [B, 80, 128, 128]，相等的值为 1
    keep = (hmax == cls_pred).float()
    cls_pred *= keep
    
    '''
    bbox坐标转换
    '''
    # [B, 4, 128, 128] ===> [B, 128, 128, 4] ===> [B, 128 * 128, 4]
    txtytwth_pred = t.cat([txty_pred, twth_pred], dim = 1) \
                        .permute(0, 2, 3, 1) \
                        .contiguous() \
                        .view(B, -1, 4)
    # [1, 1, 4]
    scale = np.array([[[512, 512, 512, 512]]])
    scale_t = t.tensor(scale.copy(), device=txtytwth_pred.device).float()
    # [B, 128 * 128, 4] 最后的维度为：[xmin, ymin, xmax, ymax]
    pre = decode_lxlyrxry(txtytwth_pred)
    # 这里是坐标点的转换，得坐标点为： x现/512 = x原/原始宽度
    pre = pre / scale_t
    # 夹紧区间，避免有在 [0,1] 区间之外的值 [128 * 128, 4]
    bbox_pred = t.clamp(pre[0], min = 0., max = 1.)
    
    '''
    得到 topk 取值、索引、类别
    topk_scores: topk个关键点每个的置信度[B, topk]
    topk_inds: topk个关键点每个的索引 [B, topk]
    top_clses: topk个关键点每个的类别 [B, topk] 
    '''
    topk_scores, topk_ind, top_clses = get_topk(topk, cls_pred)
    '''
    [100, 4]，获取关键点上预测的边框坐标，
    最后的维度为：[xmin, ymin, xmax, ymax]
    '''
    topk_bbox_pred = bbox_pred[topk_ind[0]]

    # [topk, 4]
    topk_bbox_pred = topk_bbox_pred.cpu().detach().numpy()
    # [topk]
    topk_scores = topk_scores[0].cpu().detach().numpy()
    # [topk]
    top_clses = top_clses[0].cpu().detach().numpy()
    
    return topk_bbox_pred, topk_scores, top_clses

