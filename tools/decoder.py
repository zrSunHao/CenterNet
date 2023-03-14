import torch as t

'''
输入：
    txtytwth_pred  [B, 128 * 128, 4]
    最后的维度为：[cx, xy, w, h]
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
        ])
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
    # 因 x,y 坐标的值全为 0.xxx 所以需加位置坐标值 grid_cell
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
    