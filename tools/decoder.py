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



def gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)



# 选取 topk 个满足要求的点
def get_topk(topk, scores):
    B, C, H, W = scores.size()
    topk_scores, topk_inds = t.topk(scores.view(B, C, -1), topk)
    topk_inds = topk_inds % (H *W)
    topk_score, topk_ind = t.topk(topk_scores.view(B, -1), topk_ind)
    topk_inds = gather_feat(topk_inds.view(B, -1, -1), topk).view(B, topk)
    top_clses = t.floor_divide(topk_ind, topk).int()
    return topk_score, topk_inds, top_clses
    