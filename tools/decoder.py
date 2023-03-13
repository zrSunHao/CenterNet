import torch as t

'''
模型的预测输出 ===> 直观的检测框信息
'''
def decode(pred):

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
    