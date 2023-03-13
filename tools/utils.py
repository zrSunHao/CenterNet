import numpy as np

'''
确定高斯圆的最小半径
保留 IoU 与 GT(ground truth) 大于阈值 0.7 的预测值
用高斯圆来计算heatmap中标签范围详见：
    https://blog.csdn.net/x550262257/article/details/121289242
    https://zhuanlan.zhihu.com/p/452632600
    https://zhuanlan.zhihu.com/p/388024445
输入：
    det_size: 缩放后标注框的大小为 [box_w, box_y]
    min_overlap: 阈值
返回值：
    高斯圆的最小半径
'''
def gaussian_radius(det_size, min_overlap=0.7):
    box_w, box_h = det_size
    
    # 一角点在真值框内，一角点在真值框外
    a3 = 1
    b3 = 1 * (box_w + box_h)
    c3 = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2  

    # 两角点均在真值框内
    a2 = 4
    b2 = 2 * (box_w + box_h)
    c2 = box_w * box_h * (1 - min_overlap)
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2  

    # 两角点均在真值框外
    a1 = 4 * min_overlap
    b1 = -2 * min_overlap * (box_w + box_h)
    c1 = box_w * box_h * (min_overlap - 1)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2  

    return min(r1, r2, r3)


'''
将原始 bbox 标注映射到特征图上
输入：
    gt_label: round truth: [xmin, ymin, xmax, ymax, class_id]
    w, h: 输入图像的尺寸 512
    s: 缩放的尺度  4
输出：
    中心点坐标: (grid_x, grid_y)
    偏移量: (tx, ty)
    尺度: (tw, th)
    二维高斯函数方差: sigma_w, sigma_h
'''
def generate_txtytwth(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]

    '''
    计算中心点、高度、宽度
    这里坐标乘当前的 w、h 的原因：
        在 COCODataset 中,图片resize后,标注框box的坐标需调整
        坐标时除以了原始的 w、h
        公式为：xmin = xmin / 图片原始宽度 * 调整之后的图片宽度
    当前项目的公式为：
        x = x / 图片原始宽度 * 512
        y = y / 图片原始宽度 * 512
    '''
    c_x = (xmax * w + xmin * w) / 2 
    c_y = (ymax * h + ymin * h) / 2
    box_w = xmax * w - xmin * w
    box_h = ymax * h - ymin * h

    # 按尺度缩小
    c_x_s = c_x / s
    c_y_s = c_y / s
    box_w_s = box_w / s
    box_h_s = box_h / s

    r = gaussian_radius([box_w_s, box_h_s])
    # r / 3 的原因：正态分布 99.7 的权重都在正负 3sigma 范围内
    sigma_w = sigma_h = r / 3

    if box_w < 1e-28 or box_h < 1e-28:
        return False
    
    
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y

    tw = np.log(box_w_s)
    th = np.log(box_h_s)

    return grid_x, grid_y, tx, ty, tw, th, sigma_w, sigma_h


'''
创建高斯热力图，生成可用的标注信息
输入：
    input_size: 输入图像的尺寸 512
    stride: 缩放的步长  4
    classes_num: 类别总数 80
    label_list: 原始标记值 [B, x, 5]
输出：
    gt_tensor: 高斯热力图，(B, H*W, classes_num+1+1)
'''
def gt_creator(input_size, stride, classes_num, label_list=[]):
    batch_size = len(label_list)
    w = input_size  # 512
    h = input_size  # 512

    s = stride
    ws = w // s     # 尺寸缩放，向下取整
    hs = h // s     # 尺寸缩放，向下取整
    
    '''
    要输出的图片标注信息: 
        [B, H, W, classes_num + 4 + 1]
    即：
        [B, 128, 128, 80 + 4 + 1]
    '''
    gt_tensor = np.zeros([batch_size, hs, ws, classes_num + 4 + 1])

    for batch_index in range(batch_size):           # 每一个图像
        # label_list[batch_index]: [x, 5]
        for gt_label in label_list[batch_index]:    # 每一个标注,[5]
            gt_cls = gt_label[-1]                   # 该标注物体所属类别
            result = generate_txtytwth(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, sigma_w, sigma_h = result
                gt_tensor[batch_index, grid_y, grid_x, int(gt_cls)] = 1.0
                gt_tensor[batch_index, grid_y, grid_x, classes_num: classes_num + 4] = np.array([tx, ty, tw, th])
                gt_tensor[batch_index, grid_y, grid_x, classes_num + 4] = 1.0

                # 创建高斯热力图
                a1 = grid_x - 3 * int(sigma_w)
                a2 = grid_x + 3 * int(sigma_w)
                b1 = grid_y - 3 * int(sigma_h)
                b2 = grid_y + 3 * int(sigma_h)
                for i in range(a1, a2):
                    for j in range(b1, b2):

                        if i < ws and i < hs:
                            v = np.exp(- (i - grid_x)**2 / (2*sigma_w**2) - (j - grid_y)**2 / (2*sigma_h**2))
                            pre_v = gt_tensor[batch_index, j, i, int(gt_cls)]
                            gt_tensor[batch_index, j, i, int(gt_cls)] = max(v, pre_v)

    gt_tensor = gt_tensor.reshape(batch_size, -1, classes_num + 4 +1)
    return gt_tensor


