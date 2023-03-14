import os
import numpy as np
import torch as t
import cv2

from models import CenterNet
from configs import DefaultCfg 
from dataproviders import Augmentation

# 配置
cfg = DefaultCfg()
device = t.device(cfg.device)

class_labels = cfg.coco_class_labels
class_index = cfg.coco_class_index
class_color = []
for _ in range(cfg.classes_num):
    color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
    class_color.append(color)


# 加载预训练模型
#assert not cfg.net_path is None
model_ft = CenterNet(classes_num = cfg.classes_num,
                     topk = cfg.topk)
# model_path = os.path.join(cfg.models_root, cfg.net_path)
# if os.path.exists(model_path):
#     state_dict = t.load(model_path)
#     model_ft.load_state_dict(state_dict)
model_ft.to(device)

transform = Augmentation()
for index, file in enumerate(os.listdir(cfg.test_img_dir)):
    path = cfg.test_img_dir + '/' + file
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = transform(img, boxes=None, labels=None)

    x = t.from_numpy(img[0][:,:,(2,1,0)]).permute(2,0,1)
    # [3, 512, 512] ===> [1, 3, 512, 512]
    x = x.unsqueeze(0).to(device)

    '''
    bbox_pred:  
        关键点边框坐标 [100, 4]
        最后的维度为：[xmin, ymin, xmax, ymax]
    scores:  topk 个关键点置信度 [topk]
    clses:  topk 个关键点所属类别的索引编码 [topk]
    '''
    bbox_pred, scores, cls_ind = model_ft.generate(x)
    '''
    img.shape 
        [h, w, c]
    当前的坐标值为
        x现 / 512 = x原 / 原始宽度
        y现 / 512 = y原 / 原始高度
    乘以原宽高可得：
        x原 = x现/512 * 原始宽度
        y原 = y现/512 * 原始高度
    '''
    bbox_pred = bbox_pred * np.array([[img.shape[1],
                                       img.shape[0],
                                       img.shape[1],
                                       img.shape[0]]])
    
    # 循环标注 topk 个点
    for i,box in enumerate(bbox_pred):
         # 只有达到置信度的最低阈值，才标注目标
        if scores[i] > cfg.score:

            # 关键点所属类别的索引
            cls_indx = cls_ind[i]
            cls_id = class_index[int(cls_indx)]
            cls_name = class_labels[cls_id]
            label = '%s:%.3f' % (cls_name,scores[i])

            # 标注框两个角点的坐标
            xmin, ymin, xmax, ymax = box
            # 坐标取整
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            # 标注框的颜色
            color = class_color[int(cls_indx)]
            cv2.rectangle(img, pt1, pt2, color, thickness=2)

            # 标签标注框的两个角点
            pt1 = (int(xmin), int(abs(ymin) - 15)) # 左上角
            pt2 = (int(xmin + int(xmax-xmin) * 0.55), int(ymin)) # 右下角
            color = class_color[int(cls_indx)]
            # thickness为负值，表示填充整个矩形
            cv2.rectangle(img, pt1, pt2, color, thickness=-1)

            cv2.putText(img, 
                        text = label, 
                        org = (int(xmin),int(ymin)),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.5,
                        color = (0,0,0),
                        thickness = 2
                        )
    
    img_path = os.path.join(cfg.test_out_dir, str(index).zfill(3)+'.jpg')
    cv2.imwrite(img_path, img)


