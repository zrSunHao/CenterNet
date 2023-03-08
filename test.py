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
assert not cfg.net_path is None
model_ft = CenterNet(classes_num = cfg.classes_num,
                     topk = cfg.topk)
model_path = os.path.join(cfg.models_root, cfg.net_path)
if os.path.exists(model_path):
    state_dict = t.load(model_path)
    model_ft.load_state_dict(state_dict)
model_ft.to(device)

transform = Augmentation()
for index,file in enumerate(os.listdir(cfg.test_img_dir)):
    path = cfg.test_img_dir+'/'+file
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    x = t.from_numpy(transform(img,boxes=None,labels=None)[0][:,:,(2,1,0)]).permute(2,0,1)
    x = x.unsqueeze(0).to(device)

    bbox_pred,score,cls_ind = model_ft.generate(x)
    bbox_pred = bbox_pred * np.array([[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]])
    
    for i,box in enumerate(bbox_pred):
        if score[i]>0.35:
            cls_indx = cls_ind[i]
            cls_id = class_index[int(cls_indx)]
            cls_name = class_labels[cls_id]
            label = '%s:%.3f' % (cls_name,score[i])
            xmin,ymin,xmax,ymax = box

            cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),class_color[int(cls_indx)],2)
            cv2.rectangle(img,(int(xmin),int(abs(ymin)-15)),(int(xmin+int(xmax-xmin)*0.55),int(ymin)),class_color[int(cls_indx)],-1)
            cv2.putText(img,label,(int(xmin),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
    
    cv2.imwrite(os.path.join(cfg.test_out_dir,str(index).zfill(3)+'.jpg'),img)


