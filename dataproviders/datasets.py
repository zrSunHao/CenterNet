import os
import numpy as np
import cv2
import torch as t
import torchvision as tv
from torch.utils.data import Dataset
# 安装命令 conda install -c conda-forge pycocotools
from pycocotools.coco import COCO

'''
COCO dataset class.
'''
class COCODataset(Dataset):

    '''
    COCO dataset initialization.
    Annotation data are read into memory by COCO API.
    Args:
        data_dir (str): dataset root directory.
        json_file (str): COCO json file name.
        name (str): COCO data name (e.g.  'train2017' or 'val2017').
        min_size (int): bounding boxes smaller than this are ignored.
    '''
    def __init__(self,
                 data_dir='./data/',
                 anno_dir='annotations/',
                 json_file='instances_train2017.json',
                 name='train2017',
                 transform=None,
                 min_size=1):

        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.max_labels = 50
        self.min_size = min_size
        assert transform is not None
        self.transform = transform
        self.coco = COCO(self.data_dir + anno_dir + json_file)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        # 获取图片 id，以及图片标注
        id_ = self.ids[index]
        imgIds = [int(id_)]
        anno_ids = self.coco.getAnnIds(imgIds = imgIds, iscrowd = None)
        annotations = self.coco.loadAnns(anno_ids)
        print(id_)
        # 读取图像，做预处理
        img_file = os.path.join(self.data_dir, self.name, '{:012}'.format(id_)+'.jpg')
        img = cv2.imread(img_file)
        assert img is not None
        height, width, channels = img.shape

        # 预处理，将 bbox 原始标注 [xmin, ymin, w, h] 转化为：
        # [xmin, ymin, xmax, ymax, label]
        target = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:
                bbox = anno['bbox']
                xmin = np.max((0, bbox[0]))
                ymin = np.max((0, bbox[1]))
                w = np.max((0, bbox[2]))
                h = np.max((0, bbox[3]))
                xmax = np.min((width -1, xmin + w - 1))
                ymax = np.min((height -1, ymin + h - 1))
                if xmax > xmin and ymax > ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)
                    xmin /= width
                    ymin /= height
                    xmax /= width
                    ymax /= height
                    target.append([xmin, ymin, xmax, ymax, cls_id])
            else:
                print('No bbox !!!')
        
        if len(target) == 0:
            target = np.zeros([1, 5])
        else:
            target = np.array(target)
        
        img, boxes, labels = self.transform(img, target[:,:4], target[:,4]) 
        # cv2 读取的通道为 bgr，需转为 rgb
        img = img[:, :, (2,1,0)]
        labels = np.expand_dims(labels,axis=1)
        target = np.hstack((boxes, labels))

        return t.from_numpy(img).permute(2, 0, 1), target
