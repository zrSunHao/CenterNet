import os
import numpy as np
import cv2
import torch as t
from torch.utils.data import Dataset

# 安装命令 conda install -c conda-forge pycocotools
from pycocotools.coco import COCO


'''
COCO 数据集类
提供一种方式去获取数据，以及其对应的真实Label
'''
class COCODataset(Dataset):

    '''
    COCO 数据集初始化
    将标注的数据通过 COCO API 读入到内存中
    参数:
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

        # 加载 coco 数据集的批注信息
        self.coco = COCO(self.data_dir + anno_dir + json_file)
        # 获取图像的 Id 信息
        self.ids = self.coco.getImgIds()
        # 获取图像的所属类别信息
        self.class_ids = sorted(self.coco.getCatIds())
        

    def __len__(self):
        return len(self.ids)

    '''
    获取一张图像的数据以及标注信息
    输入参数：
        图片索引
    输出结果：
        图像数据 tensor ()
        图像标注信息 (xmin, ymin, xmax, ymax, label)
    '''
    def __getitem__(self, index):

        # 获取图像 id
        id_ = self.ids[index]
        imgIds = [int(id_)]
        # 获取图像标注信息（一个图像有多个标注信息）
        anno_ids = self.coco.getAnnIds(imgIds = imgIds, iscrowd = None)
        annotations = self.coco.loadAnns(anno_ids)

        # 加载图像数据
        img_file = os.path.join(self.data_dir, self.name, '{:012}'.format(id_)+'.jpg')
        img = cv2.imread(img_file)
        assert img is not None
        height, width, channels = img.shape

        '''
        预处理，将 bbox 原始标注 [xmin, ymin, w, h] 
        转化为：
            [xmin, ymin, xmax, ymax, label]
        每个标注信息包含：
            id: 标注 Id
            image_id: 图像 Id
            segmentation: 目标区域分割点的信息
            area: 图像目标区域的面积
            iscrowd: 目标区域是否冲别
            bbox: 目标区域的框
            category_id: 目标区域物体的类别
        '''
        target = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:
                bbox = anno['bbox']   # 原始标注 [xmin, ymin, w, h] 
                xmin = np.max((0, bbox[0]))
                ymin = np.max((0, bbox[1]))
                w = np.max((0, bbox[2]))
                h = np.max((0, bbox[3]))
                xmax = np.min((width -1, xmin + w - 1))
                ymax = np.min((height -1, ymin + h - 1))
                if xmax > xmin and ymax > ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)   # 类别索引
                    '''
                    图片resize后，标注框box的坐标需调整
                    如 xmin = xmin / 图片原始宽度 * 调整之后的图片宽度
                    此处 *调整之后的图片宽度 计算
                        在 utils 的 generate_txtytwth 函数中有体现 
                    https://blog.csdn.net/qq_28949847/article/details/106154492
                    '''
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
        
        # 图像数据增强，了解此处的transform，去看 Augmentation 类即可
        img, boxes, labels = self.transform(img, target[:,:4], target[:,4]) 
        # cv2 读取的通道为 bgr，需转为 rgb
        img = img[:, :, (2,1,0)]
        labels = np.expand_dims(labels, axis=1)
        target = np.hstack((boxes, labels))
        # hwc ===> chw
        img = t.from_numpy(img).permute(2, 0, 1)
        return img, target
