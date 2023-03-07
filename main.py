import os
import torch as t
import torchvision as tv


from dataproviders import COCODataset,Augmentation
from configs import DefaultCfg 
from tools import gt_creator
import numpy as np

cfg = DefaultCfg()
# dataset = COCODataset(data_dir = cfg.data_root,
#                       anno_dir = cfg.anno_dir,
#                       name = cfg.train_dir,
#                       transform=Augmentation(img_size = cfg.img_size,)
#                       )
# img = dataset.__getitem__(0)

#