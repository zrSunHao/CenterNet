from dataproviders import COCODataset,Augmentation
from configs import DefaultCfg 

cfg = DefaultCfg()
dataset = COCODataset(data_dir = cfg.data_root,
                      anno_dir = cfg.anno_dir,
                      name = cfg.train_dir,
                      transform=Augmentation(img_size = cfg.img_size,)
                      )
img = dataset.__getitem__(0)
print(img)
