from .compose import Compose as Compose
from .normalize import Normalize as Normalize
from .resize import Resize as Resize

# 数据增强
class Augmentation(object):
    def __init__(self, img_size=512, mean=(0.406,0.456,0.485), std=(0.255, 0.224, 0.229)):
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.augment = Compose([            # 组合对数据的处理
            Resize(self.img_size),          # 调整图片大小
            Normalize(self.mean, self.std)  # 标准化
        ])

    '''
    这里只对 img 进行了处理
    boxes 和 labels直接返回，未进行处理
    '''
    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)