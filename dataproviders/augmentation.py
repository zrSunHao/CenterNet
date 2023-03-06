from .compose import Compose as Compose
from .normalize import Normalize as Normalize
from .resize import Resize as Resize

class Augmentation(object):
    def __init__(self, img_size=512, mean=(0.406,0.456,0.485), std=(0.255, 0.224, 0.229)):
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.augment = Compose([
            Resize(self.img_size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)