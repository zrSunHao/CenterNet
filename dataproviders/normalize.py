import numpy as np

class Normalize(object):

    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32) 
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes, labels=None):
        image = image.astype(np.float32)
        image /= 255
        image -= self.mean
        image /= self.std
        return image, boxes, labels
