import cv2

# 调整图片大小
class Resize(object):
    def __init__(self,size = 300):
        self.size = size

    def __call__(self, img, boxes = None, labels = None):
        img = cv2.resize(img, (self.size, self.size))
        return img, boxes, labels