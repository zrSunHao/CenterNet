'''
对数据处理的组合
'''
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, boxes = None, labels = None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img,boxes,labels
        