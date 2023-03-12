import numpy as np

'''
数据标准化，归一化
    提升模型的收敛速度
    提升模型的精度
    防止模型梯度爆炸
    https://blog.csdn.net/weixin_38313518/article/details/79950654
'''
class Normalize(object):

    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32) 
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255
        '''
        通过减去数据对应维度的统计平均值，来消除公共的部分，
        以凸显个体之间的特征和差异
        https://blog.csdn.net/qq_19329785/article/details/84569604
        '''
        image -= self.mean
        '''
        减去平均值，然后除以标准差，可以让正态分布的特征变为标准正态分布
        标准正态分布有很多优良特性。
        现在普遍直接用BN或者LN之类的操作代替简单归一化。
        '''
        image /= self.std   
        return image, boxes, labels
