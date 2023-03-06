import visdom
import time
import numpy as np

'''
封装了 Visdom 的基本操作，但是仍然可以通过 self.vis.function
或者 self.function 调用原生的 Visdom 接口，比如：
self.text('hello visdom')
self.histogram(t.randn(1000))
self.line(t.arange(0,10), t.arange(1,11))
'''
class Visualizer(object):

    def __init__(self, env='GAN', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False,**kwargs)
        self.index = {}  # 记录待绘制点的下标，例如：保存('loss',23)，即loss的第23个点
        self.log_text = ''

    '''
    修改 Visdom 的配置
    '''
    def reinit(self, env='GAN', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    '''
    绘制多个数据点
    @params d: dict(name,value) i.e. ('loss',0.11)
    '''
    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    '''
    绘制多张图片
    '''
    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    '''
    self.plot('loss',1.00)
    '''
    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs)
        self.index[name] = x + 1

    '''
    self.img('input_img',t.Tensor(64,64))
    self.img('input_img',t.Tensor(3,64,64))
    self.img('input_imgs',t.Tensor(100,1,64,64))
    self.img('input_img',t.Tensor(100,3,64,64))
    '''
    def img(self, name, img_, **kwargs):
        self.vis.images(img_.cpu().numpy(),
                        win=(name),
                        opts=dict(title=name),
                        **kwargs)

    '''
    self.log({'loss':1,'lr':0.0001})
    '''
    def log(self, info, win='log_text'):
        self.log_text += (
            '[{time}]{info}<br>'.format(
                time=time.strftime('%m%d_%H%M%S', info=info))
        )
        self.vis.text(self.log_text, win)

    '''
    自定义的 plot、image、log、plot_many 等方法除外
    self.function 等价于 self.vis.function
    '''
    def __getattr__(self, name):
        return getattr(self.vis, name)
