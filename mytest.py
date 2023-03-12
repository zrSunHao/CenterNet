import torch as t
import torch.nn as nn
from torchvision.models import resnet18

a = t.range(1, 96, 1).view(2, 3, 4, 4)
print(a)

b = a.view(2, 3, -1)
print(b, '\n')

top, inds = t.topk(b, 1)
print(top, '\n')
print(inds, '\n')

tp, ind = t.topk(top.view(2, -1), 2)
print(tp, '\n')
print(ind, '\n')

t1 = t.rand((8, 512, 1, 1))
t2 = t.rand((8, 512, 1, 1))
t3 = t.rand((8, 512, 1, 1))
t4 = t.rand((8, 512, 1, 1))
d = t.cat([t1, t2, t3, t4], dim=1)
print(d.size(), '\n')


a = t.tensor([[1, 2, 3], [1, 2, 3]])
b = t.tensor([[4, 5, 6], [4, 5, 6]])
c = t.cat([a, b], dim=1)
print(c, '\n')

x = t.tensor([[-0.4311, -0.1024, -0.0550,   0.937, -0.2043, -0.2630],
              [-0.4245, -0.1128, -0.0431,   0.0879, -0.2021, -0.2569],
              [-0.4231, -0.1122, -0.0431,  0.0845, -0.1974, -0.2506],])
print(t.sigmoid(x))
