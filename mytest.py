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

t1 = t.rand((8,512,1,1))
t2 = t.rand((8,512,1,1))
t3 = t.rand((8,512,1,1))
t4 = t.rand((8,512,1,1))
d = t.cat([t1,t2,t3,t4],dim=1)
print(d.size(), '\n')

# backbone = resnet18(pretrained=True)
# backbone = nn.Sequential(
#             *list(backbone.children())
#         )
#print(backbone)

tw = t.tensor([1,2,3,4,5,6])
print(tw, '\n')
tw = tw / 2
tw = tw.int().float()
print(tw)
