import torch as t
import torch.nn as nn


class SPP(nn.Module):

    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        return t.cat([x, x_1, x_2, x_3], dim=1)
