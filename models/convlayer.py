import torch.nn as nn

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=1
                              )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        m = self.conv(x)
        m = self.norm(m)
        m = self.relu(m)
        return m
