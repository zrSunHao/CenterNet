import torch.nn as nn

class DeCovLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeCovLayer, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        stride = stride,
                                        padding = 1,
                                        output_padding = 0)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        m = self.convT(x)
        m = self.norm(m)
        m = self.relu(m)
        return m