import torch

import torch.nn as nn

import torch.nn.functional as F


def avg_pooling(x: torch.Tensor):
    return F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze()

def initialize_weights(net):
    """
    initialize network
    note:It's different to initialize discriminator and classifier.
    For detail,please check the initialization of resnet and wgan-gp.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class BatchNorm(nn.Module):
    """
    BatchBorm block: custom_conv-bn-leaky_relu where downsampling is performed if input channels is not equal to output channels。
    otherwise,the input size is kept.
    """
    def __init__(self, in_channels, out_channels):
        super(BatchNorm, self).__init__()
        stride = 1 if in_channels == out_channels else 2
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))
