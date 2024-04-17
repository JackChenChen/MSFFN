from torch import nn
from networks.ops import BatchNorm, initialize_weights, avg_pooling
import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64, depth=7, downsampling=4, output_dim=35):
        super(UNet, self).__init__()

        # c
        self.start_filts = 64
        self.input_dim = 1
        self.size = 128
        self.depth = depth
        self.output_dim = output_dim
        self.downsampling = downsampling

        self.final_outs = self.start_filts
        down_conv = []
        down_conv.append(BatchNorm(self.input_dim, self.start_filts))

        ins = 64
        for i in range(1, self.downsampling):
            self.outs = self.start_filts * (2 ** i)
            down_conv.append(BatchNorm(ins, self.outs))
            ins = self.outs
        self.final_outs += self.outs
        self.down_convs = nn.Sequential(*down_conv)

        convs = []
        for _ in range(self.depth - self.downsampling):
            convs.append(BatchNorm(self.outs, self.outs))
        self.final_outs += self.outs
        self.convs = nn.Sequential(*convs)
        self.last_size = math.ceil(self.size / 2 ** self.downsampling)

        self.fc = nn.Sequential(
            nn.Linear(self.final_outs, self.output_dim)
        )
        initialize_weights(self)

        # UNet
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16,
            name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8,
            name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4,
            name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2,
            name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        x = dec1

        # c
        for idx, module in enumerate(self.down_convs, 1):
            x = module(x)
            if idx == 1:
                feature1 = x
            if idx == self.downsampling:
                feature2 = x
        for module in self.convs:
            x = module(x)
        feature3 = x
        x = torch.cat((self.avg_pooling(feature1), self.avg_pooling(feature2), self.avg_pooling(feature3)), 1)  # 0
        x = self.fc(x)

        return dec1, x

    @staticmethod
    def avg_pooling(feature):
        return F.avg_pool2d(feature, kernel_size=feature.size()[2:]).squeeze()
        # return torch.unsqueeze(F.avg_pool2d(feature, kernel_size=feature.size()[2:]).squeeze(),0)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=features,
                        kernel_size=3, padding=1, bias=False,),
                    nn.BatchNorm2d(num_features=features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=features, out_channels=features,
                        kernel_size=3, padding=1, bias=False,),
                    nn.BatchNorm2d(num_features=features),
                    nn.ReLU(inplace=True),)

class UNet_stage1(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UNet_stage1, self).__init__()

        # UNet
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16,
            name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8,
            name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4,
            name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2,
            name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=features,
                        kernel_size=3, padding=1, bias=False,),
                    nn.BatchNorm2d(num_features=features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=features, out_channels=features,
                        kernel_size=3, padding=1, bias=False,),
                    nn.BatchNorm2d(num_features=features),
                    nn.ReLU(inplace=True),)


if __name__ == '__main__':
    import torch

    img = torch.randn(2, 3, 128, 128).cuda()
    net = UNet().cuda()
    out1, out2 = net(img)
    print(out1.size())
    print(out2.size())

