import numpy as np
import torch.nn as nn
import torch

# 把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding_size=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding_size),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            # nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            # nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, multiple=2):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 32*multiple)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32*multiple, 64*multiple, (1, 5), (0, 2))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64*multiple, 128*multiple, (5, 1), (2, 0))
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up8 = nn.ConvTranspose2d(128*multiple, 64*multiple, 2, stride=2)
        self.conv8 = DoubleConv(128*multiple, 64*multiple, (1, 5), (0, 2))
        self.up9 = nn.ConvTranspose2d(64*multiple, 32*multiple, 2, stride=2)
        self.conv9 = DoubleConv(64*multiple, 32*multiple, (5, 1), (2, 0))
        self.conv10 = nn.Conv2d(32*multiple, out_ch, 1)

        self.epi_conv1 = DoubleConv(in_ch, 8*multiple, 1, 0)
        self.epi_conv2 = DoubleConv(8*multiple, 16*multiple, 1, 0)
        self.epi_conv3 = DoubleConv(16*multiple, 32*multiple, 1, 0)

    def forward(self, hic, epi):
        c1 = self.conv1(hic)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)

        up_8 = self.up8(c3)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        _array = [c9]
        e1 = self.epi_conv1(epi)
        e2 = self.epi_conv2(e1)
        e3 = self.epi_conv3(e2)
        _array.append(e3)
        c9 = torch.cat(_array, dim=1)
        c9 = self.conv9(c9)

        c10 = self.conv10(c9)
        
        out = nn.ReLU()(c10)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
