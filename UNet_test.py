import torch.nn as nn
import torch
from torch import autograd

#把常用的2个卷积操作简单封装下
# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch), #添加了BN层
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, input):
#         return self.conv(input)

#把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding_size=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding_size),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64, 1, 0)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

        self.test_conv = nn.Conv2d(64, 64, 1)
        self.large_conv = DoubleConv(in_ch, 32, 1, 0)
        self.epi_conv = DoubleConv(in_ch, 16)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, hic, epi):
        c1 = self.conv1(hic)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        # p2 = self.pool2(c2)
        # c3 = self.conv3(p2)

        c2 = nn.Sigmoid()(c2)

        c8 = self.dropout(c2)

        # up_8 = self.up8(c3)
        # merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        # c9 = self.test_conv(c9)
        # _row = self.row_conv(hic)
        # _row = torch.cat([_row] * 400, dim=2)
        # _col = self.col_conv(hic)
        # _col = torch.cat([_col] * 400, dim=3)
        # c9 = torch.cat([c9, _row, _col], dim=1)
        # c9 = self.conv9(c9)

        c9 = self.test_conv(c9)
        _array = [c9]
        for i in range(2):
            _array.append(self.epi_conv(epi))
        _array.append(self.large_conv(hic))
        c9 = torch.cat(_array, dim=1)
        c9 = self.conv9(c9)

        c10 = self.conv10(c9)
        
        out = nn.ReLU()(c10)
        # out = nn.Sigmoid()(c10)
        return out