import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import gzip
import sys
import torch.optim as optim
conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=conv2d1_filters_numbers,
            kernel_size=conv2d1_filters_size, stride=1,
            padding=int((conv2d1_filters_size - 1) / 2))
        self.conv1_1 = nn.Conv2d(
            in_channels=conv2d1_filters_numbers,
            out_channels=conv2d1_filters_numbers,
            kernel_size=3, stride=1,
            padding=int((3 - 1) / 2))
        self.conv2 = nn.Conv2d(
            in_channels=conv2d1_filters_numbers,
            out_channels=conv2d2_filters_numbers,
            kernel_size=conv2d2_filters_size, stride=1,
            padding=int((conv2d2_filters_size - 1) / 2))
        self.conv3 = nn.Conv2d(
            in_channels=conv2d2_filters_numbers, out_channels=1,
            kernel_size=conv2d3_filters_size, stride=1,
            padding=int((conv2d3_filters_size - 1) / 2))

    def forward(self, x):
        #print("start forwardingf")
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x