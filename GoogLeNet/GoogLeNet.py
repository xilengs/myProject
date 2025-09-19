import torch
import torch.nn as nn
from torchinfo import summary

# 在每一层卷积后加一层BN，提升网络训练稳定性和收敛速度
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3reduce, kernel_size=1),
            BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5reduce, kernel_size=1),
            BasicConv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# auxiliary classifiers
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        self.feature = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            BasicConv2d(in_channels, 128, kernel_size=1),
        )
        self.clf = nn.Sequential(
            nn.Linear(4*4*128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 4*4*128)
        x = self.clf(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # ceil_mode=True, 向上取证，防止丢弃边界的像素
        # 这里padding=1也可以达到同样效果
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 表格里将conv2a(1x1) 和 conv2b(3x3) 合并成了一个Conv2模块, 但是在网络图中可以看到有两个conv层
        self.conv2 = BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxPool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 自适应池化, 指定输出尺寸, 不需要自己计算kernel_size和stride
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxPool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxPool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxPool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux1, aux2





