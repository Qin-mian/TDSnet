import torch
import torch.nn as nn
import torch.nn.functional as F

"""
author: Qmian
date: 2024-10-07
description: TDSNet
"""

def cov7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def cov3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class DDModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDModule, self).__init__()

        # 第一个分支
        self.branch1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.branch1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

        # 第二个分支
        self.branch2_1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.branch2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        # 第三个分支
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        # 第四个分支
        self.branch4_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.branch4_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1)

        # 第五个分支
        self.branch5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.branch5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)

        # 第六个分支
        self.branch6_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.branch6_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, dilation=1, padding=2)

        # 第七个分支
        self.branch7_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.branch7_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, dilation=2, padding=4)

        # 批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.bn6 = nn.BatchNorm2d(out_channels)
        self.bn7 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = self.branch1_2(F.relu(self.bn1(self.branch1_1(x))))
        out2 = self.branch2_2(F.relu(self.bn2(self.branch2_1(x))))
        out3 = F.relu(self.bn3(self.branch3(x)))
        out4 = self.branch4_2(F.relu(self.bn4(self.branch4_1(x))))
        out5 = self.branch5_2(F.relu(self.bn5(self.branch5_1(x))))
        out6 = self.branch6_2(F.relu(self.bn6(self.branch6_1(x))))
        out7 = self.branch7_2(F.relu(self.bn7(self.branch7_1(x))))
        out = torch.cat([out1, out2, out3, out4, out5, out6, out7], 1)

        return out


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   dilation=dilation, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBnet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBnet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TDSnet(nn.Module):
    def __init__(self,num_classes=6,base_kernel = 64):
        super(TDSnet, self).__init__()
        self.ReLu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv0 = nn.Conv2d(1, 32, kernel_size=7, stride = 1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.Maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # DDmodule支路
        self.block1 = DDModule(32, base_kernel) #实际上out_channels是其7倍

        self.block2 = DDModule(base_kernel*8, base_kernel*2)
        self.block3 = DDModule(base_kernel*14, base_kernel*4)

        self.block4 = DDModule(base_kernel*32, base_kernel*8)

        self.block5 = DDModule(base_kernel*64, base_kernel*8)

        # 可行变卷积
        self.block6 = SeparableConv(base_kernel*64, 512, kernel_size=3, padding=1)

        # SEB注意力
        self.block7 = SEBnet(512, 16)

        # 全局平均池化
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

        # 卷积支路
        self.conv1 = nn.Conv2d(32, base_kernel,kernel_size=7, stride=1, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(base_kernel)

        self.conv2 = nn.Conv2d(base_kernel, base_kernel*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_kernel*2)
        self.conv3 = nn.Conv2d(base_kernel*2, base_kernel*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(base_kernel*4)

        self.conv4 = nn.Conv2d(base_kernel*4, base_kernel*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(base_kernel*8)
        self.conv5 = nn.Conv2d(base_kernel*8, base_kernel*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(base_kernel*8)

        self.conv6 = nn.Conv2d(base_kernel*8, base_kernel*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(base_kernel*8)
        self.conv7 = nn.Conv2d(base_kernel*8, base_kernel*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(base_kernel*8)

    def forward(self,x):
        x = self.ReLu(self.bn1(self.conv0(x)))
        x = self.Maxpooling(x)

        # 1
        x1 = self.block1(x)
        x2 = self.ReLu(self.bn3(self.conv1(x)))
        x = torch.cat((x1, x2), 1)

        # 2
        x3 = self.block3(self.Maxpooling(self.block2(x)))
        x4 = self.Maxpooling(self.ReLu(self.bn5(self.conv3(self.ReLu(self.bn4(self.conv2(x2)))))))
        x = torch.cat((x3, x4), 1)

        # 3
        x6 = self.block4(self.Maxpooling(x))
        x7 = self.Maxpooling(self.ReLu(self.bn7(self.conv5(self.ReLu(self.bn6(self.conv4(x4)))))))
        x = torch.cat((x6, x7), 1)

        # 4
        x8 = self.block5(self.Maxpooling(x))
        x9 = self.Maxpooling(self.ReLu(self.bn9(self.conv7(self.ReLu(self.bn8(self.conv6(x7)))))))
        x = torch.cat((x8, x9), 1)

        x = self.block6(x)
        x = self.block7(x)
        x = self.bn2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


def TDSnet_small(num_classes = 6):
    return TDSnet(num_classes, 32)

def TDSnet_normal(num_classes = 6):
    return TDSnet(num_classes, 64)
