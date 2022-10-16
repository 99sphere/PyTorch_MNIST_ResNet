import torch.nn as nn

"""
This file includes implementation of ConvBlock, ResBlock, BottleneckBlock
ConvBlock is for ResBlock and BottleneckBlock
ResBlock is for ResNet18 and ResNet34
BottleneckBlock is for ResNet50, ResNet101 and ResNet152
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)  
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_block2 = ConvBlock(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        res_x = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        if res_x.shape[-3] == x.shape[-3]:
            x = x + res_x
        else:
            res_x = self.shortcut(res_x)
            x = x + res_x
            x = self.bn(x)
        x = self.relu(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_block2 = ConvBlock(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_block3 = ConvBlock(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res_x = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if res_x.shape[-3] == x.shape[-3]:
            x = x + res_x
        else:
            res_x = self.shortcut(res_x)

            x = x + res_x
            x = self.bn(x)
        x = self.relu(x)
        return x