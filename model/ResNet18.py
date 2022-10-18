import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
from model.submodules import ConvBlock, ResBlock


class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, nblk=[2, 2, 2, 2]):
        super(ResNet18, self).__init__()

        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels = 64
        res_blocks = []
        for i, n in enumerate(nblk):
            out_channels = 64 * (2**i)
            for j in range(n):
                if i != 0 and j == 0:
                    res_blocks.append(
                        ResBlock(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        )
                    )
                else:
                    res_blocks.append(
                        ResBlock(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                in_channels = out_channels
        self.ResBlocks = nn.Sequential(*res_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(nker * 2 * 2 * 2, 10)

    def __str__(self):
        return "ResNet18"

    def forward(self, x):
        x = self.enc(x)
        x = self.max_pool(x)
        x = self.ResBlocks(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


if __name__ == "__main__":

    #####################################################################
    # Check model architecture and total number of parameters.          #
    # For MNIST dataset, just rescale the last fc layer's weight shape. #
    # The number of params and arthitecture is exactly same.            #
    #####################################################################

    my_resnet18 = ResNet18(3, 10)
    resnet18 = models.resnet18(pretrained=False)
    resnet18.fc = nn.Linear(512, 10)

    device = torch.device("cuda")
    my_resnet18.to(device)
    resnet18.to(device)

    summary(my_resnet18, (3, 28, 28))
    summary(resnet18, (3, 28, 28))
