import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
from model.submodules import ConvBlock, BottleneckBlock


class ResNet152(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, nblk=[3, 8, 36, 3]):
        super(ResNet152, self).__init__()

        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels = 64
        bottleneck_blocks = []
        for i, n in enumerate(nblk):
            out_channels = 64 * (2 ** (i + 2))

            for j in range(n):
                if i == 0 and j == 0:
                    mid_channels = in_channels
                    bottleneck_blocks.append(
                        BottleneckBlock(
                            in_channels=in_channels,
                            mid_channels=mid_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                    in_channels = out_channels

                elif i != 0 and j == 0:
                    mid_channels *= 2
                    bottleneck_blocks.append(
                        BottleneckBlock(
                            in_channels=in_channels,
                            mid_channels=mid_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        )
                    )
                    in_channels = out_channels

                else:
                    out_channels = in_channels
                    bottleneck_blocks.append(
                        BottleneckBlock(
                            in_channels=in_channels,
                            mid_channels=mid_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )

        self.BottleneckBlocks = nn.Sequential(*bottleneck_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(nker * 2 * 2 * 2 * 2 * 2, 10)

    def __str__(self):
        return "ResNet152"

    def forward(self, x):
        x = self.enc(x)
        x = self.max_pool(x)
        x = self.BottleneckBlocks(x)
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

    my_resnet = ResNet152(3, 10)
    resnet152 = models.resnet152(pretrained=False)
    resnet152.fc = nn.Linear(2048, 10)

    device = torch.device("cuda")
    my_resnet.to(device)
    resnet152.to(device)

    summary(my_resnet, (3, 28, 28))
    summary(resnet152, (3, 28, 28))
