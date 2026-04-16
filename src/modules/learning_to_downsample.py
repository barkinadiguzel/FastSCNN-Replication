from src.blocks.conv import ConvBNReLU, DSConv
import torch.nn as nn

class LearningToDownsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2),
            DSConv(32, 48, 2),
            DSConv(48, 64, 2)
        )

    def forward(self, x):
        return self.net(x)
