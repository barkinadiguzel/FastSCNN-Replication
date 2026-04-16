import torch.nn as nn
from src.blocks.conv import DSConv

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            DSConv(128, 128),
            DSConv(128, 128),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)
