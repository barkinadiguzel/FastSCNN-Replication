import torch.nn as nn
from src.blocks.bottleneck import Bottleneck
from src.blocks.ppm import PPM

class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            Bottleneck(64, 64, 2, 6),
            Bottleneck(64, 64, 1, 6),
            Bottleneck(64, 64, 1, 6),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(64, 96, 2, 6),
            Bottleneck(96, 96, 1, 6),
            Bottleneck(96, 96, 1, 6),
        )

        self.stage3 = nn.Sequential(
            Bottleneck(96, 128, 1, 6),
            Bottleneck(128, 128, 1, 6),
            Bottleneck(128, 128, 1, 6),
        )

        self.ppm = PPM(128, 128)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.ppm(x)
        return x
