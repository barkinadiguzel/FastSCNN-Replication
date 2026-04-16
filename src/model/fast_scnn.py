import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CityscapesConfig
from src.modules.learning_to_downsample import LearningToDownsample
from src.modules.global_extractor import GlobalFeatureExtractor
from src.modules.classifier import Classifier
from src.blocks.fusion import FeatureFusionModule


class FastSCNN(nn.Module):
    def __init__(self, config=CityscapesConfig):
        super().__init__()

        self.config = config

        self.down = LearningToDownsample()
        self.global_net = GlobalFeatureExtractor()

        self.fusion = FeatureFusionModule(
            high_ch=64,
            low_ch=128,
            out_ch=128
        )

        self.classifier = Classifier(config.NUM_CLASSES)

    def forward(self, x):
        input_size = x.shape[2:]

        high = self.down(x)
        low = self.global_net(high)

        x = self.fusion(high, low)
        x = self.classifier(x)

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x
