import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionModule(nn.Module):
    def __init__(self, high_ch, low_ch, out_ch):
        super().__init__()

        self.high = nn.Conv2d(high_ch, out_ch, 1, bias=False)

        self.low_dw = nn.Conv2d(
            low_ch, low_ch, 3,
            padding=1, groups=low_ch,
            bias=False
        )
        self.low_pw = nn.Conv2d(low_ch, out_ch, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high, low):
        size = high.shape[2:]

        low = F.interpolate(low, size, mode='bilinear', align_corners=False)

        low = self.low_dw(low)
        low = self.low_pw(low)

        high = self.high(high)

        x = high + low
        return self.relu(self.bn(x))
