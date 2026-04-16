import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.scales = [1, 2, 3, 6]

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, out_ch // 4, 1, bias=False),
                nn.BatchNorm2d(out_ch // 4),
                nn.ReLU(inplace=True)
            )
            for s in self.scales
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        outs = [x]

        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, (h, w), mode='bilinear', align_corners=False)
            outs.append(y)

        return self.fuse(torch.cat(outs, dim=1))
