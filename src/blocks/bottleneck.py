import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expansion):
        super().__init__()

        hidden = in_ch * expansion
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, stride, 1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        return self.block(x)
