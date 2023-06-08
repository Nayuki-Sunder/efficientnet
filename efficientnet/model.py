import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class SEBlock(nn.Module):
    def __init__(self, in_ch, se_ch):
        super().__init__()

        self.fc1 = nn.Conv2d(in_ch, se_ch, kernel_size=1)
        self.fc2 = nn.Conv2d(se_ch, in_ch, kernel_size=1)

    def forward(self, x):
        w = F.avg_pool2d(x, x.shape[2])
        w = F.silu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        out = x * w
        return out

def drop_path(x, drop_prob=0., training=False, scale_by_keep=True):

    shape = (x.shape[0]) + (1) * (x.ndim - 1)
    keep_prob = 1 - drop_prob
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if drop_prob == 0. or not training:
        return x

    if keep_prob > 0. and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0., scale_by_keep=True):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, se_ratio, death_rate=0.):
        super().__init__()

        self.stride = stride
        self.death_rate = death_rate
        self.expand_ratio = expand_ratio

        mid_ch = expand_ratio * in_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, padding=(1 if kernel_size == 3 else 2), groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        se_ch = int(in_ch * se_ratio)
        self.se = SEBlock(mid_ch, se_ch)

        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.droppath = DropPath(drop_prob=death_rate, scale_by_keep=True)

        self.has_skip = (stride == 1) and (in_ch == out_ch)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else F.silu(self.bn1(self.conv1(x)))
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.death_rate > 0:
                out = self.droppath(out)
            out = out + x

        return out

class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()

        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(in_ch=32)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_ch):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size', 'stride']]

        b = 0
        blocks = sum(self.cfg['num_blocks'])
        
        for expansion, out_ch, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                death_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_ch, out_ch, kernel_size, stride, expansion, se_ratio=0.5, death_rate=death_rate)
                )
                in_ch = out_ch
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, out.shape[2])
        out = rearrange(out, 'b c h w -> b (c h w)')
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0.:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out

def EfficientNetB0():
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg)