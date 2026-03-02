import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class DiffConvBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=2):
        super(DiffConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        attn = self.sigmoid(self.conv1(pool))
        return x * attn


class BottleneckBlock(nn.Module):

    def __init__(self, in_c, out_c, se_rate, drop_rate, do_attn=True):
        super(BottleneckBlock, self).__init__()

        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=drop_rate, bias=False),
                nn.BatchNorm2d(out_c)
            )

        self.conv = nn.Sequential(
            ConvBlock(in_c, out_c, kernel_size=1, stride=drop_rate, padding=0),
            ConvBlock(out_c, out_c),
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_c, out_c // se_rate, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c // se_rate, out_c, kernel_size=1, bias=False)
        )

        self.do_attn = do_attn
        if self.do_attn:
            self.attention = SpatialAttention()

    def forward(self, x):
        res = x
        conv_res = self.conv(x)
        scale = self.se(conv_res)
        attn = conv_res * scale
        if self.do_attn:
            attn = attn + conv_res
            attn = self.attention(attn) + attn

        if self.downsample is not None:
            res = self.downsample(res)

        attn += res
        x = F.relu(attn)
        return x


class SEResNet(nn.Module):

    def __init__(self, in_c, out_c, blocks, se_rate=8, drop_rate=1, do_attn=False):
        super(SEResNet, self).__init__()
        self.conv_head = BottleneckBlock(in_c, out_c, se_rate, drop_rate, do_attn=do_attn)
        layers = []
        for _ in range(blocks - 1):
            layers.append(BottleneckBlock(out_c, out_c, se_rate, drop_rate, do_attn=do_attn))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_head(x)
        out = self.layers(x)
        return out

class SEResNetDecoder(nn.Module):

    def __init__(self, in_c, out_c, blocks, se_rate=8, drop_rate=2):
        super(SEResNetDecoder, self).__init__()

        self.conv_head = BottleneckBlock(in_c, out_c, se_rate, 1, False)
        layers = []
        for _ in range(blocks - 1):
            layers.append(BottleneckBlock(out_c, out_c, se_rate, 1, False))
            layers.append(BottleneckBlock(out_c, out_c * drop_rate, se_rate, drop_rate, False))
            out_c *= drop_rate
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_head(x)
        out = self.layers(x)
        return out

class DiffusionNet(nn.Module):

    def __init__(self, in_c, out_c, blocks):
        super(DiffusionNet, self).__init__()

        self.conv_head = DiffConvBlock(in_c, out_c)
        layers = []
        for _ in range(blocks - 1):
            layers.append(DiffConvBlock(out_c, out_c))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_head(x)
        out = self.layers(x)
        return out
