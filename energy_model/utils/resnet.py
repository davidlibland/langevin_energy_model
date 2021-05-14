import torch.nn as nn

from energy_model.utils.math import identity


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        norm_layer="batch_norm",
        activation=None,
        spectral_norm=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer == "batch_norm":
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.functional.relu
        if spectral_norm:
            conv_mod = nn.utils.spectral_norm
        else:
            conv_mod = lambda x: x
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = conv_mod(conv1x1(in_channels, out_channels, stride))
            if norm_layer is not None:
                downsample = nn.Sequential(downsample, norm_layer(out_channels),)
        if norm_layer is None:
            norm_layer = lambda _: identity
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_mod(conv3x3(in_channels, out_channels, stride))
        self.bn1 = norm_layer(out_channels)
        self.activation = activation
        self.conv2 = conv_mod(conv3x3(out_channels, out_channels))
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out
