import torch
import torch.nn as nn


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
        if norm_layer is None:
            norm_layer = Identity
        if activation is None:
            activation = nn.ReLU
        if spectral_norm:
            conv_mod = nn.utils.spectral_norm
        else:
            conv_mod = lambda x: x
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv_mod(conv1x1(in_channels, out_channels, stride)),
                norm_layer(out_channels),
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_mod(conv3x3(in_channels, out_channels, stride))
        self.bn1 = norm_layer(out_channels)
        self.activation = activation()
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


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
