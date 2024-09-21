"""
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2024.04.20.
"""

from typing import Callable, Optional

import torch
from torch import Tensor

import torch.nn as nn
from torch.nn.quantized import FloatFunctional

import torchvision.models as models


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
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


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class QuantizableBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # 'add()' or '+' must be replaced with 'FloatFunctional.add()' due to quantization
        ###########################################
        self.float_functional = FloatFunctional()
        ###########################################

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        ###########################################
        self.float_functional.add_relu(out, identity)
        ###########################################

        return out


class QuantizableBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        # 'add()' or '+' must be replaced with 'FloatFunctional.add()' due to quantization
        ###########################################
        self.float_functional = FloatFunctional()
        ###########################################

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        ###########################################
        out = self.float_functional.add_relu(out, identity)
        ###########################################

        return out


models.resnet.BasicBlock = QuantizableBasicBlock
models.resnet.Bottleneck = QuantizableBottleneck


class ResExtractor(nn.Module):
    """Feature extractor based on ResNet structure
            Selectable from resnet18 to resnet152

    Args:
        resnetnum: Desired resnet version
                    (choices=['18','34','50','101','152'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, resnetnum="50", pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == "18":
            resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == "34":
            resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == "50":
            resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == "101":
            resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == "152":
            resnet = models.resnet152(pretrained=pretrained)

        self.model_front = nn.Sequential(*list(resnet.children())[:-2])

    def front(self, x):
        """In the resnet structure, input 'x' passes through conv layers except for fc layers."""
        return self.model_front(x)


class MnExtractor(nn.Module):
    """Feature extractor based on MobileNetv2 structure
    Args:
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, pretrained=True):
        super(MnExtractor, self).__init__()

        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.modules_front = list(self.net.children())[:-1]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """In the resnet structure, input 'x' passes through conv layers except for fc layers."""
        return self.model_front(x)


class Baseline_ResNet_color(nn.Module):
    """Classification network of color category based on ResNet18 structure."""

    def __init__(self, res_num="18"):
        super().__init__()

        self.encoder = ResExtractor(res_num)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        if res_num == "18":
            self.color_linear = nn.Linear(512, 19)
        elif res_num == "50":
            self.color_linear = nn.Linear(2048, 19)

    def _forward_base(self, x):
        """Forward propagation with input 'x'."""
        feat = self.encoder.front(x)
        flatten = self.avg_pool(feat).squeeze()

        out = self.color_linear(flatten)

        return out


class Baseline_MNet_color(nn.Module):
    """Classification network of emotion categories based on MobileNetv2 structure."""

    def __init__(self):
        super(Baseline_MNet_color, self).__init__()

        self.encoder = MnExtractor()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.color_linear = nn.Linear(1280, 19)

    def forward(self, x):
        """Forward propagation with input 'x'"""
        feat = self.encoder.front(x["image"])
        flatten = self.avg_pool(feat).squeeze()

        out = self.color_linear(flatten)

        return out


class Quantizable_ResNet_color(Baseline_ResNet_color):
    def __init__(self, res_num="18"):
        super(Quantizable_ResNet_color, self).__init__(res_num=res_num)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = self._foward_implement(x)
        out = self.dequant(out)
        return out

    def fuse_model(self) -> None:
        for module_name, module in self.named_modules():
            # torch.quantization.fuse_modules 사용시 convert to Quantization model의 weight가 갱신이 안됨.
            if isinstance(module, models.resnet.BasicBlock):
                # Fuse conv1, bn1, relu together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv1", "bn1", "relu"], inplace=True
                )
                # Fuse conv2 and bn2 together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv2", "bn2"], inplace=True
                )
                if module.downsample:
                    torch.ao.quantization.fuse_modules_qat(
                        module.downsample, ["0", "1"], inplace=True
                    )

            elif isinstance(module, models.resnet.Bottleneck):
                # Fuse conv1, bn1, relu1 together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv1", "bn1", "relu1"], inplace=True
                )
                # Fuse conv2 and bn2 and relu2 together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv2", "bn2", "relu2"], inplace=True
                )
                # Fuse conv3, bn3 together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv3", "bn3"], inplace=True
                )
                if module.downsample:
                    torch.ao.quantization.fuse_modules_qat(
                        module.downsample, ["0", "1"], inplace=True
                    )


if __name__ == "__main__":
    pass
