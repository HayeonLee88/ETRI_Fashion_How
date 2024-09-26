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

from typing import Any

import torch
from torch import Tensor

import torch.nn as nn
import torchvision.models as models

from torchvision.models.resnet import BasicBlock, Bottleneck

from torch.nn.quantized import FloatFunctional


class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)

        # 'add()' or '+' must be replaced with 'FloatFunctional.add()' due to quantization
        ###########################################
        self.skip_add_relu = FloatFunctional()
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
        out = self.skip_add_relu.add_relu(out, identity)
        ###########################################

        return out


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=False)

        # 'add()' or '+' must be replaced with 'FloatFunctional.add()' due to quantization
        ###########################################
        self.skip_add_relu = FloatFunctional()
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
        out = self.skip_add_relu.add_relu(out, identity)
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
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == "34":
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == "50":
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == "101":
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == "152":
            self.resnet = models.resnet152(pretrained=pretrained)

        self.resnet.relu = nn.ReLU(inplace=False)
        # 19개의 클래스에 맞게 fc레이어 output 수정
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 19)

    def front(self, x):
        """In the resnet structure, input 'x' passes through all layers."""
        return self.resnet(x)


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
    # """ Classification network of color category based on ResNet18 structure. """
    def __init__(self, res_num="18"):
        super().__init__()
        self.model = ResExtractor(res_num)

    def _foward_implement(self, x):
        """Forward propagation with input 'x'."""
        out = self.model.front(x)

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

        torch.ao.quantization.fuse_modules_qat(
            self.model.resnet, ["conv1", "bn1", "relu"], inplace=True
        )
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

                # Fuse conv1, bn1, relu together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv1", "bn1", "relu1"], inplace=True
                )
                # Fuse conv2 and bn2 together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv2", "bn2", "relu2"], inplace=True
                )
                # Fuse conv3, bn3 and relu together
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv3", "bn3"], inplace=True
                )
                if module.downsample:
                    # module.downsample[1] = nn.Identity()
                    torch.ao.quantization.fuse_modules_qat(
                        module.downsample, ["0", "1"], inplace=True
                    )


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


if __name__ == "__main__":
    pass
