from __future__ import annotations

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.add = FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.add.add_relu(out, x if self.down is None else self.down(x))


class ResNetCifar(nn.Module):
    def __init__(self, width: int = 64, blocks: tuple[int, ...] = (2, 2, 2, 2), num_classes: int = 10):
        super().__init__()
        self.quant, self.dequant = QuantStub(), DeQuantStub()
        self.stem_conv = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(width)
        self.stem_relu = nn.ReLU()
        layers, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * 2**i
            for j in range(n):
                layers.append(Block(cin, cout, 2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.pool(self.layers(x)).flatten(1)
        return self.dequant(self.fc(x))

    def fuse(self) -> "ResNetCifar":
        """Merge conv+batchnorm(+relu) into single layers. Needed before int8 quantization. Call on an eval-mode copy."""
        assert not self.training, "call .eval() first"
        groups = [["stem_conv", "stem_bn", "stem_relu"]]
        for i, block in enumerate(self.layers):
            groups += [[f"layers.{i}.conv1", f"layers.{i}.bn1", f"layers.{i}.relu1"], [f"layers.{i}.conv2", f"layers.{i}.bn2"]]
            if block.down is not None:
                groups.append([f"layers.{i}.down.0", f"layers.{i}.down.1"])
        return fuse_modules(self, groups, inplace=True)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
