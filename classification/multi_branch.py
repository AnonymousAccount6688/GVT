import numpy as np
import time
import torch
from torch import nn
from torchvision.models.resnet import resnet50


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet50(pretrained=None)
        # self.net = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #    nn.BatchNorm2d(256)
        # )

    def forward(self, x):
        return self.net(x)


class MultiBranch2(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([BaseNet() for i in range(8)])

    def forward(self, xs):
        return self.example(xs)

    def example(self, xs):
        futures = []
        for i in range(8):
            futures.append(torch.jit.fork(self.branches[i], xs[i]))

        results = []
        for future in futures:
            results.append(torch.jit.wait(future))
        return torch.cat(results, 1)


class MultiBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([BaseNet() for i in range(8)])

    def forward(self, xs):
        return torch.cat([self.branches[i](xs[i]) for i in range(8)], 1)


if __name__ == "__main__":
    # x = torch.rand((2, 3, 224, 224))
    net = MultiBranch().cuda()

    # net = MultiBranch2().cuda()

    times = []
    for i in range(100):
        xs = [torch.randn(1, 3, 256, 256).cuda() for j in range(8)]
        # x2 = torch.randn(1, 4, 150).cuda()

        start = time.time()
        predict = net(xs)
        end = time.time()
        times.append(end - start)

    print(f"FPS: {1.0 / np.mean(times):.3f}")

    # FPS 66.365
    # 9.493