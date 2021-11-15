from torch import nn


class CrossNet(nn.Module):
    def __init__(self):
        super(CrossNet, self).__init__()

    def forward(self, x):
        # ret = torch.zeros_like(x)
        # return ret * x
        return x
