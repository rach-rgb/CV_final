import torch
from torch import nn


class CrossNet(nn.Module):
    def __init__(self, num_classes):
        super(CrossNet, self).__init__()
        # self.rel = torch.ones(4, 4)  # get relevance matrix (prior knowledge or train?)
        #
        # self.x_complement = nn.Sequential(  # get N*(C*C)
        #
        # )
        # self.layer1 =
        # self.layer2 =


    def forward(self, x):
        # x_c = self.x_complement(x)
        # x = x * x_c
        # x = self.layer1(x, self.rel)
        # x = self.layer2(x)
        # return F.relu(x)

        return torch.ones_like(x)
