import math, torch
import torch.nn as nn
import torch.nn.functional as F


# RoI-wise attention vector generating network
class RAVNet(nn.Module):
    def __init__(self, num_classes, d_k=64):
        super(RAVNet, self).__init__()

        # list of number of instances(RoI predictions) in each image
        # len(num_instances): RoI-head batch size
        # num_instances[x]: number of RoIs in image x
        self.num_instances = None

        # class relevance information
        self.prior_rel = nn.Parameter(torch.zeros(81, 81), requires_grad=False)

        # single-head self attention
        self.query_fc = nn.Linear(num_classes+1, d_k)
        self.key_fc = nn.Linear(num_classes+1, d_k)
        self.value_fc = nn.Linear(num_classes+1, d_k)

        # unpack FC
        self.unpack_fc = nn.Linear(d_k, num_classes+1)

    # save class relevance information
    def set_rel(self, rel):
        self.prior_rel.data = rel

    # save number of instances
    def pass_hint(self, num_instances):
        self.num_instances = num_instances

    def forward(self, x):
        # parse predictions by RoI source and add 0 paddings
        out = torch.split(x, self.num_instances)
        out = [F.pad(y, pad=[0, 0, 0, max(self.num_instances)-y.size(0)], mode='constant', value=0) for y in out]
        out = torch.stack(out)

        # calculate query, key, and relevance based value
        q = self.query_fc(out)
        k = self.key_fc(out)
        v = self.value_fc(torch.matmul(out, self.prior_rel))

        # calculate single head attention
        out = torch.matmul(q, k.transpose(1, 2))
        out = out / math.sqrt(k.size(2))
        for y in range(0, len(self.num_instances)):
            out[y][:, self.num_instances[y]:] = -math.inf

        out = F.softmax(out, dim=2)

        # multiply attention and value
        out = torch.matmul(out, v)

        # recover dimension
        out = self.unpack_fc(out)

        # flatten each attnetion scores
        out = out.view(-1, out.size(2))
        d = max(self.num_instances)
        out = torch.cat([out[d*y: d*y+self.num_instances[y]] for y in range(0, len(self.num_instances))])

        return out
