import math, torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNet(nn.Module):
    def __init__(self, num_classes, d_k=64):
        super(CrossNet, self).__init__()
        self.num_instances = None
        self._track = False

        self.prior_rel = nn.Parameter(torch.zeros(81, 81), requires_grad=False)

        # single-head self attention
        self.query_fc = nn.Linear(num_classes+1, d_k)
        self.key_fc = nn.Linear(num_classes+1, d_k)
        self.value_fc = nn.Linear(num_classes+1, d_k)

        # unpack FC
        self.unpack_fc = nn.Linear(d_k, num_classes+1)

    def set_rel(self, rel):
        self.prior_rel.data = rel

    def pass_hint(self, num_instances):
        self.num_instances = num_instances

    def track(self, _track):
        self._track = _track

    def forward(self, x):
        out = torch.split(x, self.num_instances)
        out = [F.pad(y, pad=[0, 0, 0, max(self.num_instances)-y.size(0)], mode='constant', value=0) for y in out]
        out = torch.stack(out)

        q = self.query_fc(out)
        k = self.key_fc(out)
        v = self.value_fc(torch.matmul(out, self.prior_rel))

        # self-attention
        out = torch.matmul(q, k.transpose(1, 2))
        out = out / math.sqrt(k.size(2))
        for y in range(0, len(self.num_instances)):
            out[y][:, self.num_instances[y]:] = -math.inf

        out = F.softmax(out, dim=2)

        if self._track:
            print('Input')
            print(x)
            print('Attention')
            print(out)
            print('Conditional Probability')
            print(v)

        # relevance based value
        out = torch.matmul(out, v)

        # unpack
        out = torch.sigmoid(self.unpack_fc(out))

        out = out.view(-1, out.size(2))
        d = max(self.num_instances)
        out = torch.cat([out[d*y: d*y+self.num_instances[y]] for y in range(0, len(self.num_instances))])

        if self._track:
            print('result')
            print(out)

        return out
