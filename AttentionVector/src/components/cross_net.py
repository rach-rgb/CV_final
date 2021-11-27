import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class CrossNet(nn.Module):
    def __init__(self, num_classes, ROI_batch, d_k=64, r_k=10):
        super(CrossNet, self).__init__()
        self.prior_rel = None
        self.batch = ROI_batch
        self.r_k = r_k

        # single-head self attention
        self.query_fc = nn.Linear(num_classes+1, d_k)
        self.key_fc = nn.Linear(num_classes+1, d_k)

        # excitation FC
        self.fc = nn.Linear(num_classes+1, num_classes+1)

    def set_rel(self, rel):
        self.prior_rel = rel

    def forward(self, x):
        x = x.view(-1, self.batch, x.size(1))

        q = self.query_fc(x)
        k = self.key_fc(x)

        # self-attention
        out = torch.matmul(q, k.transpose(1, 2))
        out = out / math.sqrt(k.size(2))
        out = F.softmax(out, dim=2)

        # add relevance
        label = torch.argmax(x, dim=2)
        r = torch.zeros_like(x, dtype=torch.float32, device='cuda:0')
        for b in range(0, x.size(0)):
            for i in range(0, x.size(1)):
                li = label[b][i]
                if li == x.size(2):  # skip
                    continue
                for j in torch.topk(out[b][i], self.r_k)[1]:
                    lj = label[b][j]
                    if (lj == x.size(2)) or (li == lj):  # skip
                        continue
                    r[b][i][lj] = r[b][i][lj] + self.prior_rel[lj][li] * out[b][i][j] * x[b][j][lj]

        out = F.relu(r)

        # apply weight & activate
        out = torch.sigmoid(self.fc(out))

        out = out.view(-1, out.size(2))

        return out
