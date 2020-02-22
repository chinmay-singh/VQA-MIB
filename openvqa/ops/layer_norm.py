# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x, to_print=False):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        if to_print:
            print("Layer norm params, a = {}, b = {}".format(self.a_2, self.b_2))

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
