import torch.nn as nn
from torch import Tensor


class Transpose(nn.Module):
    def __init__(self, shape):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class AugMask(nn.Module):
    def __init__(self):
        super(AugMask, self).__init__()