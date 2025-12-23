import torch.nn as nn
from torch import Tensor


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * nn.functional.sigmoid(x)


class GLU(nn.Module):
    """
    Takes in a linear layer with 2x dim size. Acts as a pure activation function.
    """
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()