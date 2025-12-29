import torch.nn as nn
from torch import Tensor
from modules import Linear
from activation import Swish


class FeedForwardBlock(nn.Module):
    def __init__(self,
                 num_features: int = 512,
                 expansion_factor: int = 4,
                 bias: bool = True,
                 drop_out_p: int = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(num_features),
            Linear(in_features=num_features,
                   out_features=num_features * expansion_factor,
                   bias=bias),
            Swish(),
            nn.Dropout(p=drop_out_p),
            Linear(in_features=num_features * expansion_factor,
                   out_features=num_features,
                   bias=bias),
            nn.Dropout(p=drop_out_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)