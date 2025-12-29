import torch
import torch.nn as nn
from torch import Tensor
from activation import Swish, GLU
from modules import Transpose
from typing import Tuple

class PointwiseConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias = True,):
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DepthwiseConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 bias = False,):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels, # when groups=in_channels, each channels is convolved with its own set of filters
            stride=stride,
            padding=padding,
            bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ConformerConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 drop_out_p: float = 0.1,
                 gated_factor: int = 2,):
        super(ConformerConvBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1,2)),
            PointwiseConv1d(in_channels=in_channels,
                            out_channels=in_channels * gated_factor),
            GLU(dim=1),
            DepthwiseConv1d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(num_features=in_channels),
            Swish(),
            PointwiseConv1d(in_channels=in_channels,
                            out_channels=in_channels),
            nn.Dropout(p=drop_out_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x).transpose(1,2)




class ConvSubsampling(nn.Module):
    """"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
    ):
        super(ConvSubsampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths