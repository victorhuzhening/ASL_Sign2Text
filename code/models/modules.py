import torch
import torch.nn as nn
from torch import rand, randint
from torch import Tensor
from typing import Tuple


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
    def __init__(self,
                 left_hand_slice: slice = slice(0,63),
                 right_hand_slice: slice = slice(63, 126),
                 body_slice: slice = slice(126, 225),
                 time_probs: float = 0.2,
                 body_probs: float = 0.3,
                 feature_mask_p: Tuple = (0.33, 0.33, 0.34),
                 num_time_masks: int = 2,
                 mask_time_frac: float = 0.05,
                 mask_part_frac: float = 0.05,
                 mask_method: str = "zero"):
        super(AugMask, self).__init__()
        assert mask_method in ["zero", "mean"]
        self.left_hand_slice = left_hand_slice
        self.right_hand_slice = right_hand_slice
        self.body_slice = body_slice

        self.time_probs = time_probs
        self.body_probs = body_probs
        self.feature_mask_p = feature_mask_p

        self.num_time_masks = num_time_masks
        self.mask_time_frac = mask_time_frac
        self.mask_part_frac = mask_part_frac
        self.mask_method = mask_method

    def fill(self, inputs: Tensor, batch: int, t0: int, t1: int, part_slice: slice | None, mean_all: Tensor | None):
        """Helper for filling in only valid regions, i.e. non-padded Tensor elements"""
        if t1 < t0:
            return
        if self.mask_method == "zero":
            if part_slice is None:
                inputs[batch, t0:t1, :] = 0.0
            else:
                inputs[batch, t0:t1, part_slice] = 0.0
        else:
            if part_slice is None:
                inputs[batch, t0:t1, :] = mean_all[batch].unsqueeze(0)
            else:
                inputs[batch, t0:t1, part_slice] = mean_all[batch, part_slice].unsqueeze(0)


    @torch.no_grad()
    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        if (not self.training) or ((self.time_probs == 0.0) and (self.body_probs == 0.0)):
            return inputs

        batch_size, time_dim, feature_dim = inputs.shape
        out = inputs.clone()

        mean_all = None
        if self.mask_method == "mean":
            mean_all = torch.empty((batch_size, feature_dim), device=inputs.device, dtype=inputs.dtype)
            for b in range(batch_size):
                valid_len = int(input_lengths[b].item())
                if valid_len > 0:
                    mean_all[b] = out[b, :valid_len].mean(dim=0)
                else:
                    mean_all[b].zero_()

        for batch in range(batch_size):
            valid_len = int(input_lengths[batch].item())
            if valid_len < 1:
                continue

            if self.time_probs > 0.0 and float(rand(()).item()) < self.time_probs:
                max_mask_len = max(1, int(valid_len * self.mask_time_frac))

                partitions = randint(0, valid_len + 1, (self.num_time_masks - 1,))
                partitions, _ = torch.sort(partitions)
                boundaries = torch.cat([Tensor([0]), partitions, Tensor([valid_len])])

                for mask in range(self.num_time_masks):
                    seg0 = int(boundaries[mask].item())
                    seg1 = int(boundaries[mask+ 1].item())
                    seg_len = seg1 - seg0
                    if seg_len <= 0:
                        continue

                    mask_len = int(randint(1, min(max_mask_len, seg_len) + 1, ()).item())
                    start = int(randint(seg0, seg1 - mask_len, ()).item())
                    end = start + mask_len
                    self.fill(out, batch, start, end, part_slice = None, mean_all = mean_all)


            if self.body_probs > 0.0 and float(rand(()).item()) < self.body_probs:
                choose_part = float(rand(()).item())
                left, right, body = self.feature_mask_p
                if choose_part <= left:
                    part_slice = self.left_hand_slice
                elif choose_part <= left + right:
                    part_slice = self.right_hand_slice
                else:
                    part_slice = self.body_slice

                max_mask_len = max(1, int(valid_len * self.mask_part_frac))
                mask_len = int(randint(1, min(max_mask_len, valid_len) + 1, ()).item())
                start = int(randint(0, valid_len - mask_len + 1, ()).item())
                end = start + mask_len
                self.fill(out, batch, start, end, part_slice = part_slice, mean_all=mean_all)

        return out