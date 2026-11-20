"""U-TAE: a U-Net with temporal attention, for an irregular series of images (milestone B7).

Written here after Garnot and Landrieu, "Panoptic segmentation of satellite image time series with
convolutional temporal attention networks", ICCV 2021 (U-TAE), and Garnot and Landrieu, "Lightweight
temporal self-attention for classifying satellite image time series", 2020 (L-TAE).

How it works, in four steps:
1. The encoder (convolutions, shrinking the image three times by 2) runs on every date separately, with
   the same weights. A date is 8 channels: 7 bands (missing values 0) and a "clear" flag.
2. At the coarsest scale (1/8 of the size), each pixel's sequence of feature vectors goes through L-TAE:
   every date gets a time code (a sinusoid of its day, plus a learned vector for "before" or "after"),
   and 16 attention heads each compute one weight per date (weights sum to 1 over the dates; padded
   dates get weight 0). Each head then averages its own group of feature channels over the dates.
   This is a weighted average over time whose weights depend on what each date shows and when it was taken.
3. The same attention weights, enlarged to each finer scale, average the encoder's features there too.
4. A U-Net decoder grows the averaged features back to full size; the last layer gives 2 numbers per
   pixel (not burnt, burnt).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="reflect"), nn.GroupNorm(4, c_out), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode="reflect"), nn.GroupNorm(4, c_out), nn.ReLU())


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(c_in, c_in, 4, stride=2, padding=1), nn.GroupNorm(4, c_in), nn.ReLU())
        self.conv = conv_block(c_in, c_out)

    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.GroupNorm(4, c_out), nn.ReLU())
        self.skip = nn.Sequential(nn.Conv2d(c_skip, c_skip, 1), nn.GroupNorm(4, c_skip), nn.ReLU())
        self.conv = conv_block(c_out + c_skip, c_out)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), self.skip(skip)], dim=1))


class LTAE(nn.Module):
    """Lightweight temporal attention: one learned query per head, keys from the dates' features plus their
    time codes. Input (N, T, C); returns the output (N, d_out) and the attention weights (heads, N, T)."""

    def __init__(self, c_in=128, d_model=256, n_head=16, d_k=4, mlp=(256, 128), period=1000.0):
        super().__init__()
        self.n_head, self.d_k, self.d_model, self.period = n_head, d_k, d_model, period
        self.inconv = nn.Sequential(nn.LayerNorm(c_in), nn.Linear(c_in, d_model))
        self.period_code = nn.Embedding(2, d_model)
        self.query = nn.Parameter(torch.randn(n_head, d_k) * (2.0 / d_k) ** 0.5)
        self.key = nn.Linear(d_model, n_head * d_k)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp[0]), nn.LayerNorm(mlp[0]), nn.ReLU(),
                                 nn.Linear(mlp[0], mlp[1]), nn.LayerNorm(mlp[1]), nn.ReLU())

    def time_code(self, days: torch.Tensor) -> torch.Tensor:
        """Sinusoids of the day (N, T) -> (N, T, d_model), as in the transformer's position code."""
        i = torch.arange(self.d_model // 2, device=days.device, dtype=torch.float32)
        freq = self.period ** (-2 * i / self.d_model)
        a = days.float()[..., None] * freq
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    def forward(self, x, days, period, pad):
        N, T, _ = x.shape
        e = self.inconv(x) + self.time_code(days) + self.period_code(period)
        k = self.key(e).view(N, T, self.n_head, self.d_k).permute(2, 0, 1, 3)        # (h, N, T, d_k)
        logit = (k * self.query[:, None, None, :]).sum(-1) / math.sqrt(self.d_k)       # (h, N, T)
        logit = logit.masked_fill(pad[None], float("-inf"))
        att = torch.softmax(logit, dim=-1)
        v = e.view(N, T, self.n_head, self.d_model // self.n_head).permute(2, 0, 1, 3)  # (h, N, T, d/h)
        out = (att[..., None] * v).sum(2).permute(1, 0, 2).reshape(N, self.d_model)
        return self.mlp(out), att


def aggregate(f: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
    """f (B, T, C, H, W), att (h, B, T, H', W') -> (B, C, H, W): each head averages its group of channels."""
    B, T, C, H, W = f.shape
    h = att.shape[0]
    if att.shape[-2:] != (H, W):
        att = F.interpolate(att.reshape(h * B, T, *att.shape[-2:]), size=(H, W), mode="bilinear",
                            align_corners=False).reshape(h, B, T, H, W)
    g = f.reshape(B, T, h, C // h, H, W).permute(2, 0, 1, 3, 4, 5)       # (h, B, T, C/h, H, W)
    out = (att[:, :, :, None] * g).sum(2)                                  # (h, B, C/h, H, W)
    return out.permute(1, 0, 2, 3, 4).reshape(B, C, H, W)


class UTAE(nn.Module):
    def __init__(self, c_in=8, enc=(64, 64, 64, 128), dec=(32, 32, 64, 128), n_head=16, n_classes=2):
        super().__init__()
        self.first = conv_block(c_in, enc[0])
        self.downs = nn.ModuleList([Down(enc[i], enc[i + 1]) for i in range(len(enc) - 1)])
        self.ltae = LTAE(c_in=enc[-1], d_model=256, n_head=n_head, d_k=4, mlp=(256, dec[-1]))
        self.ups = nn.ModuleList([Up(dec[i + 1], enc[i], dec[i]) for i in reversed(range(len(enc) - 1))])
        self.out = nn.Conv2d(dec[0], n_classes, 1)

    def forward(self, x, days, period, pad):
        """x (B, T, 8, H, W) with H, W multiples of 8; days (B, T) days since the after period began;
        period (B, T) 0 before, 1 after; pad (B, T) True for padded (absent) dates. Returns (B, 2, H, W)."""
        B, T = x.shape[:2]
        feats = [self.first(x.flatten(0, 1))]
        for d in self.downs:
            feats.append(d(feats[-1]))
        feats = [f.view(B, T, *f.shape[1:]) for f in feats]
        top = feats[-1]                                                        # (B, T, C, h, w)
        _, _, C, h, w = top.shape
        seq = top.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, C)
        rep = lambda a: a[:, None, None].expand(B, h, w, T).reshape(B * h * w, T)
        out, att = self.ltae(seq, rep(days), rep(period), rep(pad))
        y = out.view(B, h, w, -1).permute(0, 3, 1, 2)
        att = att.view(-1, B, h, w, T).permute(0, 1, 4, 2, 3)                  # (heads, B, T, h, w)
        for up, f in zip(self.ups, reversed(feats[:-1])):
            y = up(y, aggregate(f, att))
        return self.out(y)
