"""FC-Siam-diff (Daudt, Le Saux & Boulch 2018, arXiv:1810.08462): a Siamese U-Net for
change detection between two co-registered multi-band images.

- Encoder: 4 levels of conv blocks, the SAME weights applied to both dates (Siamese),
  so both epochs are measured with one instrument.
- Decoder: upsampling path whose skip connections carry |features(after) - features(before)|
  at every scale: the network compares the two dates at each spatial resolution.
- Output: 2-class logits per pixel (no change / change).

Requires PyTorch (installed in .venv-vllm, not in the main project environment).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin: int, cout: int, n: int = 2, p_drop: float = 0.2) -> nn.Sequential:
    layers = []
    for i in range(n):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
                   nn.ReLU(inplace=True), nn.Dropout2d(p_drop)]
    return nn.Sequential(*layers)


class FCSiamDiff(nn.Module):
    def __init__(self, in_ch: int = 5, widths=(16, 32, 64, 128)):
        super().__init__()
        w = widths
        self.enc = nn.ModuleList([block(in_ch, w[0]), block(w[0], w[1]), block(w[1], w[2], 3), block(w[2], w[3], 3)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[3], w[3], 2, stride=2), nn.ConvTranspose2d(w[3], w[2], 2, stride=2),
                                 nn.ConvTranspose2d(w[2], w[1], 2, stride=2), nn.ConvTranspose2d(w[1], w[0], 2, stride=2)])
        self.dec = nn.ModuleList([block(2 * w[3], w[3], 3), block(2 * w[2], w[2], 3),
                                  block(2 * w[1], w[1]), block(2 * w[0], w[0])])
        self.head = nn.Conv2d(w[0], 2, 1)

    def encode(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        return x, skips

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        _, s1 = self.encode(before)
        h, s2 = self.encode(after)  # same weights: Siamese. Decode from the 'after' bottleneck (as in the paper)
        x = self.up[0](h)
        for k in range(4):
            if k > 0:
                x = self.up[k](x)
            skip = torch.abs(s2[3 - k] - s1[3 - k])
            x = self._pad_to(x, skip)
            x = self.dec[k](torch.cat([x, skip], 1))
        return self.head(x)

    @staticmethod
    def _pad_to(x, ref):
        dy, dx = ref.shape[-2] - x.shape[-2], ref.shape[-1] - x.shape[-1]
        return F.pad(x, (0, dx, 0, dy)) if (dy or dx) else x
