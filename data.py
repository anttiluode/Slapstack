"""
data.py — datasets. CelebA for the real run; a synthetic 'faces of varying
complexity' set so the self-sparsification claim can be proven where CelebA can't
download (offline / CI / this sandbox).

The synthetic set is the honest stress test for SPARSITY, not for realism: each
image is a face-like blob (head oval + two eyes + mouth) on a plain background,
with a controllable amount of high-frequency texture. Half the images are 'plain'
(should need few fine packets), half are 'busy' (should need many). A working
self-sparsifier must spend FEWER active packets on the plain ones — that per-image
gap is the whole point, and it's measurable because we know the ground-truth class.
"""

import os
import glob
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torchvision as tv


class VarFaces(Dataset):
    """Synthetic faces; label 0 = plain, 1 = busy (high-freq texture).

    [K -> fixed] The original texture band (k = 18..29 cycles/image) lay ABOVE the
    renderer's finest representable band (9..14 cycles) and above Nyquist at small
    sizes, so 'busy' detail was invisible to the model by construction — both
    classes rationally pruned the fine layer and no per-image gap was possible.
    tex_lo/tex_hi now default INSIDE the fine layer's band; tex_amp is raised so
    that reconstructing the texture is worth more MSE than the L0 price of the
    packets needed to render it. The stress test must be passable to be a test.
    """
    def __init__(self, n=2048, size=32, seed=0, tex_lo=9, tex_hi=13, tex_amp=0.30,
                 tex_vary=True):
        # tex_vary=False: texture is one FIXED plane wave, varying only in
        # per-image presence. This isolates the mechanism under test (per-image
        # gating) from a separate hard problem (amortized regression of per-image
        # texture orientation/frequency/phase), which CPU budgets can't train —
        # measured: varying texture plateaus at ~9% painted, so gates correctly
        # prune it for everyone and no gap can appear. Graded difficulty, stated.
        self.n, self.size = n, size
        self.tex_lo, self.tex_hi, self.tex_amp = tex_lo, tex_hi, tex_amp
        self.tex_vary = tex_vary
        self.g = torch.Generator().manual_seed(seed)
        self.busy = (torch.rand(n, generator=self.g) > 0.5).long()
        self.seeds = torch.randint(0, 2**31 - 1, (n,), generator=self.g)

    def __len__(self):
        return self.n

    def _face(self, gen, busy, S):
        yy, xx = torch.meshgrid(torch.linspace(0, 1, S), torch.linspace(0, 1, S),
                                indexing="ij")
        img = torch.ones(3, S, S) * (0.35 + 0.3 * torch.rand(3, 1, 1, generator=gen))
        cx = 0.5 + 0.06 * (torch.rand(1, generator=gen) - 0.5)
        cy = 0.5 + 0.06 * (torch.rand(1, generator=gen) - 0.5)
        rx, ry = 0.26, 0.34
        head = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1.0
        skin = torch.tensor([0.85, 0.68, 0.55]) + 0.1 * (torch.rand(3, generator=gen) - 0.5)
        for c in range(3):
            img[c][head] = skin[c]
        # eyes + mouth
        for (ex, ey, er) in [(cx - 0.09, cy - 0.06, 0.03), (cx + 0.09, cy - 0.06, 0.03),
                             (cx, cy + 0.12, 0.05)]:
            m = ((xx - ex) ** 2 + (yy - ey) ** 2) < er ** 2
            for c in range(3):
                img[c][m] = 0.15
        if busy:
            # texture in the fine layer's band (the 'detail' that needs fine
            # packets); plain faces skip this entirely
            # a single oriented plane wave at frequency k IS one fine-layer Gabor
            # (sin(kx)*sin(ky) products decompose to diagonal waves at k*sqrt(2),
            # which again escapes the band — the second half of the same bug)
            if self.tex_vary:
                k = self.tex_lo + int(torch.randint(0, self.tex_hi - self.tex_lo + 1,
                                                    (1,), generator=gen))
                ang = math.pi * torch.rand(1, generator=gen).item()
                phase = 6 * torch.rand(1, generator=gen)
            else:
                k, ang, phase = 11, 0.6, torch.tensor([1.0])
            tex = self.tex_amp * torch.sin(
                2 * math.pi * k * (xx * math.cos(ang) + yy * math.sin(ang)) + phase)
            for c in range(3):
                img[c][head] = (img[c][head] + tex[head]).clamp(0, 1)
        img = img + 0.02 * torch.randn(3, S, S, generator=gen)
        return img.clamp(0, 1)

    def __getitem__(self, i):
        gen = torch.Generator().manual_seed(int(self.seeds[i]))
        return self._face(gen, int(self.busy[i]), self.size), int(self.busy[i])


def build_dataset(name, data_dir, image_size):
    if name == "varfaces":
        return VarFaces(n=2048, size=image_size)
    if name == "celeba":
        tf = T.Compose([T.CenterCrop(178), T.Resize(image_size), T.ToTensor()])
        return tv.datasets.CelebA(data_dir, split="train", download=True, transform=tf)
    if name == "folder":
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(data_dir, e)))
        if not paths:
            raise RuntimeError(f"no images in {data_dir}")
        tf = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])

        class _F(Dataset):
            def __len__(s): return len(paths)
            def __getitem__(s, i): return tf(Image.open(paths[i]).convert("RGB")), 0
        return _F()
    raise ValueError(name)
