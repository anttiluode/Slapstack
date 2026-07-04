"""
fashion_data.py — Fashion-MNIST from raw IDX files (fetched from the
zalandoresearch GitHub repo, so it works in sandboxes where the AWS mirror
doesn't), plus a MODEL-INDEPENDENT per-image complexity score.

Why Fashion-MNIST for the layer-gate bet:
  - 28px, same geometry as the VarFaces runs (Nyquist = 14 = top of the fine band).
  - Complexity varies WITHIN the dataset for real reasons: trousers and bags are
    mostly flat regions; pullovers, shirts, sneakers carry fine texture (knit,
    prints, treads). No synthetic label tells the model this — if a per-image
    gap appears, it was discovered, not planted.

The complexity score is the referee, not the model: fraction of AC spectral
energy in the fine layer's radial band (9..14 cycles/image). It is computed
from the INPUT with an FFT, sees no gates, and is fixed before training.
Correlating active-packet count against it is the honest version of the
plain/busy gap — continuous, and defined without access to class labels.
"""

import gzip
import struct
import math
import os
import torch
from torch.utils.data import Dataset

FMNIST_URLS = {
    "train-images-idx3-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

CLASSES = ["tshirt", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "ankleboot"]


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
    return data.reshape(dims)


def maybe_download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    import urllib.request
    for name, url in FMNIST_URLS.items():
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            print(f"fetching {name} ...")
            urllib.request.urlretrieve(url, p)


def fine_band_energy(x, f_lo=9.0, f_hi=14.0):
    """Model-independent complexity referee.
    x: (B,C,H,W) in [0,1]. Returns (B,) = fraction of AC power at radial
    frequency f_lo..f_hi cycles/image. Computed once, before training.
    """
    xg = x.mean(1)                               # grayscale
    H, W = xg.shape[-2:]
    F = torch.fft.fft2(xg)
    P = (F.real ** 2 + F.imag ** 2)
    fy = torch.fft.fftfreq(H) * H                # cycles/image
    fx = torch.fft.fftfreq(W) * W
    R = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = ((R >= f_lo) & (R <= f_hi)).to(P.dtype)
    ac = (R > 0.5).to(P.dtype)                   # exclude DC
    e_band = (P * band).flatten(1).sum(1)
    e_ac = (P * ac).flatten(1).sum(1).clamp_min(1e-12)
    return e_band / e_ac


class FashionIDX(Dataset):
    """Returns (image (1,28,28) float in [0,1], class label, complexity score)."""
    def __init__(self, data_dir, split="train", limit=None):
        maybe_download(data_dir)
        tag = "train" if split == "train" else "t10k"
        imgs = _read_idx(os.path.join(data_dir, f"{tag}-images-idx3-ubyte.gz"))
        labs = _read_idx(os.path.join(data_dir, f"{tag}-labels-idx1-ubyte.gz"))
        if limit is not None:
            imgs, labs = imgs[:limit], labs[:limit]
        self.x = imgs.float().div_(255.0).unsqueeze(1)      # (N,1,28,28)
        self.y = labs.long()
        # precompute the referee once, in chunks
        scores = []
        for i in range(0, len(self.x), 2048):
            scores.append(fine_band_energy(self.x[i:i + 2048]))
        self.c = torch.cat(scores)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.c[i]
