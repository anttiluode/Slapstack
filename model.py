"""
model.py — a self-sparsifying stack of differentiable Gabor-packet layers.

The claim being built (and measured in benchmark.py):
  Instead of guessing how many wave packets an image needs, the network learns a
  binary GATE per packet. An L0-style penalty makes gates want to be OFF; the
  reconstruction loss makes the useful ones stay ON. At convergence the ACTIVE
  packet count is an *output* of training, not a hyperparameter — and it drops
  to a small fraction of the budget with (ideally) no reconstruction cost.

Three design decisions that make this honest rather than hand-wavy:

  1. MULTI-SCALE STACK. Packets live in L "layers", each a fixed spatial frequency
     band (coarse -> fine). This is the "stack of sparse layers": the gate lets the
     network keep many fine packets on a detailed face and switch a whole fine layer
     off on a blank one. Sparsity is per-layer measurable.

  2. HARD-CONCRETE GATES (Louizos et al. 2018) give a true, differentiable L0
     surrogate: gates are genuinely 0 at test time (real parameter savings, not just
     small weights), while training stays gradient-friendly. We report the *expected*
     active count during training and the *hard* active count at eval.

  3. PER-IMAGE (amortized) GATES. The gate logits are produced by the encoder from
     the image, so different inputs switch on different packets — a plain face uses
     fewer than a busy one. That per-image budget is the interesting, measurable thing.

Everything is CPU-runnable at small sizes; the same code scales to GPU/CelebA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 11  # per-packet: pos(2) scale(1) theta(1) freq(1) coeff(3x2)

# hard-concrete constants
_BETA = 2.0 / 3.0
_GAMMA, _ZETA = -0.1, 1.1
_EPS = 1e-6


def hard_concrete(log_alpha, training, u=None):
    """Return gate z in [0,1] and the per-gate 'prob active' used for the L0 penalty.
    At test time returns a deterministic {0..1} clamp of the mean (mostly 0/1)."""
    if training:
        if u is None:
            u = torch.rand_like(log_alpha).clamp(_EPS, 1 - _EPS)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / _BETA)
    else:
        s = torch.sigmoid(log_alpha)
    s_bar = s * (_ZETA - _GAMMA) + _GAMMA
    z = s_bar.clamp(0, 1)
    # probability the hard-concrete gate is non-zero (closed form)
    p_active = torch.sigmoid(log_alpha - _BETA * math.log(-_GAMMA / _ZETA))
    return z, p_active


class MultiScaleRenderer(nn.Module):
    """L layers of Gabor packets; layer l has a fixed frequency band. Gated sum."""
    def __init__(self, image_size=64, packets_per_layer=(64, 128, 256), chunk=64):
        super().__init__()
        self.H = self.W = image_size
        self.ppl = list(packets_per_layer)
        self.N = sum(self.ppl)
        self.L = len(self.ppl)
        self.chunk = chunk
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        # .contiguous() matters: meshgrid returns expanded views, and registering
        # those as buffers makes load_state_dict fail on reload ([K] found the
        # hard way — checkpoints were unloadable)
        self.register_buffer("GX", gx[None, None].contiguous())
        self.register_buffer("GY", gy[None, None].contiguous())
        # per-layer frequency band (coarse -> fine) and anchor grids
        bands, anchors, layer_of = [], [], []
        for l, n in enumerate(self.ppl):
            f_lo = 1.0 + 4.0 * l
            f_hi = f_lo + 5.0
            bands.append((f_lo, f_hi))
            side = int(math.ceil(math.sqrt(n)))
            ax = torch.linspace(0.06, 0.94, side)
            a = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)[:n]
            a = a.clamp(1e-3, 1 - 1e-3)
            anchors.append(torch.log(a / (1 - a)))
            layer_of += [l] * n
        self.bands = bands
        self.register_buffer("anchor_logit", torch.cat(anchors, 0))   # (N,2)
        self.register_buffer("layer_of", torch.tensor(layer_of))       # (N,)

    def activate(self, raw):
        ax = self.anchor_logit[:, 0][None]
        ay = self.anchor_logit[:, 1][None]
        px = torch.sigmoid(ax + raw[..., 0])
        py = torch.sigmoid(ay + raw[..., 1])
        sigma = 0.010 + 0.13 * torch.sigmoid(raw[..., 2])
        theta = raw[..., 3]
        # per-packet frequency confined to its layer's band
        lo = torch.tensor([self.bands[l][0] for l in self.layer_of], device=raw.device)
        hi = torch.tensor([self.bands[l][1] for l in self.layer_of], device=raw.device)
        freq = lo[None] + (hi - lo)[None] * torch.sigmoid(raw[..., 4])
        coeff = torch.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)
        return px, py, sigma, theta, freq, coeff

    def _chunk(self, px, py, sigma, theta, freq, coeff, gate):
        px_, py_ = px[..., None, None], py[..., None, None]
        s_ = sigma[..., None, None]; th = theta[..., None, None]; f_ = freq[..., None, None]
        g_ = gate[..., None, None]
        dx = self.GX - px_; dy = self.GY - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_)) * g_
        cos = torch.cos(2 * math.pi * f_ * xr)
        sin = torch.sin(2 * math.pi * f_ * xr)
        chans = []
        for c in range(3):
            a = coeff[:, :, c, 0][..., None, None]
            b = coeff[:, :, c, 1][..., None, None]
            chans.append((env * (a * cos - b * sin)).sum(1))
        return torch.stack(chans, 1)

    def forward(self, raw, gate):
        px, py, sigma, theta, freq, coeff = self.activate(raw.float())
        B = raw.shape[0]
        out = torch.zeros(B, 3, self.H, self.W, device=raw.device)
        for i in range(0, self.N, self.chunk):
            sl = slice(i, i + self.chunk)
            out = out + self._chunk(px[:, sl], py[:, sl], sigma[:, sl], theta[:, sl],
                                    freq[:, sl], coeff[:, sl], gate[:, sl])
        return torch.sigmoid(out)


class Encoder(nn.Module):
    def __init__(self, image_size=64, latent=128, ch=32):
        super().__init__()
        layers, c_in, sz, c = [], 3, image_size, ch
        while sz > 4:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True)]
            c_in, sz, c = c, sz // 2, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.flat = c_in * sz * sz
        self.fc_mu = nn.Linear(self.flat, latent)
        self.fc_lv = nn.Linear(self.flat, latent)
        # [K4] with lv init ~0 the posterior starts at sigma=1: early z is pure
        # noise, the decoder learns to ignore it, then KL kills mu at ANY beta
        # (measured: mu-std -> 0.01 even at beta=0.005). Start sigma small.
        nn.init.constant_(self.fc_lv.bias, -4.0)
        self.ctx_dim = self.flat                 # full conv features, exposed to gates

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h), h    # h = content side-channel


class Decoder(nn.Module):
    """Outputs packet params from z, and per-packet gate logits from an image-derived
    content signal (NOT the KL-bottlenecked z) so gates can condition per-image."""
    def __init__(self, latent=128, n_packets=448, hidden=512, ctx_dim=None):
        super().__init__()
        self.N = n_packets
        self.trunk = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True))
        # [K5] amortization skip: packet params from z alone never learned to paint
        # per-image texture (direct-fit PSNR 35 / 95% texture vs amortized 2% —
        # the dictionary is fine, the regression is the bottleneck). ctx feeds the
        # param head deterministically; z remains for sampling/interp.
        self.ctx_proj = (nn.Linear(ctx_dim, hidden) if ctx_dim is not None else None)
        self.head_param = nn.Linear(hidden, n_packets * K)
        gate_in = ctx_dim if ctx_dim is not None else hidden
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, 256), nn.LeakyReLU(0.2, True),
            nn.Linear(256, n_packets))
        nn.init.zeros_(self.head_param.bias); self.head_param.weight.data *= 0.1
        nn.init.constant_(self.gate_net[-1].bias, 2.0)   # start mostly ON
        self.gate_net[-1].weight.data *= 0.1

    def forward(self, z, ctx):
        h = self.trunk(z)
        if self.ctx_proj is not None:
            h = h + self.ctx_proj(ctx)
        return self.head_param(h).view(-1, self.N, K), self.gate_net(ctx)


class SparseSplatVAE(nn.Module):
    def __init__(self, image_size=64, latent=128, packets_per_layer=(64, 128, 256),
                 chunk=64):
        super().__init__()
        self.ren = MultiScaleRenderer(image_size, packets_per_layer, chunk)
        self.enc = Encoder(image_size, latent)
        self.dec = Decoder(latent, self.ren.N, ctx_dim=self.enc.ctx_dim)
        self.latent = latent

    def forward(self, x, gate_u=None):
        mu, lv, ctx = self.enc(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        raw, glog = self.dec(z, ctx)
        gate, p_active = hard_concrete(glog, self.training, gate_u)
        recon = self.ren(raw, gate)
        return dict(recon=recon, mu=mu, lv=lv, gate=gate, p_active=p_active, glog=glog)

    @torch.no_grad()
    def encode_gate_stats(self, x):
        """Eval-time HARD active counts, per layer. The honest deployed number."""
        self.eval()
        mu, _, ctx = self.enc(x)
        raw, glog = self.dec(mu, ctx)
        gate, _ = hard_concrete(glog, training=False)
        active = (gate > 1e-4).float()               # truly-nonzero packets
        per_layer = []
        for l in range(self.ren.L):
            m = (self.ren.layer_of == l)
            per_layer.append(active[:, m].sum(1))     # (B,) per-image count in layer l
        return active.sum(1), torch.stack(per_layer, 1), gate


def kl_divergence(mu, lv):
    return -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1))


def l0_cost(p_active):
    """Expected number of active gates per image (differentiable L0 surrogate)."""
    return p_active.sum(1).mean()
