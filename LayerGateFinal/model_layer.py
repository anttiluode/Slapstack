"""
model_layer.py — the [B1] open bet from splatstack, implemented:

    "Per-layer per-image gates. One binary decision per (image, layer) instead
     of per packet aggregates the margins ... the door problem shrinks to 3 doors."

Design, and how each K from the splatstack ledger is answered:

  GATE FACTORIZATION.  effective_gate[b, n] = g_pop[n] * g_layer[b, layer_of[n]]
    - g_pop:   STATIC hard-concrete gates, one per packet, learned parameters.
               These do the [V]-verified population pruning (dataset-level).
    - g_layer: PER-IMAGE hard-concrete gates, one per (image, layer), logits
               from the encoder ctx. These carry the per-image budget — L
               decisions per image, not N. Closing a layer saves
               lambda * (sum of surviving g_pop in that layer): the aggregated
               margin the bet predicts (measured 12x in the VarFaces ledger).

  ANTI-[K5] DOOR FIX.  The per-packet gates died two ways: saturated logits
  (sigma' ~ 0.02, gradient crawls) and hard-zero gates (recon gradient gone,
  gate can never reopen). Two mechanical countermeasures, both cheap:
    (a) layer-gate logits are soft-clamped to +/-CLAMP via CLAMP*tanh(x/CLAMP);
        sigma' never falls below sigma'(CLAMP), so no crawl regime exists.
    (b) training-time stochasticity of hard-concrete means a negatively-biased
        gate still opens on some u draws inside the clamp range, so a "closed"
        layer keeps sampling reconstruction gradient. The clamp guarantees the
        reopen probability never hits 0 during training.
  Whether this is ENOUGH is the experiment, not the assumption. run_fashion.py
  logs door transitions (open->close and close->open counts on a fixed probe
  batch every epoch); if close->open stays at 0, the door is still one-way and
  the bet dies honestly.

  [K2] fix kept (contiguous meshgrid buffers). [K3] fix kept (fc_lv bias -4,
  ctx skip to the param head).

  CHANNELS PARAMETERIZED. Fashion-MNIST is grayscale; per-packet param count
  drops from 11 to 7 (pos 2, scale 1, theta 1, freq 1, coeff 1x2). No reason
  to pay for RGB coefficients that would just learn to be equal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# hard-concrete constants (Louizos et al. 2018)
_BETA = 2.0 / 3.0
_GAMMA, _ZETA = -0.1, 1.1
_EPS = 1e-6
_SHIFT = _BETA * math.log(-_GAMMA / _ZETA)


def hard_concrete(log_alpha, training, u=None):
    if training:
        if u is None:
            u = torch.rand_like(log_alpha).clamp(_EPS, 1 - _EPS)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / _BETA)
    else:
        s = torch.sigmoid(log_alpha)
    s_bar = s * (_ZETA - _GAMMA) + _GAMMA
    z = s_bar.clamp(0, 1)
    p_active = torch.sigmoid(log_alpha - _SHIFT)
    return z, p_active


class MultiScaleRenderer(nn.Module):
    """Unchanged geometry from splatstack; channels now a parameter."""
    def __init__(self, image_size=28, packets_per_layer=(20, 40, 80),
                 channels=1, chunk=64):
        super().__init__()
        self.H = self.W = image_size
        self.C = channels
        self.ppl = list(packets_per_layer)
        self.N = sum(self.ppl)
        self.L = len(self.ppl)
        self.K = 5 + 2 * channels
        self.chunk = chunk
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        # [K2] .contiguous() or checkpoints are unloadable
        self.register_buffer("GX", gx[None, None].contiguous())
        self.register_buffer("GY", gy[None, None].contiguous())
        bands, anchors, layer_of = [], [], []
        for l, n in enumerate(self.ppl):
            f_lo = 1.0 + 4.0 * l
            bands.append((f_lo, f_lo + 5.0))
            side = int(math.ceil(math.sqrt(n)))
            ax = torch.linspace(0.06, 0.94, side)
            a = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)[:n]
            anchors.append(torch.log(a.clamp(1e-3, 1 - 1e-3) /
                                     (1 - a.clamp(1e-3, 1 - 1e-3))))
            layer_of += [l] * n
        self.bands = bands
        self.register_buffer("anchor_logit", torch.cat(anchors, 0))
        self.register_buffer("layer_of", torch.tensor(layer_of))
        self.register_buffer("f_lo", torch.tensor([bands[l][0] for l in layer_of]))
        self.register_buffer("f_hi", torch.tensor([bands[l][1] for l in layer_of]))

    def activate(self, raw):
        px = torch.sigmoid(self.anchor_logit[:, 0][None] + raw[..., 0])
        py = torch.sigmoid(self.anchor_logit[:, 1][None] + raw[..., 1])
        sigma = 0.010 + 0.13 * torch.sigmoid(raw[..., 2])
        theta = raw[..., 3]
        freq = self.f_lo[None] + (self.f_hi - self.f_lo)[None] * torch.sigmoid(raw[..., 4])
        coeff = torch.tanh(raw[..., 5:]).reshape(*raw.shape[:2], self.C, 2)
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
        for c in range(self.C):
            a = coeff[:, :, c, 0][..., None, None]
            b = coeff[:, :, c, 1][..., None, None]
            chans.append((env * (a * cos - b * sin)).sum(1))
        return torch.stack(chans, 1)

    def forward(self, raw, gate):
        px, py, sigma, theta, freq, coeff = self.activate(raw.float())
        out = torch.zeros(raw.shape[0], self.C, self.H, self.W, device=raw.device)
        for i in range(0, self.N, self.chunk):
            sl = slice(i, i + self.chunk)
            out = out + self._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                                    theta[:, sl], freq[:, sl], coeff[:, sl],
                                    gate[:, sl])
        return torch.sigmoid(out)


class Encoder(nn.Module):
    def __init__(self, image_size=28, channels=1, latent=64, ch=32):
        super().__init__()
        layers, c_in, sz, c = [], channels, image_size, ch
        while sz > 4:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c),
                       nn.LeakyReLU(0.2, True)]
            c_in, sz, c = c, sz // 2, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.flat = c_in * sz * sz
        self.fc_mu = nn.Linear(self.flat, latent)
        self.fc_lv = nn.Linear(self.flat, latent)
        nn.init.constant_(self.fc_lv.bias, -4.0)   # [K3] sigma-init fix
        self.ctx_dim = self.flat

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h), h


class Decoder(nn.Module):
    """Packet params from z (+ctx skip, [K3]); LAYER gate logits from ctx."""
    def __init__(self, latent, n_packets, n_layers, K, ctx_dim,
                 hidden=512, layer_clamp=2.5):
        super().__init__()
        self.N, self.L, self.K = n_packets, n_layers, K
        self.layer_clamp = layer_clamp
        self.trunk = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True))
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        self.head_param = nn.Linear(hidden, n_packets * K)
        self.layer_gate_net = nn.Sequential(
            nn.Linear(ctx_dim, 128), nn.LeakyReLU(0.2, True),
            nn.Linear(128, n_layers))
        nn.init.zeros_(self.head_param.bias); self.head_param.weight.data *= 0.1
        nn.init.constant_(self.layer_gate_net[-1].bias, 1.5)   # start open
        self.layer_gate_net[-1].weight.data *= 0.1

    def forward(self, z, ctx):
        h = self.trunk(z) + self.ctx_proj(ctx)
        raw = self.head_param(h).view(-1, self.N, self.K)
        lg = self.layer_gate_net(ctx)
        # anti-[K5] soft clamp: logits live in (-clamp, +clamp); the gradient of
        # sigma never enters the crawl regime and reopen probability never
        # hits 0 during stochastic training. tanh keeps it differentiable.
        lg = self.layer_clamp * torch.tanh(lg / self.layer_clamp)
        return raw, lg


class LayerGatedSplatVAE(nn.Module):
    def __init__(self, image_size=28, channels=1, latent=64,
                 packets_per_layer=(20, 40, 80), chunk=64, layer_clamp=2.5,
                 door_mode="hc"):
        # door_mode:
        #   "hc"      — stochastic hard-concrete doors (original). Two-way via
        #               sampling, but the sampling is multiplicative noise on
        #               whole frequency bands: measured cost ~0.006 MSE on
        #               Fashion (run 3, ep39->40->52), and it creates a
        #               noise-suppression force pinning logits at +clamp.
        #   "sigmoid" — deterministic sigmoid doors during training, hard
        #               threshold (>0.5) at eval. Zero sampling noise, gradient
        #               alive at both clamp ends by construction, trivially
        #               two-way. Honest cost: doors are GRADED in training
        #               (the [K6] concern) and only binarized at eval.
        super().__init__()
        self.door_mode = door_mode
        self.ren = MultiScaleRenderer(image_size, packets_per_layer, channels, chunk)
        self.enc = Encoder(image_size, channels, latent)
        self.dec = Decoder(latent, self.ren.N, self.ren.L, self.ren.K,
                           self.enc.ctx_dim, layer_clamp=layer_clamp)
        # static population gates: one logit per packet, start open
        self.pop_log_alpha = nn.Parameter(torch.full((self.ren.N,), 2.0))
        self.latent = latent

    def gates(self, layer_logits, training):
        """Returns effective gate (B,N), and the pieces for the L0 prices."""
        g_pop, p_pop = hard_concrete(self.pop_log_alpha[None], training)     # (1,N)
        if self.door_mode == "sigmoid":
            g_lay = torch.sigmoid(layer_logits)
            if not training:
                g_lay = (g_lay > 0.5).float()
            p_lay = torch.sigmoid(layer_logits)
        else:
            g_lay, p_lay = hard_concrete(layer_logits, training)             # (B,L)
        g_lay_full = g_lay[:, self.ren.layer_of]                             # (B,N)
        p_lay_full = p_lay[:, self.ren.layer_of]
        return g_pop * g_lay_full, p_pop, p_lay, p_pop * p_lay_full

    def forward(self, x, force_doors_open=False, hard_doors=False):
        mu, lv, ctx = self.enc(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        raw, layer_logits = self.dec(z, ctx)
        if force_doors_open:
            # pop-pruning phase: doors pinned open so population gates are
            # priced against their TRUE reconstruction value, before any
            # per-image door decision exists. Fixes the ordering bug where
            # the fine door was priced against 80 unpruned packets and
            # rationally amputated the whole limb.
            gate, p_pop = None, None
            g_pop, p_pop = hard_concrete(self.pop_log_alpha[None], self.training)
            gate = g_pop.expand(x.shape[0], -1)
            p_lay = torch.ones(x.shape[0], self.ren.L, device=x.device)
            p_eff = p_pop.expand(x.shape[0], -1)
        elif hard_doors:
            # calibration phase: doors frozen at their BINARY eval states so
            # the painter re-calibrates amplitudes against what eval will
            # actually apply. Closes the graded-train/binary-eval gap ([K6]
            # ghost: measured 2.5 dB on the sigmoid run). No gradient to
            # doors; painter-only adaptation.
            g_pop, p_pop = hard_concrete(self.pop_log_alpha[None], self.training)
            if self.door_mode == "sigmoid":
                d = (torch.sigmoid(layer_logits) > 0.5).float()
            else:
                d, _ = hard_concrete(layer_logits, training=False)
                d = (d > 1e-4).float()
            d = d.detach()
            gate = g_pop * d[:, self.ren.layer_of]
            p_lay = d
            p_eff = p_pop * d[:, self.ren.layer_of]
        else:
            gate, p_pop, p_lay, p_eff = self.gates(layer_logits, self.training)
        recon = self.ren(raw, gate)
        return dict(recon=recon, mu=mu, lv=lv, gate=gate, p_pop=p_pop,
                    p_lay=p_lay, p_eff=p_eff, layer_logits=layer_logits)

    @torch.no_grad()
    def encode_stats(self, x):
        """Eval-time HARD numbers: per-image active packet count, per-layer
        counts, and the binary layer-door states. The deployed truth."""
        self.eval()
        mu, _, ctx = self.enc(x)
        raw, layer_logits = self.dec(mu, ctx)
        gate, _, _, _ = self.gates(layer_logits, training=False)
        active = (gate > 1e-4).float()
        per_layer = torch.stack([active[:, self.ren.layer_of == l].sum(1)
                                 for l in range(self.ren.L)], 1)
        g_lay, _ = hard_concrete(layer_logits, training=False)
        doors = (g_lay > 1e-4).float()                        # (B,L) binary
        return active.sum(1), per_layer, doors, raw, gate

    @torch.no_grad()
    def render_uniform(self, x, force_layers_open=True, topk=None):
        """Baselines: (a) all layer doors forced open (uniform budget);
        (b) optionally keep only global top-k population packets (matched-rate
        uniform). Returns recon and per-image active count."""
        self.eval()
        mu, _, ctx = self.enc(x)
        raw, _ = self.dec(mu, ctx)
        g_pop, _ = hard_concrete(self.pop_log_alpha[None], training=False)
        g = g_pop.expand(x.shape[0], -1).clone()
        if topk is not None:
            keep = torch.zeros_like(self.pop_log_alpha, dtype=torch.bool)
            keep[torch.topk(self.pop_log_alpha, k=min(topk, self.ren.N)).indices] = True
            g = g * keep[None].float()
        recon = self.ren(raw, g)
        return recon, (g > 1e-4).float().sum(1)


def kl_divergence(mu, lv):
    return -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1))
