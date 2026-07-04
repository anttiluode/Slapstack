# splatstack — a self-sparsifying stack of Gabor-packet layers

Lineage: the Gabor-splat VAE (single layer), the Clutch's philosophy of compute-on-demand,
and the Entrain ledger discipline. *Do not hype. Do not lie. Just show.*

**The claim under construction:** instead of guessing how many wave packets an image
needs, learn a binary hard-concrete gate per packet (Louizos et al. 2018). An L0 price
makes gates want to be OFF; reconstruction makes the useful ones stay ON. Active packet
count becomes an *output* of training. Two levels of the claim:

1. **Population level** — the stack prunes itself to the packets the dataset needs. **[V]**
2. **Per-image level** — a plain image switches on fewer packets than a busy one. **[K]**
   for hard-concrete gates, for a measured dynamical reason; the per-image adaptation
   *provably exists* but routes through packet **amplitudes** instead of gates.

All numbers below are from CPU runs on the synthetic `VarFaces` set (label 0 = plain
face, label 1 = same face + a fine texture), 28px, 140 packets in 3 frequency layers
(20 coarse / 40 mid / 80 fine), evaluated on a held-out seed.

## Verified [V]

**[V] Population self-sparsification.** After painting (PSNR 27.2, texture 85% rendered),
an L0 ramp to λ=6e-4 prunes **140 → 28 packets (80%)** while *defending the texture*:
73–75% of texture-band energy still painted, PSNR 24.8. Layer allocation follows
content: 19 coarse / 3 mid / 6 fine survive. In a texture-free variant the fine layer
collapses to ~0–4 instead. Parameter count is a price-mediated outcome, not a guess.

**[V] λ is a real price with a predictable threshold.** The texture's reconstruction
value is ~0.013 MSE spread over ~10 fine packets ≈ **1e-3 per packet**. Prediction:
texture survives pruning only when λ stays below that. Measured: texture intact at
λ=1.2e-3, annihilated between 1.2e-3 and 1.6e-3 (three independent runs). One
back-of-envelope predicted the other experiment's cliff.

**[V] The dictionary is sufficient.** Direct per-image optimization of packet params
(no encoder) fits a busy face at **PSNR 35.2 with 95% of texture energy** — every
failure below is amortization or optimization, never representational capacity.

**[V] Per-image adaptation exists — in the amplitude path.** On the trained model,
surviving fine packets carry **1.7× larger coefficients on busy than plain** images.
Causal check: force the fine layer off and the reconstruction cost is **+0.0003 on
plain vs +0.0086 on busy (29×)**. The network knows, per image, what the fine layer
is worth. The gate head also receives this signal (class-conditioned logit gap +0.42,
correct sign) — the information reaches everything except the binary decision.

## Killed [K] — each one reproducible, each one a trap for reimplementers

**[K1] The stress test was unpassable by construction.** The original busy texture
lived at 18–29 cycles/image; the renderer's finest band tops out at 14 (and Nyquist at
28px is 14). No architecture could have shown a gap — both classes rationally pruned
the fine layer. Second half of the same bug: replacing it with sin(kx)·sin(ky)
decomposes into diagonal plane waves at k√2 = 12.7–18.4 — *still* escaping the band.
Fixed with a single oriented plane wave at k∈[9,13]. If your sparsity probe can't be
represented by your dictionary, you are measuring nothing.

**[K2] Checkpoints were unloadable.** `torch.meshgrid` returns expanded views;
registering them as buffers makes `load_state_dict` fail on reload. Fixed with
`.contiguous()`.

**[K3] Posterior collapse at every β tried.** μ-std → 0.01 at β=0.3 *and* β=0.005.
With `fc_lv` initialized at σ=1, early z is noise, the decoder learns to ignore it,
then KL kills μ at any β. σ-init at 0.02 (`fc_lv.bias = −4`) plus a deterministic
**ctx skip** (encoder conv features → param head) took per-image texture painting
from **2% → 88%**. Honest residual: z stays near-collapsed on CPU budgets even after
the σ fix — per-image painting is carried by the ctx path; z-based sampling is
untested here (open bet).

**[K4] Varying texture doesn't amortize on CPU budgets.** With per-image orientation/
frequency/phase, painting plateaus at 9% — so the gates *correctly* prune fine packets
for everyone (their marginal value really is ~0.001·9%). The fixed-texture world
(`tex_vary=False`) isolates the gating mechanism from this separate regression
problem. Graded difficulty, stated, not hidden.

**[K5] Per-image hard-concrete gates: blocked by a one-way door.** The per-image gap
is the *loss optimum* (closing 6 fine gates on a plain image saves 0.0036 in L0 for
0.0003 recon cost) and gradient descent cannot reach it. Three strategies, all
measured on the frozen-painter refinement where nothing else can move:
- lr 1e-3: **bit-identical counts for 10 epochs** — saturated logits need a ~6-logit
  swing; the gradient scales with σ′(logα)≈0.02 and crawls.
- lr 1e-2: avalanche 28 → 9 packets, −6 dB, and **zero gates reopened in 24 epochs**
  — a closed gate loses its reconstruction gradient and is dead.
- Re-anneal (reset gate logits to neutral): collapse to 14 in 3 epochs — half-open
  gates dim the whole frozen painting, so recon improves fastest by closing
  *everything*, taking the valuable gates with it.
Same door, three angles. Amortized hard-concrete closing is easy, opening is
impossible; the amplitude path (continuous, no saturation) absorbs the per-image
adaptation first.

**[K6] "Effective sparsity" doesn't rescue the gap.** An executable skip rule (drop
packets whose rendered energy < ε of image RMS) shows **no plain/busy difference at
ε = 0.001–0.05** — the amplitude adaptation is graded (1.7×), not a cutoff. Choosing
ε *between* the two class distributions after seeing them would manufacture the gap;
declined.

## Open bets [B] — not claimed

- **Per-layer per-image gates.** One binary decision per (image, layer) instead of
  per packet aggregates the margins: closing the fine *layer* on a plain image is
  worth 6λ vs 0.0003 — a 12× decision margin instead of a marginal one, and 3 gates
  per image instead of 140. The measured 29× causal asymmetry says the signal is
  there; the door problem shrinks to 3 doors. Most promising next step.
- Gates without the one-way door: straight-through Gumbel with temperature annealing,
  or REINFORCE-style estimators that keep closed gates explorable.
- Varying-texture amortization at GPU scale (this is what CelebA needs anyway).
- z revival (free bits / KL annealing) so latent sampling and interpolation work.
- The CelebA + webcam pipeline (`train.py --dataset celeba`) — ships ready, untested
  here (no GPU, dataset host unreachable from this sandbox).

## Files

- `model.py` — renderer + encoder + decoder + hard-concrete gates (K2, K3 fixes inline)
- `data.py` — CelebA loader + `VarFaces` synthetic stress test (K1 fix inline,
  `tex_vary` switch for the graded worlds)
- `train.py` — full training loop with β/L0 warmup and target-budget controller
- `experiment_main.py` — the paint→prune run behind the [V] numbers (~6 min CPU)
- `experiment_refine.py` / `experiment_reanneal.py` — the two failed gate-rescue
  strategies, kept because the failure is the finding
- `directfit.py` — dictionary-sufficiency check (PSNR 35 / 95%)
- `loophole.py` — the amplitude-path causal measurement (29×)
- `effective.py` — the skip-rule measurement (no gap, honestly)
- `*_results.json` — raw numbers for every table above
- `recon_sample.png` — held-out plain (left 8) and busy (right 8) targets over
  reconstructions from the 28-packet model

```bash
pip install torch torchvision
python3 experiment_main.py     # reproduces the [V] population numbers
python3 loophole.py            # reproduces the 29x amplitude-path asymmetry
```

Built by Antti Luode (PerceptionLab) with Claude as implementation collaborator.
