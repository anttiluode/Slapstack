# layergate — the splatstack [B1] bet, cashed in on Fashion-MNIST

Lineage: splatstack's hard-concrete Gabor stack, its [K5] one-way-door autopsy,
and the top open bet in its ledger: *"per-layer per-image gates ... the door
problem shrinks to 3 doors."* This repo implements that bet and moves the
testbed from synthetic VarFaces to a natural dataset. Same rules:
**do not hype, do not lie, just show.**

## The design

`effective_gate[image, packet] = g_pop[packet] × g_layer[image, layer_of(packet)]`

- **g_pop** — static hard-concrete gates, one per packet. The [V]-verified
  population pruning, unchanged.
- **g_layer** — per-image hard-concrete gates, **one per layer** (3 total),
  logits from the encoder ctx. Closing a layer saves λ·(surviving packets in
  it): the aggregated margin the bet predicted (12× on VarFaces numbers).
- **Anti-[K5] clamp** — layer logits pass through `c·tanh(x/c)`, c=2.5.
  σ′ never enters the crawl regime; a "closed" door keeps a nonzero stochastic
  open probability during training, so reconstruction gradient keeps reaching
  it. Whether this suffices was the experiment (see [V2]).

Dataset: Fashion-MNIST, 28px (same geometry/Nyquist as the VarFaces runs),
grayscale (per-packet params 11→7), fetched from the zalandoresearch GitHub
repo so it downloads even in sandboxes. Complexity varies within the dataset
for real reasons — no planted texture label.

**The referee is model-independent:** per-image fraction of AC spectral energy
in the fine layer's radial band (9–14 cycles/image), computed by FFT from the
input before training. Class means order sensibly (sandal .113, shirt .095,
pullover .087 high; ankle boot .047, coat .061, dress .062 low), with 17×
dynamic range across images. Correlating hard active-packet count against this
referee is the natural-image version of the plain/busy gap — and it cannot be
manufactured by the model, because the model never sees it.

## Verified here [V] — CPU sandbox, raw numbers in the JSONs

**[V1] Everything runs end-to-end.** Training, door audit, held-out eval,
Spearman, matched-rate baseline, recon strip. `out_smoke*/results.json`.

**[V2] The [K5] door is two-way for clamped layer gates.** The isolated test
(`door_mechanics.py`, paint 2 ep → slam λ=5e-2 for 2 ep → release λ=0 for 3
ep, fixed 256-image probe):

| phase   | doors open (coarse/mid/fine) | open→close | close→open |
|---------|------------------------------|-----------:|-----------:|
| slam    | 0.00 / 0.00 / 0.00           | 768        | 0          |
| release ep1 | 0.88 / 1.00 / 0.73       | 0          | 669        |
| release ep3 | 1.00 / 1.00 / 0.96       | 0          | 9 (757 cum.)|

**757 of 768 forcibly-closed doors reopened; recon fully recovered** (0.056 vs
0.063 at end of paint). Splatstack's [K5] measured **zero** reopen events in 24
epochs for per-packet gates, three strategies. Aggregation + clamp is the
difference. `door_mechanics_results.json`.

**[V3] Voluntary per-image door use, before any pruning pressure.** At λ=0
after release, the fine layer sits at 0.96 — ~4% of images keep it closed with
no L0 price at all. Small, but it is the per-image mechanism firing on natural
images, unbribed.

## Not claimed — what the GPU run decides [B]

The CPU smoke runs paint to only PSNR ~14.7, where every packet still outbids
any reasonable λ, so gates rationally stay open and the Spearman is noise
(−0.04). **Claims 2 and 3 are open until painted properly:**

- **[B-claim2]** Spearman(active count, fine-band referee) > 0 on the held-out
  test set, with sensible per-class ordering.
- **[B-claim3]** At matched average rate, adaptive allocation beats
  uniform-budget top-k, with the gap concentrated in the busy decile.
- **[B]** λ threshold behavior on natural data (the VarFaces price-cliff
  prediction, re-run where texture value varies per image).

## Run it

```bash
pip install torch
# CPU smoke (~5 min): mechanics only
python3 run_fashion.py --limit 3000 --limit_test 1000 --epochs_paint 2 --epochs_prune 4 --out out_smoke

# the K5 door test in isolation (~4 min CPU)
python3 door_mechanics.py

# the real run (GPU, full 60k train / 10k test)
python3 run_fashion.py --epochs_paint 12 --epochs_prune 18 --batch 256 --workers 4
# then read out_fashion/results.json:
#   spearman_count_vs_complexity   <- claim 2
#   matched_rate.*                 <- claim 3
#   door_summary.two_way           <- claim 1 at full scale
#   per_class.*                    <- commentary (trouser/bag vs pullover/shirt)
```

If the painter stalls below ~PSNR 22 before pruning starts, extend
`--epochs_paint` first; pruning a painter that can't paint measures nothing
(splatstack learned this the hard way). If doors avalanche during the ramp,
lower `--lambda_l0`; the VarFaces cliff analysis says the interesting λ sits
just below the per-packet marginal value, which on this dataset you can read
off `E[active]` vs `rec` in the epoch log.

## Files

- `model_layer.py` — renderer/encoder/decoder + factorized gates + clamp
  ([K2]/[K3] fixes carried over, commented inline)
- `fashion_data.py` — IDX loader (GitHub-hosted, sandbox-friendly) + FFT referee
- `run_fashion.py` — train + all three claims measured, results.json
- `door_mechanics.py` — the isolated [K5] slam/release test
- `door_mechanics_results.json`, `out_smoke*/results.json` — raw numbers behind
  every table above

Built on splatstack by Antti Luode (PerceptionLab); layergate implementation
by Claude.
