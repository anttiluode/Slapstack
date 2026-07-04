# FINDINGS — layergate on Fashion-MNIST, complete arc

Splatstack's [B1] open bet (per-layer per-image gates), implemented and taken
to a natural dataset. Six GPU runs, three sandbox diagnostics, one closed
chapter. Every number below traces to a results JSON in this repo or to the
run logs. Rules throughout: do not hype, do not lie, just show.

## Verified [V]

**V1. Hard-concrete doors at the layer level are immobile, and the reason is
not the L0 price.** Three full runs, three prices (0.048, 0.020, 0.011 —
above, inside, and at the bottom of the measured value corridor): zero door
transitions in every case except one global avalanche when the price exceeded
even the busiest image's value. Cause: hard-concrete sampling at the layer
level is multiplicative noise on whole frequency bands. Reconstruction pushes
door logits to the clamp to suppress that noise — a force independent of
packet economics that outbids any price small enough to be selective.

**V2. The sampling-noise tax is ~0.006 MSE (~2 dB), measured three ways.**
rec 0.0157→0.0097 the epoch doors were pinned open (same weights), 0.0089→
0.0152 the epoch sampling resumed, and reproduced in the next run. Per-packet
gate noise averages out over 140 packets; per-layer gate noise does not. The
aggregation that fixes the margin problem ([K5]) concentrates the variance
problem. Same coin, two sides — this is the sentence the whole repo exists
to earn.

**V3. Deterministic sigmoid doors are two-way in live pruning.** 205 closes,
171 reopens across the door phase, stable mixed equilibrium (fine door ~62–86%
open) with genuine per-image variance (per-class fine-door open rates from
0.67 to 0.97). The [K5] one-way door is fully repaired by: aggregation (3
doors) + logit clamp + noise-free estimator. Also verified in isolation:
door_mechanics.py, 757/768 forced-closed doors reopened on release.

**V4. The graded-train/binary-eval gap ([K6]'s ghost) costs 2.5 dB** (train
18.4 dB → eval 15.9 dB on the sigmoid run). Painter-only calibration against
binary doors recovers ~1 dB (15.9 → 16.9).

**V5. The dictionary is not the bottleneck.** Direct per-image fit: 25.7 dB
mean on busy images (27 dB max) at 300 CPU Adam steps, vs 18–20.5 dB amortized.
The recurring "stall" is amortization plus (pre-fix) sampling noise. Bonus
inversion: PLAIN images direct-fit WORSE (20.7 dB) than busy ones — smooth
Gabors handle knit texture better than broadband silhouette edges.

**V6. Population pruning removed nothing at lambda=6e-4.** On a diverse
natural dataset every packet earns its keep on average (139/140 survivors).
VarFaces' 80% pruning reflected that synthetic set's redundancy, not a
universal of the mechanism.

## Killed [K]

**K-A. Claim 2 (active count tracks input complexity) is unpassable on
Fashion-MNIST at 28px under MSE — by measurement, twice over.** The realized
per-image fine-layer value (loophole port, trained checkpoint) spans only
2.07x p90/p10 and tracks NEITHER referee: Spearman 0.11 vs fine-band FFT
texture, −0.06 vs Sobel edge budget. Amplitude ratio busy/plain 0.967 (the
VarFaces analogue measured 29x). The dictionary-level corridor is real (7x,
plain 0.0053 vs busy 0.0376) and amortization erases it: the painter
allocates fine-layer work by internal idiosyncrasies no input statistic we
named captures. This is [K1] at dataset scale: the probe cannot produce a
value spread on the claimed axis, so nothing on that axis can be measured.

**K-B. My original schedule (round 1).** Door priced against 80 unpruned
packets (population pruning never ran first) at a lambda imported blind from
VarFaces, putting the price at 0.048 — above the top of the value corridor.
The avalanche was rational; the test was unpassable as configured.

**K-C. My corridor conflated dictionary value with realized value** (round
2–3). The door decides on realized (amortized) deltas; those turned out flat
where the dictionary spread 7x.

**K-D. Calibration leaks.** (1) lambda=0 during calib made opening free;
recon drifted doors open 0.63→0.87 (fixed in code: calib keeps the door
price). (2) The encoder backdoor: freezing the gate net does not freeze the
doors, because doors read ctx and the painter's gradient reshapes ctx — the
painter picks the lock. Documented, not fixed; any amortized gate conditioned
on a shared representation is manipulable by the pathway it gates.

**K-E. Claim 3 not established.** The matched-rate gap (+0.46 dB overall,
+0.70 dB plain decile, calibrated run) is contaminated: only the adaptive arm
was calibrated; the uniform top-k baseline renders through a painter never
trained for that configuration. A fair test needs a separately trained
fixed-k baseline — not worth the GPU given K-A.

## The one-sentence verdict

The gating mechanism works — two-way, mobile, economically rational, with its
failure modes isolated and priced — and the phenomenon it was built to detect
does not exist on this dataset at this resolution under this loss.

## Open bets [B] — for a different dataset, not more Fashion runs

- **B1'. Interior-texture regime.** 64px+ images where texture area dominates
  boundary length (texture datasets, or CIFAR-scale crops). The validated
  machinery transfers as-is; the directfit + loophole referees decide in
  minutes whether a realized corridor exists BEFORE any training run. That
  pre-flight check is the procedural lesson of this whole arc.
- **B2. What DOES realized fine value track?** 2.07x spread, uncorrelated
  with texture and edges. Garment area, brightness, latent-space geometry?
  One cheap script against the existing checkpoint would say. Curiosity, not
  blocker.
- **B3. The encoder backdoor (K-D.2) as a subject.** Gate manipulation
  through shared representations is a general failure mode of amortized
  conditional computation. Fashion accidentally produced a clean specimen.

## Run index

| run | config | outcome |
|---|---|---|
| GPU 1 | hc, single-phase, lam 6e-4 | avalanche at full lam, all-closed, PSNR 18.27 |
| GPU 2 | hc, paint 50 | same avalanche, PSNR 18.68 |
| GPU 3 | hc, 3-phase, price 0.020 | stalemate all-open, PSNR 20.61 (best), noise tax measured |
| GPU 4 | sigmoid, 3-phase, price 0.020 | two-way doors, K6 gap 2.5 dB, PSNR 15.90 |
| GPU 5 | hc, price 0.011 | stalemate all-open, PSNR 20.48 (hc immobility confirmed) |
| GPU 6 | sigmoid + calib 8 | two-way, calib leaks found, PSNR 16.93 |
| sandbox | door_mechanics | 757/768 reopens: [K5] door is two-way |
| sandbox+GPU | directfit | dictionary >= 25.7 dB busy; corridor 7x |
| GPU | loophole | realized values flat/idiosyncratic; corridor erased |
| GPU | edge_referee | boundary hypothesis dead; spread 2.07x tracks nothing named |

Built on splatstack by Antti Luode (PerceptionLab); layergate implementation
and analysis by Claude.
