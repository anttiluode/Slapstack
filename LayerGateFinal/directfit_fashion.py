"""
directfit_fashion.py — the referee that decides whether a lambda window EXISTS
on Fashion-MNIST, before any more training money is spent.

Splatstack's directfit.py separated dictionary capacity from amortization on
VarFaces (PSNR 35 direct vs 2% amortized). This is the same move plus one more
number: the PER-IMAGE VALUE of the fine layer,

    value_i = mse(best fit, coarse+mid only) - mse(best fit, all layers)

measured by direct per-image optimization of packet params (no encoder, no
amortization, no gates — pure dictionary). Two verdicts fall out:

  1. DICTIONARY CEILING: best-possible PSNR of the 140-packet dictionary on
     these images. If this is ~18 dB, the painter didn't stall, it FINISHED,
     and the "representational ceiling" claim is [V]. If it's 25+, the stall
     is amortization and the ceiling claim is dead.

  2. THE WINDOW: distribution of value_i over plain-decile vs busy-decile
     images. A per-image door can only produce a gap if there exists a price
     p with  value_plain < p < value_busy  for a meaningful fraction of
     images. Report the spread; if it's tight, claim 2 is untestable on this
     dataset at 28px and THAT is the finding — no schedule or lambda tuning
     can rescue it, same lesson as [K1].

The lambda translation: door price = lambda * (surviving fine packets). With
n_fine survivors after population pruning, the window in lambda units is
(value_plain / n_fine, value_busy / n_fine).

CPU-runnable in minutes for a handful of images; --n_per 16 on GPU for the
full-precision version.
"""

import json
import math
import argparse
import torch

from fashion_data import FashionIDX
from model_layer import MultiScaleRenderer


def direct_fit(renderer, x, layer_mask, steps=400, lr=5e-2, seed=0):
    """Fit raw packet params to a single image (1,C,H,W) with some layers
    masked off. Returns best MSE. Pure dictionary power, no encoder."""
    torch.manual_seed(seed)
    device = x.device
    raw = torch.zeros(1, renderer.N, renderer.K, device=device, requires_grad=True)
    with torch.no_grad():
        raw += 0.01 * torch.randn_like(raw)
    gate = layer_mask[renderer.layer_of][None].float().to(device)   # (1,N) fixed
    opt = torch.optim.Adam([raw], lr=lr)
    best = float("inf")
    for t in range(steps):
        recon = renderer(raw, gate)
        mse = ((recon - x) ** 2).mean()
        opt.zero_grad(); mse.backward(); opt.step()
        best = min(best, mse.item())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./fmnist")
    ap.add_argument("--n_per", type=int, default=8,
                    help="images per decile (plainest / busiest)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--packets", default="20,40,80")
    ap.add_argument("--out", default="directfit_fashion_results.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    te = FashionIDX(args.data_dir, "test")
    order = torch.argsort(te.c)
    picks = dict(plain=order[:args.n_per].tolist(),
                 busy=order[-args.n_per:].tolist())

    ppl = tuple(int(v) for v in args.packets.split(","))
    ren = MultiScaleRenderer(28, ppl, channels=1).to(device)
    L = ren.L
    all_on = torch.ones(L, dtype=torch.bool)
    no_fine = all_on.clone(); no_fine[-1] = False

    rows = []
    for group, idxs in picks.items():
        for i in idxs:
            x = te.x[i][None].to(device)
            m_full = direct_fit(ren, x, all_on, steps=args.steps)
            m_nof = direct_fit(ren, x, no_fine, steps=args.steps)
            rows.append(dict(idx=int(i), group=group,
                             complexity=float(te.c[i]),
                             mse_full=m_full, mse_no_fine=m_nof,
                             psnr_full=10 * math.log10(1 / max(m_full, 1e-12)),
                             fine_value=m_nof - m_full))
            print(f"{group:5s} idx {i:5d} cplx {te.c[i]:.3f} "
                  f"full {m_full:.5f} ({rows[-1]['psnr_full']:.1f} dB) "
                  f"no-fine {m_nof:.5f} value {rows[-1]['fine_value']:+.5f}")

    def stats(g):
        v = [r["fine_value"] for r in rows if r["group"] == g]
        p = [r["psnr_full"] for r in rows if r["group"] == g]
        v_s = sorted(v)
        return dict(fine_value_mean=sum(v) / len(v),
                    fine_value_min=v_s[0], fine_value_max=v_s[-1],
                    fine_value_median=v_s[len(v_s) // 2],
                    psnr_full_mean=sum(p) / len(p))

    s_plain, s_busy = stats("plain"), stats("busy")
    # window: prices p with value_plain_median < p < value_busy_median
    window = dict(
        exists=bool(s_busy["fine_value_median"] > s_plain["fine_value_max"]),
        soft_exists=bool(s_busy["fine_value_median"] > s_plain["fine_value_median"] * 1.5),
        price_low=s_plain["fine_value_median"],
        price_high=s_busy["fine_value_median"],
        note="lambda_door = price / n_surviving_fine_packets")
    out = dict(rows=rows, plain=s_plain, busy=s_busy, window=window,
               config=vars(args))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(dict(plain=s_plain, busy=s_busy, window=window), indent=2))


if __name__ == "__main__":
    main()
