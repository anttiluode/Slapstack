"""
loophole_fashion.py — the loophole.py move, on the trained Fashion checkpoint.

Run 3 left two live explanations for why zero doors moved at price 0.020:

  (H1) RATIONAL ALL-OPEN: the REALIZED per-image fine value (delta-MSE of
       closing the fine door on the amortized painter) sits above 0.020 for
       essentially every image. The directfit corridor (0.011..0.038) was
       DICTIONARY values; the amortized painter's coarse+mid arm is not at its
       own optimum and can lean on fine packets, so realized > dictionary is
       possible. If true: the price was wrong, move it into the REALIZED
       corridor this script measures.

  (H2) NOISE-SUPPRESSION STALEMATE: recon pushes door logits to the clamp not
       because of packet value but to suppress hard-concrete sampling noise
       (measured cost ~0.006 MSE when sampling is on). If true: the realized
       corridor exists and the price was inside it, yet doors never moved —
       the estimator needs a noise fix, not a price fix.

This script decides between them on the trained model, no retraining:

  For each held-out image: render with doors as trained (all open), then with
  the fine layer forced off. delta_i = mse_off_i - mse_on_i is the realized
  fine value. Report the distribution by complexity decile, where the tried
  price (0.020) sits in it, the realized corridor (plain-decile max ..
  busy-decile median), and Spearman(delta, complexity) — the amplitude-path
  information check (VarFaces analogue measured 29x there).

Also reports the fine-packet coefficient-energy ratio busy/plain: if >1 the
per-image signal reached the amplitude path (splatstack's [V]), even though
the binary door never used it.

Usage:
    python3 loophole_fashion.py --ckpt out_fashion/model.pt
"""

import json
import math
import argparse
import torch

from fashion_data import FashionIDX
from model_layer import LayerGatedSplatVAE, hard_concrete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_fashion/model.pt")
    ap.add_argument("--data_dir", default="./fmnist")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--packets", default="20,40,80")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--tried_price", type=float, default=0.020)
    ap.add_argument("--out", default="loophole_fashion_results.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ppl = tuple(int(v) for v in args.packets.split(","))
    model = LayerGatedSplatVAE(28, 1, args.latent, ppl).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    te = FashionIDX(args.data_dir, "test")
    fine_mask = (model.ren.layer_of == model.ren.L - 1).to(device)

    deltas, comps, fine_energy = [], [], []
    with torch.no_grad():
        for i in range(0, len(te.x), args.batch):
            x = te.x[i:i + args.batch].to(device)
            mu, _, ctx = model.enc(x)
            raw, layer_logits = model.dec(mu, ctx)
            gate, _, _, _ = model.gates(layer_logits, training=False)
            r_on = model.ren(raw, gate)
            r_off = model.ren(raw, gate * (~fine_mask)[None].float())
            e_on = ((r_on - x) ** 2).flatten(1).mean(1)
            e_off = ((r_off - x) ** 2).flatten(1).mean(1)
            deltas.append((e_off - e_on).cpu())
            comps.append(te.c[i:i + args.batch])
            # amplitude path: coefficient energy of fine packets, per image
            coeff = torch.tanh(raw[..., 5:])
            fe = (coeff[:, fine_mask] ** 2).flatten(1).sum(1)
            fine_energy.append(fe.cpu())
    delta = torch.cat(deltas); comp = torch.cat(comps)
    fen = torch.cat(fine_energy)

    def spearman(a, b):
        def ranks(x):
            r = torch.empty_like(x)
            r[torch.argsort(x)] = torch.arange(len(x), dtype=x.dtype)
            return r
        ra, rb = ranks(a.float()), ranks(b.float())
        ra = ra - ra.mean(); rb = rb - rb.mean()
        return (ra * rb).sum().item() / max((ra.norm() * rb.norm()).item(), 1e-12)

    lo = comp <= comp.quantile(0.1)
    hi = comp >= comp.quantile(0.9)
    d_lo, d_hi = delta[lo], delta[hi]
    corridor = dict(
        realized_plain_median=d_lo.median().item(),
        realized_plain_p90=d_lo.quantile(0.9).item(),
        realized_busy_median=d_hi.median().item(),
        realized_corridor_exists=bool(d_hi.median() > d_lo.quantile(0.9)),
        suggested_door_price=float((d_lo.quantile(0.9) + d_hi.median()) / 2),
    )
    verdict = dict(
        tried_price=args.tried_price,
        frac_images_below_tried_price=float((delta < args.tried_price).float().mean()),
        spearman_delta_vs_complexity=spearman(delta, comp),
        amplitude_ratio_busy_over_plain=float(fen[hi].mean() / max(fen[lo].mean(), 1e-12)),
        delta_deciles=[delta[(comp >= comp.quantile(q / 10)) &
                             (comp <= comp.quantile((q + 1) / 10))].median().item()
                       for q in range(10)],
        H1_rational_all_open=bool((delta < args.tried_price).float().mean() < 0.05),
        H2_noise_stalemate=bool((delta < args.tried_price).float().mean() > 0.20),
    )
    out = dict(corridor=corridor, verdict=verdict,
               delta_mean=delta.mean().item(), config=vars(args))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    if verdict["H2_noise_stalemate"]:
        print("\n>> H2: a meaningful fraction of images sat BELOW the tried "
              "price and the doors still never closed -> noise-suppression "
              "stalemate. Fix the estimator (see README next-steps), not the price.")
    elif verdict["H1_rational_all_open"]:
        print("\n>> H1: essentially every image's realized fine value exceeds "
              "the tried price -> all-open was rational. Re-run the door phase "
              f"with --door_price {corridor['suggested_door_price']:.4f} "
              "(the REALIZED corridor midpoint).")
    else:
        print("\n>> Mixed regime: some images below price, doors closed for "
              "none. Partial stalemate; both the price and the estimator "
              "likely need attention.")


if __name__ == "__main__":
    main()
