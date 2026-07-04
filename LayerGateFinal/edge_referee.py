"""
edge_referee.py — nails (or kills) the boundary-dominance hypothesis on an
existing checkpoint. No retraining.

The loophole measurement showed realized fine-layer value is FLAT across the
FFT texture referee (deciles 0.0094..0.0101, Spearman 0.11, amplitude ratio
0.967 vs VarFaces' 29x). Hypothesis: at 28px under MSE, fine-frequency budget
is spent on OBJECT BOUNDARIES (broadband edges), not interior texture; every
garment has similar silhouette length, so fine value is ~constant. Supporting
prior: directfit showed PLAIN images fit WORSE than busy ones (20.7 vs 25.7
dB) — smooth Gabors struggle with silhouette edges, not knit patterns.

Test: compute per-image realized fine value delta_i (fine door forced off on
the trained model), then correlate against TWO referees:
  texture referee — fine-band FFT energy fraction (interior texture)
  edge referee    — total Sobel gradient magnitude (boundary budget)
plus the spread of delta itself.

Readings:
  spearman(delta, edge) >> spearman(delta, texture)  -> boundary dominance [V]
  delta spread tiny (p90/p10 close to 1)             -> value genuinely
       constant; per-image frequency adaptivity has no headroom on this
       dataset under MSE, and claim 2 is unpassable here BY MEASUREMENT.

Usage: python3 edge_referee.py --ckpt out_fashion/model.pt
"""

import json
import argparse
import torch
import torch.nn.functional as F

from fashion_data import FashionIDX
from model_layer import LayerGatedSplatVAE


def sobel_energy(x):
    """(B,1,H,W) -> (B,) total gradient magnitude: boundary-budget proxy."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).flatten(1).sum(1)


def spearman(a, b):
    def ranks(x):
        r = torch.empty_like(x)
        r[torch.argsort(x)] = torch.arange(len(x), dtype=x.dtype)
        return r
    ra, rb = ranks(a.float()), ranks(b.float())
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return (ra * rb).sum().item() / max((ra.norm() * rb.norm()).item(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_fashion/model.pt")
    ap.add_argument("--data_dir", default="./fmnist")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--packets", default="20,40,80")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--door_mode", default="hc", choices=["hc", "sigmoid"])
    ap.add_argument("--out", default="edge_referee_results.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ppl = tuple(int(v) for v in args.packets.split(","))
    model = LayerGatedSplatVAE(28, 1, args.latent, ppl,
                               door_mode=args.door_mode).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    te = FashionIDX(args.data_dir, "test")
    fine_mask = (model.ren.layer_of == model.ren.L - 1).to(device)

    deltas, edges = [], []
    with torch.no_grad():
        for i in range(0, len(te.x), args.batch):
            x = te.x[i:i + args.batch].to(device)
            mu, _, ctx = model.enc(x)
            raw, layer_logits = model.dec(mu, ctx)
            # force ALL doors open so delta isolates fine-layer value even on
            # checkpoints whose trained doors already close for some images
            g_pop = model.gates(torch.full_like(layer_logits, 10.0),
                                training=False)[0]
            r_on = model.ren(raw, g_pop)
            r_off = model.ren(raw, g_pop * (~fine_mask)[None].float())
            deltas.append((((r_off - x) ** 2).flatten(1).mean(1)
                           - ((r_on - x) ** 2).flatten(1).mean(1)).cpu())
            edges.append(sobel_energy(x).cpu())
    delta = torch.cat(deltas)
    edge = torch.cat(edges)
    tex = te.c

    q = lambda t, p: t.quantile(p).item()
    out = dict(
        spearman_delta_vs_edge=spearman(delta, edge),
        spearman_delta_vs_texture=spearman(delta, tex),
        spearman_edge_vs_texture=spearman(edge, tex),
        delta_p10=q(delta, 0.1), delta_median=q(delta, 0.5),
        delta_p90=q(delta, 0.9),
        delta_spread_p90_over_p10=q(delta, 0.9) / max(q(delta, 0.1), 1e-9),
        # delta medians across EDGE deciles (the new referee's version of the
        # flat table from loophole)
        delta_by_edge_decile=[
            delta[(edge >= edge.quantile(k / 10)) &
                  (edge <= edge.quantile((k + 1) / 10))].median().item()
            for k in range(10)],
        config=vars(args),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    se, st = out["spearman_delta_vs_edge"], out["spearman_delta_vs_texture"]
    if se > 0.4 and se > 2 * abs(st):
        print("\n>> BOUNDARY DOMINANCE: fine value tracks edge budget, not "
              "interior texture. Claim 2's premise is false on this dataset "
              "under MSE — the finding, not a failure.")
    elif out["delta_spread_p90_over_p10"] < 1.6:
        print("\n>> FLAT VALUE: fine value barely varies across images by any "
              "referee. Per-image frequency adaptivity has no headroom here.")
    else:
        print("\n>> Value varies but tracks neither referee cleanly — "
              "unexplained structure; look at extreme-delta images directly.")


if __name__ == "__main__":
    main()
