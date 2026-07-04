"""
run_fashion.py — train the layer-gated splat VAE on Fashion-MNIST and measure
the three claims. Every number lands in results JSON; nothing is claimed that
isn't in the JSON.

CLAIM 1 (door audit — answers [K5]): with 3 clamped doors per image instead of
  140 saturating ones, do gates transition BOTH ways during training?
  Measured: on a fixed probe batch, per-epoch counts of open->close and
  close->open door flips. [K5] predicts close->open == 0 forever; the bet
  predicts nonzero. Whichever happens is the finding.

CLAIM 2 (per-image adaptation on NATURAL images): does the per-image hard
  active-packet count correlate with a model-independent complexity referee
  (fine-band FFT energy fraction, fixed before training)? Reported as Spearman
  rho on the held-out test set, plus per-class means (trousers/bags should sit
  low, pullovers/shirts/sneakers high — but the classes are only commentary;
  the referee is the score).

CLAIM 3 (allocation is worth something): at MATCHED average rate, does
  adaptive per-image allocation beat giving every image the same budget?
  Baseline = same trained painter, all layer doors forced open, population
  gates cut to global top-k with k = adaptive model's average active count.
  Reported overall and on the top/bottom complexity deciles, where allocation
  should matter most.

Schedules follow splatstack train.py: paint first (no pruning pressure), then
ramp lambda. beta stays tiny (z is a passenger here, [K3] honest residual).

CPU smoke:  python3 run_fashion.py --limit 3000 --epochs_paint 2 --epochs_pop 2 --epochs_prune 3
GPU real:   python3 run_fashion.py --epochs_paint 40 --epochs_pop 12 --epochs_prune 20 --batch 256

The door phase price comes from directfit_fashion.py, not from VarFaces:
measured per-image fine-layer value spread is plain-median 0.0053 / plain-max
0.011 / busy-median 0.038 (dictionary values; the amortized painter realizes
less, so err LOW). lambda_door * n_fine_survivors must land INSIDE that
corridor or the door goes all-open / all-closed rationally.
"""

import os
import json
import math
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fashion_data import FashionIDX, CLASSES
from model_layer import LayerGatedSplatVAE, kl_divergence


def spearman(a, b):
    """Spearman rho without scipy. a, b: 1-D tensors."""
    def ranks(x):
        r = torch.empty_like(x)
        r[torch.argsort(x)] = torch.arange(len(x), dtype=x.dtype)
        return r
    ra, rb = ranks(a.float()), ranks(b.float())
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return (ra * rb).sum().item() / max(
        (ra.norm() * rb.norm()).item(), 1e-12)


def psnr(mse):
    return 10 * math.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def full_eval(model, loader, device, avg_budget_hint=None):
    """All eval-time hard numbers on a loader."""
    model.eval()
    counts, comps, labs, doors_all = [], [], [], []
    mse_sum, n = 0.0, 0
    mse_uni_sum = 0.0
    for x, y, c in loader:
        x = x.to(device)
        tot, per_layer, doors, raw, gate = model.encode_stats(x)
        recon = model.ren(raw, gate)
        mse_sum += F.mse_loss(recon, x, reduction="sum").item() / x[0].numel()
        n += len(x)
        counts.append(tot.cpu()); comps.append(c); labs.append(y)
        doors_all.append(doors.cpu())
    counts = torch.cat(counts); comps = torch.cat(comps); labs = torch.cat(labs)
    doors_all = torch.cat(doors_all)
    res = dict(
        mse=mse_sum / n, psnr=psnr(mse_sum / n),
        avg_active=counts.mean().item(),
        spearman_count_vs_complexity=spearman(counts, comps),
        door_open_rate_per_layer=doors_all.mean(0).tolist(),
        per_class=dict(),
    )
    for k in range(10):
        m = labs == k
        if m.any():
            res["per_class"][CLASSES[k]] = dict(
                avg_active=counts[m].mean().item(),
                avg_complexity=comps[m].mean().item(),
                fine_door_open=doors_all[m, -1].mean().item())
    # matched-rate uniform baseline
    k_match = int(round(avg_budget_hint if avg_budget_hint else counts.mean().item()))
    mse_u, mse_a = 0.0, 0.0
    lo_mask = comps <= comps.quantile(0.1)
    hi_mask = comps >= comps.quantile(0.9)
    dec = {"adaptive_lo": [0.0, 0], "adaptive_hi": [0.0, 0],
           "uniform_lo": [0.0, 0], "uniform_hi": [0.0, 0]}
    seen = 0
    for x, y, c in loader:
        x = x.to(device)
        tot, per_layer, doors, raw, gate = model.encode_stats(x)
        ra = model.ren(raw, gate)
        ru, _ = model.render_uniform(x, topk=k_match)
        ea = ((ra - x) ** 2).flatten(1).mean(1).cpu()
        eu = ((ru - x) ** 2).flatten(1).mean(1).cpu()
        mse_a += ea.sum().item(); mse_u += eu.sum().item()
        idx = torch.arange(seen, seen + len(x)); seen += len(x)
        for name, mask, e in [("adaptive_lo", lo_mask, ea), ("adaptive_hi", hi_mask, ea),
                              ("uniform_lo", lo_mask, eu), ("uniform_hi", hi_mask, eu)]:
            sel = mask[idx]
            dec[name][0] += e[sel].sum().item(); dec[name][1] += int(sel.sum())
    res["matched_rate"] = dict(
        k=k_match,
        psnr_adaptive=psnr(mse_a / n),
        psnr_uniform=psnr(mse_u / n),
        psnr_adaptive_plain_decile=psnr(dec["adaptive_lo"][0] / max(dec["adaptive_lo"][1], 1)),
        psnr_uniform_plain_decile=psnr(dec["uniform_lo"][0] / max(dec["uniform_lo"][1], 1)),
        psnr_adaptive_busy_decile=psnr(dec["adaptive_hi"][0] / max(dec["adaptive_hi"][1], 1)),
        psnr_uniform_busy_decile=psnr(dec["uniform_hi"][0] / max(dec["uniform_hi"][1], 1)),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./fmnist")
    ap.add_argument("--out", default="./out_fashion")
    ap.add_argument("--limit", type=int, default=None, help="train subset size (smoke)")
    ap.add_argument("--limit_test", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs_paint", type=int, default=12)
    ap.add_argument("--epochs_pop", type=int, default=10,
                    help="population pruning, doors pinned OPEN")
    ap.add_argument("--epochs_prune", type=int, default=18,
                    help="door phase: per-image layer gates, pop frozen")
    ap.add_argument("--epochs_calib", type=int, default=6,
                    help="final phase: doors frozen BINARY, painter-only "
                         "retraining to close the [K6] train/eval gap")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--lambda_pop", type=float, default=6e-4,
                    help="price/packet during population pruning")
    ap.add_argument("--lambda_door", type=float, default=2.5e-4,
                    help="price/packet during door phase; window from "
                         "directfit: (plain_max..busy_median)/n_fine_surv")
    ap.add_argument("--door_price", type=float, default=0.020,
                    help="if >0, sets lambda_door = door_price/n_fine_surv at "
                         "pop-freeze so the FINE DOOR total price lands here "
                         "regardless of pop pruning. Directfit corridor: "
                         "0.011..0.038 (dictionary values; err low). Set 0 "
                         "to use --lambda_door directly.")
    ap.add_argument("--layer_clamp", type=float, default=2.5)
    ap.add_argument("--door_mode", default="hc", choices=["hc", "sigmoid"],
                    help="sigmoid = noise-free deterministic doors (H2 fix)")
    ap.add_argument("--packets", default="20,40,80")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    tr = FashionIDX(args.data_dir, "train", limit=args.limit)
    te = FashionIDX(args.data_dir, "test", limit=args.limit_test)
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    dl_te = DataLoader(te, batch_size=256, shuffle=False, num_workers=args.workers)

    ppl = tuple(int(v) for v in args.packets.split(","))
    model = LayerGatedSplatVAE(28, 1, args.latent, ppl,
                               layer_clamp=args.layer_clamp,
                               door_mode=args.door_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # fixed probe batch for the door audit (claim 1)
    probe_x = torch.stack([te[i][0] for i in range(256)]).to(device)
    prev_doors = None
    door_log = []   # per epoch: {"epoch", "open_to_close", "close_to_open", "open_rate"}

    E = args.epochs_paint + args.epochs_pop + args.epochs_prune + args.epochs_calib
    t0 = time.time()
    pop_frozen = False
    doors_frozen = False
    for ep in range(E):
        model.train()
        # three phases: paint (no price) -> pop (population price, doors
        # pinned open) -> door (per-image price, pop frozen)
        if ep < args.epochs_paint:
            phase, lam = "paint", 0.0
        elif ep < args.epochs_paint + args.epochs_pop:
            phase = "pop"
            lam = args.lambda_pop * min(1.0, (ep - args.epochs_paint + 1) /
                                        max(args.epochs_pop // 2, 1))
        elif ep < args.epochs_paint + args.epochs_pop + args.epochs_prune:
            phase = "door"
            d0 = args.epochs_paint + args.epochs_pop
            lam = args.lambda_door * min(1.0, (ep - d0 + 1) /
                                         max(args.epochs_prune // 3, 1))
            if not pop_frozen:
                model.pop_log_alpha.requires_grad_(False)
                pop_frozen = True
                with torch.no_grad():
                    from model_layer import hard_concrete
                    g, _ = hard_concrete(model.pop_log_alpha[None], False)
                    act = (g > 1e-4).float()[0]
                    surv = [int(act[model.ren.layer_of == l].sum())
                            for l in range(model.ren.L)]
                n_fine = max(surv[-1], 1)
                if args.door_price > 0:
                    args.lambda_door = args.door_price / n_fine
                print(f"[pop frozen] survivors per layer {surv} | "
                      f"door price at lambda_door={args.lambda_door:.1e}: "
                      f"fine={args.lambda_door * n_fine:.4f} "
                      f"(directfit window ~0.011..0.038)")
        else:
            # calib keeps the door price ACTIVE. Lesson from the first calib
            # run: at lam=0 opening doors is free, recon always wants them
            # open, and the encoder is a backdoor (doors read ctx, painter
            # gradient reshapes ctx) -> open rate drifted 0.63->0.87 with the
            # gate net frozen. Keeping lam preserves the economics; the
            # encoder-backdoor leak itself remains a documented [K].
            phase, lam = "calib", args.lambda_door
            if not doors_frozen:
                for p in model.dec.layer_gate_net.parameters():
                    p.requires_grad_(False)
                doors_frozen = True
                print("[doors frozen binary] painter-only calibration")
        rec_run, l0_run, nb = 0.0, 0.0, 0
        for x, y, c in dl:
            x = x.to(device)
            out = model(x, force_doors_open=(phase == "pop"),
                        hard_doors=(phase == "calib"))
            rec = F.mse_loss(out["recon"], x)
            kl = kl_divergence(out["mu"], out["lv"])
            l0 = out["p_eff"].sum(1).mean()      # expected active packets/image
            loss = rec + args.beta * kl + lam * l0
            opt.zero_grad(); loss.backward(); opt.step()
            rec_run += rec.item(); l0_run += l0.item(); nb += 1

        # door audit on the probe batch
        with torch.no_grad():
            _, _, doors, _, _ = model.encode_stats(probe_x)
        if prev_doors is not None:
            o2c = int(((prev_doors == 1) & (doors == 0)).sum().item())
            c2o = int(((prev_doors == 0) & (doors == 1)).sum().item())
        else:
            o2c = c2o = 0
        prev_doors = doors.clone()
        door_log.append(dict(epoch=ep, lam=lam,
                             open_to_close=o2c, close_to_open=c2o,
                             open_rate_per_layer=doors.mean(0).tolist()))
        print(f"ep {ep:03d} {phase:5s} lam {lam:.1e} rec {rec_run/nb:.5f} "
              f"E[active] {l0_run/nb:6.1f} doors o->c {o2c:4d} c->o {c2o:4d} "
              f"open {['%.2f' % v for v in doors.mean(0).tolist()]} "
              f"({time.time()-t0:.0f}s)")

    print("evaluating on held-out test set ...")
    res = full_eval(model, dl_te, device)
    res["door_log"] = door_log
    res["config"] = vars(args)
    total_c2o = sum(d["close_to_open"] for d in door_log)
    total_o2c = sum(d["open_to_close"] for d in door_log)
    res["door_summary"] = dict(total_open_to_close=total_o2c,
                               total_close_to_open=total_c2o,
                               two_way=(total_c2o > 0 and total_o2c > 0))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))

    # recon strip: 8 plainest + 8 busiest test images, target over recon
    comps = te.c
    order = torch.argsort(comps)
    pick = torch.cat([order[:8], order[-8:]])
    xs = torch.stack([te[i][0] for i in pick]).to(device)
    with torch.no_grad():
        tot, _, _, raw, gate = model.encode_stats(xs)
        rec = model.ren(raw, gate)
    strip = torch.cat([xs, rec], 0).cpu()
    try:
        from torchvision.utils import save_image
        save_image(strip, os.path.join(args.out, "recon_sample.png"), nrow=16)
    except Exception:
        pass

    print(json.dumps({k: v for k, v in res.items() if k != "door_log"}, indent=2))


if __name__ == "__main__":
    main()
