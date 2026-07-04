"""
train.py — train the self-sparsifying stack.

Loss = MSE(recon, x) + beta*KL + lambda_l0 * (expected active packets per image).

Two schedules that make sparsification behave:
  - beta warmup (standard VAE)
  - L0 warmup: lambda_l0 ramps from 0 so reconstruction is learned BEFORE pruning
    pressure starts (pruning a network that can't yet reconstruct just kills it).

Optional target-budget controller: instead of picking lambda_l0 by hand, give a
target average active-packet count and let a simple integral controller adjust
lambda to hit it. This is what makes 'parameter count' a dial you set by intent
rather than trial-and-error.

CPU-friendly defaults; --dataset celeba --image_size 64 for the GPU run.
"""

import os
import math
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import SparseSplatVAE, kl_divergence, l0_cost
from data import build_dataset


def evaluate(model, loader, device, max_batches=8):
    model.eval()
    rec_sum, n = 0.0, 0
    act_plain, act_busy, np_, nb_ = 0.0, 0.0, 0, 0
    per_layer_sum = None
    with torch.no_grad():
        for bi, (x, lab) in enumerate(loader):
            if bi >= max_batches:
                break
            x = x.to(device)
            out = model(x)
            rec_sum += F.mse_loss(out["recon"], x).item() * len(x); n += len(x)
            tot, per, _ = model.encode_gate_stats(x)
            lab = lab.to(device)
            act_plain += tot[lab == 0].sum().item(); np_ += (lab == 0).sum().item()
            act_busy += tot[lab == 1].sum().item(); nb_ += (lab == 1).sum().item()
            per_layer_sum = per.sum(0) if per_layer_sum is None else per_layer_sum + per.sum(0)
    mse = rec_sum / max(n, 1)
    return dict(mse=mse, psnr=10 * math.log10(1.0 / max(mse, 1e-9)),
                act_plain=act_plain / max(np_, 1), act_busy=act_busy / max(nb_, 1),
                per_layer=(per_layer_sum / max(n, 1)).tolist() if per_layer_sum is not None else [])


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out, exist_ok=True)
    ppl = tuple(int(x) for x in args.packets_per_layer.split(","))
    ds = build_dataset(args.dataset, args.data_dir, args.image_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True)
    model = SparseSplatVAE(args.image_size, args.latent, ppl, args.chunk).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device {device} | dataset {args.dataset} | packets {ppl} (N={model.ren.N}) | "
          f"params {n_par/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    lam = 0.0
    fixed = next(iter(dl))[0][:16].to(device)
    log = open(os.path.join(args.out, "log.csv"), "a")
    step = 0
    for ep in range(args.epochs):
        model.train()
        beta = args.beta * min(1.0, (ep + 1) / max(args.beta_warmup, 1))
        l0_ramp = min(1.0, max(0.0, (ep + 1 - args.l0_warmup) / max(args.l0_ramp, 1)))
        t0 = time.time(); rs = ks = ls = 0.0; nb = 0
        for x, _ in dl:
            x = x.to(device)
            out = model(x)
            rec = F.mse_loss(out["recon"], x)
            kld = kl_divergence(out["mu"], out["lv"])
            active = l0_cost(out["p_active"])
            loss = rec + beta * kld + lam * active
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            rs += rec.item(); ks += kld.item(); ls += active.item(); nb += 1; step += 1
            if args.smoke and step >= 3:
                print(f"smoke OK rec={rec.item():.4f} active~{active.item():.0f}/{model.ren.N}")
                return
        # budget controller adjusts lambda toward a target active count
        avg_active = ls / nb
        if args.target_active > 0 and l0_ramp > 0:
            err = (avg_active - args.target_active) / model.ren.N
            lam = float(max(0.0, lam + args.lam_lr * err))
        else:
            lam = args.lam * l0_ramp
        ev = evaluate(model, dl, device)
        print(f"ep {ep+1:3d} rec {rs/nb:.4f} (PSNR {ev['psnr']:.1f}) kl {ks/nb:6.1f} "
              f"active~{avg_active:.0f}/{model.ren.N} lam {lam:.4f} | "
              f"eval plain {ev['act_plain']:.0f} busy {ev['act_busy']:.0f} "
              f"layers {[round(v,1) for v in ev['per_layer']]} | {time.time()-t0:.0f}s")
        log.write(f"{ep+1},{rs/nb:.5f},{ev['psnr']:.3f},{avg_active:.2f},{lam:.5f},"
                  f"{ev['act_plain']:.2f},{ev['act_busy']:.2f}\n"); log.flush()
        model.eval()
        with torch.no_grad():
            torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
            out = model(fixed)
            save_image(torch.cat([fixed, out["recon"].clamp(0, 1)]),
                       os.path.join(args.out, f"recon_{ep+1:03d}.png"), nrow=16)
    log.close()
    print("done ->", args.out)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="varfaces",
                   choices=["varfaces", "celeba", "folder"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out", default="./runs/sparse")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--packets_per_layer", default="64,128,256")
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta_warmup", type=int, default=8)
    p.add_argument("--lam", type=float, default=2e-3, help="fixed L0 weight (if no target)")
    p.add_argument("--l0_warmup", type=int, default=10, help="epochs before pruning starts")
    p.add_argument("--l0_ramp", type=int, default=15)
    p.add_argument("--target_active", type=float, default=0.0,
                   help=">0 enables the budget controller toward this many packets")
    p.add_argument("--lam_lr", type=float, default=5e-3)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
