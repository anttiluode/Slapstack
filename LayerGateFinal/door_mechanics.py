"""
door_mechanics.py — the [K5] question in isolation, decoupled from paint quality.

[K5] said: per-packet hard-concrete gates close easily and NEVER reopen (three
strategies, zero reopen events in 24 epochs). The layer-gate design claims two
mechanical fixes (3 aggregated doors + soft logit clamp) make the door two-way.

This tests exactly that, in ~4 minutes CPU, with no dependence on whether the
painter is good yet:

  PHASE SLAM   (2 epochs): lambda = 5e-2 (brutal, ~80x the real price).
               Prediction: doors close. If they don't, gradients are broken.
  PHASE RELEASE(3 epochs): lambda = 0. The only pressure is reconstruction.
               [K5] predicts: closed doors are dead, reopen count stays 0.
               The clamp predicts: stochastic training keeps sampling open
               states inside the clamp range, recon gradient flows, doors that
               are worth their packets REOPEN.

The number that matters: close_to_open transitions during RELEASE. Nonzero and
recon-recovering = door is two-way and [B1] survives its first contact.
Zero = the clamp is not enough and the bet needs a different estimator.
"""

import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fashion_data import FashionIDX
from model_layer import LayerGatedSplatVAE, kl_divergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./fmnist")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs_paint", type=int, default=2)
    ap.add_argument("--epochs_slam", type=int, default=2)
    ap.add_argument("--epochs_release", type=int, default=3)
    ap.add_argument("--lam_slam", type=float, default=5e-2)
    ap.add_argument("--layer_clamp", type=float, default=2.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="door_mechanics_results.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(0)
    tr = FashionIDX(args.data_dir, "train", limit=args.limit)
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    model = LayerGatedSplatVAE(28, 1, 64, (20, 40, 80),
                               layer_clamp=args.layer_clamp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    probe = torch.stack([tr[i][0] for i in range(256)]).to(device)
    prev = None
    log = []

    def epoch(lam, phase, ep):
        nonlocal prev
        model.train()
        rec_run, nb = 0.0, 0
        for x, _, _ in dl:
            x = x.to(device)
            out = model(x)
            loss = (F.mse_loss(out["recon"], x)
                    + 1e-3 * kl_divergence(out["mu"], out["lv"])
                    + lam * out["p_eff"].sum(1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            rec_run += F.mse_loss(out["recon"], x).item(); nb += 1
        with torch.no_grad():
            _, _, doors, _, _ = model.encode_stats(probe)
        o2c = c2o = 0
        if prev is not None:
            o2c = int(((prev == 1) & (doors == 0)).sum().item())
            c2o = int(((prev == 0) & (doors == 1)).sum().item())
        prev = doors.clone()
        row = dict(phase=phase, epoch=ep, lam=lam, rec=rec_run / nb,
                   open_rate=doors.mean(0).tolist(),
                   open_to_close=o2c, close_to_open=c2o)
        log.append(row)
        print(f"{phase:7s} ep{ep} lam {lam:.0e} rec {row['rec']:.5f} "
              f"open {['%.2f' % v for v in row['open_rate']]} "
              f"o->c {o2c:4d} c->o {c2o:4d}")

    e = 0
    for _ in range(args.epochs_paint):
        epoch(0.0, "paint", e); e += 1
    for _ in range(args.epochs_slam):
        epoch(args.lam_slam, "slam", e); e += 1
    for _ in range(args.epochs_release):
        epoch(0.0, "release", e); e += 1

    reopen = sum(r["close_to_open"] for r in log if r["phase"] == "release")
    closed = sum(r["open_to_close"] for r in log if r["phase"] == "slam")
    verdict = dict(doors_closed_in_slam=closed,
                   doors_reopened_in_release=reopen,
                   two_way=bool(closed > 0 and reopen > 0),
                   rec_end_paint=[r["rec"] for r in log if r["phase"] == "paint"][-1],
                   rec_end_release=log[-1]["rec"])
    out = dict(log=log, verdict=verdict, config=vars(args))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
