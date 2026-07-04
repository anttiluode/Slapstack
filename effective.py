"""Per-image EFFECTIVE active packets on the ep20 model: gate open AND rendered
contribution above threshold (packet energy > eps of image energy). This is an
executable inference rule (skip packets below eps -> real compute saving), so the
count is honest. Threshold sweep reported, plus recon cost of actually skipping."""
import torch, torch.nn.functional as F, json, math
from model import SparseSplatVAE, hard_concrete
from data import VarFaces
torch.manual_seed(0)
m = SparseSplatVAE(28,96,(20,40,80),chunk=20)
m.load_state_dict(torch.load("final_fixed_model.pt")); m.eval()
ds = VarFaces(n=512, size=28, seed=1, tex_vary=False)   # held-out
xs = torch.stack([ds[i][0] for i in range(512)])
lab = torch.tensor([ds[i][1] for i in range(512)])
res={}
with torch.no_grad():
    mu,_,ctx = m.enc(xs)
    raw, glog = m.dec(mu, ctx)
    gate,_ = hard_concrete(glog, training=False)
    px,py,sig,th,fr,coeff = m.ren.activate(raw.float())
    # per-packet rendered energy ~ gate * envelope mass * coeff magnitude
    # envelope mass = 2*pi*sigma^2 (integral of squared gaussian ~ pi*sigma^2)
    cmag = coeff.pow(2).sum(dim=(-2,-1)).sqrt()            # (B,N)
    energy = gate * cmag * (sig**2) * 2*math.pi * (28*28)  # pixel-units
    img_rms = xs.pow(2).mean(dim=(1,2,3)).sqrt()
    for eps_frac in (0.001, 0.01, 0.05):
        thr = (eps_frac * img_rms)[:,None]
        eff = (energy.sqrt() > thr) & (gate > 1e-4)
        ep_ = eff.float().sum(1)
        p, b = ep_[lab==0].mean().item(), ep_[lab==1].mean().item()
        # causal: actually skip below-threshold packets, measure recon cost per class
        g2 = gate * eff.float()
        r_skip = m.ren(raw, g2); r_full = m.ren(raw, gate)
        dp = (F.mse_loss(r_skip[lab==0], xs[lab==0]) - F.mse_loss(r_full[lab==0], xs[lab==0])).item()
        db = (F.mse_loss(r_skip[lab==1], xs[lab==1]) - F.mse_loss(r_full[lab==1], xs[lab==1])).item()
        fine = m.ren.layer_of==2
        fp = eff[:,fine][lab==0].float().sum(1).mean().item()
        fb = eff[:,fine][lab==1].float().sum(1).mean().item()
        print(f"eps={eps_frac}: effective plain {p:.1f} busy {b:.1f} "
              f"(fine layer: plain {fp:.1f} busy {fb:.1f}) | "
              f"skip cost MSE plain {dp:+.5f} busy {db:+.5f}")
        res[str(eps_frac)] = dict(plain=round(p,2), busy=round(b,2),
                                  fine_plain=round(fp,2), fine_busy=round(fb,2),
                                  skip_cost_plain=round(dp,6), skip_cost_busy=round(db,6))
json.dump(res, open("effective_results.json","w"), indent=1)
