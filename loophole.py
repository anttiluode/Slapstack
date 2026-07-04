"""Test the amplitude-loophole hypothesis on the ep20 checkpoint: on plain images,
do the surviving fine packets have near-zero COEFFICIENTS (per-image adaptation via
amplitude, not gate)? And a causal check: force the fine gates OFF on plain vs busy
and measure the recon cost each way."""
import torch, torch.nn.functional as F
from model import SparseSplatVAE, hard_concrete
from data import VarFaces
torch.manual_seed(0)
m = SparseSplatVAE(28,96,(20,40,80),chunk=20)
m.load_state_dict(torch.load("final_fixed_model.pt")); m.eval()
ds = VarFaces(n=256, size=28, seed=1, tex_vary=False)
xs = torch.stack([ds[i][0] for i in range(256)])
lab = torch.tensor([ds[i][1] for i in range(256)])
with torch.no_grad():
    mu,_,ctx = m.enc(xs)
    raw, glog = m.dec(mu, ctx)
    gate,_ = hard_concrete(glog, training=False)
    fine = (m.ren.layer_of == 2)
    on_fine = fine[None] & (gate > 1e-4)              # surviving fine packets
    # coefficient magnitude of surviving fine packets, per class
    coeff = torch.tanh(raw[...,5:11]).abs().mean(-1)  # (B,N)
    cp = coeff[lab==0][:, fine].mean().item()
    cb = coeff[lab==1][:, fine].mean().item()
    print(f"fine-packet |coeff|: plain {cp:.4f}  busy {cb:.4f}  ratio {cb/max(cp,1e-9):.1f}x")
    # causal: rendered contribution of fine packets = recon(all) - recon(fine off)
    g_off = gate.clone(); g_off[:, fine] = 0.0
    r_all = m.ren(raw, gate); r_off = m.ren(raw, g_off)
    for c,name in ((lab==0,"plain"),(lab==1,"busy")):
        d_on  = F.mse_loss(r_all[c], xs[c]).item()
        d_off = F.mse_loss(r_off[c], xs[c]).item()
        print(f"{name}: MSE gates-on {d_on:.4f}  fine-forced-off {d_off:.4f}  "
              f"cost of losing fine layer {d_off-d_on:+.4f}")
