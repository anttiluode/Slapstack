"""Final proof run with all three kills fixed:
  [K1] texture now inside the dictionary's representable band (data.py)
  [K2] contiguous grid buffers so checkpoints reload (model.py)
  [K3] beta 0.3 -> 0.005: posterior collapse made the whole generative path
       input-independent; per-image sparsity needs a per-image painter.
Protocol: 10 recon epochs, 14 prune epochs (lam=2e-3, gate_net LR x10).
Gap measured on HELD-OUT seed. Also tracks mu-std (collapse watch) and
texture-band energy ratio (is the texture actually being painted?)."""
import time, json, math, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from model import SparseSplatVAE, kl_divergence, l0_cost
from data import VarFaces

torch.manual_seed(0)
ds = VarFaces(n=1024, size=28, seed=0, tex_vary=False)
dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
hold = VarFaces(n=512, size=28, seed=1, tex_vary=False)
hl = DataLoader(hold, batch_size=64, shuffle=False)
m = SparseSplatVAE(28, 96, (20,40,80), chunk=20)
gate_p = list(m.dec.gate_net.parameters()); gids={id(p) for p in gate_p}
rest=[p for p in m.parameters() if id(p) not in gids]
opt = torch.optim.Adam([{"params":rest,"lr":3e-4},{"params":gate_p,"lr":3e-4}])
N=m.ren.N; BETA=0.005; hist=[]
print(f"N={N}, params {sum(p.numel() for p in m.parameters())/1e6:.2f}M, beta={BETA}", flush=True)

def band_energy(im):
    f = torch.fft.rfft2(im.mean(1)); H,W = im.shape[-2:]
    fy=torch.fft.fftfreq(H)*H; fx=torch.fft.rfftfreq(W)*W
    rad=(fy[:,None]**2+fx[None,:]**2).sqrt(); mask=(rad>=8.5)&(rad<=13.5)
    return (f.abs()**2*mask).sum(dim=(-2,-1)).mean().item()

def snap(tag):
    m.eval(); rec=0.0; ap=ab=0.0; npl=nb=0; n=0; perp=perb=None
    xs_b=[]; rc_b=[]; mus=[]
    with torch.no_grad():
        for x,lab in hl:
            out=m(x); rec+=F.mse_loss(out["recon"],x).item()*len(x); n+=len(x)
            mus.append(out["mu"])
            tot,pl,_=m.encode_gate_stats(x)
            ap+=tot[lab==0].sum().item(); npl+=(lab==0).sum().item()
            ab+=tot[lab==1].sum().item(); nb+=(lab==1).sum().item()
            p0=pl[lab==0].sum(0); p1=pl[lab==1].sum(0)
            perp=p0 if perp is None else perp+p0
            perb=p1 if perb is None else perb+p1
            xs_b.append(x[lab==1]); rc_b.append(out["recon"][lab==1])
    mse=rec/n; psnr=float(10*torch.log10(torch.tensor(1/mse)))
    xb=torch.cat(xs_b); rb=torch.cat(rc_b)
    tex_ratio = band_energy(rb)/max(band_energy(xb),1e-9)
    mustd = torch.cat(mus).std(0).mean().item()
    row=dict(tag=tag,psnr=round(psnr,2),plain=round(ap/npl,1),busy=round(ab/nb,1),
             layers_plain=[round(v,1) for v in (perp/npl).tolist()],
             layers_busy=[round(v,1) for v in (perb/nb).tolist()],
             tex_painted=round(tex_ratio,3), mu_std=round(mustd,3))
    hist.append(row)
    print(f"  [{tag}] PSNR {psnr:.1f} | plain {row['plain']} busy {row['busy']} /{N} | "
          f"Lp {row['layers_plain']} Lb {row['layers_busy']} | "
          f"tex painted {100*tex_ratio:.0f}% | mu-std {mustd:.2f}", flush=True)

t0=time.time()
for ep in range(12):
    m.train()
    for x,_ in dl:
        out=m(x); loss=F.mse_loss(out["recon"],x)+BETA*kl_divergence(out["mu"],out["lv"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    if ep%3==2: snap(f"recon ep{ep+1}")
LAM=6e-4
for ep in range(20):
    lam = LAM * min(1.0, (ep+1)/6.0)   # ramp: no pruning shock, gates can defend
    m.train()
    for x,_ in dl:
        out=m(x)
        loss=F.mse_loss(out["recon"],x)+BETA*kl_divergence(out["mu"],out["lv"])+lam*l0_cost(out["p_active"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    if ep%2==1: snap(f"prune ep{ep+1}")
print(f"total {time.time()-t0:.0f}s", flush=True)
torch.save(m.state_dict(),"final_fixed_model.pt")
json.dump(hist, open("final_fixed_results.json","w"), indent=1)
