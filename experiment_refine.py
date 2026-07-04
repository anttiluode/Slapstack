"""Phase 3, gate refinement: painter frozen, only gate_net trains.
With params fixed, busy's fine layer is worth 0.0086 (>> 6 gates * lam) and
plain's is worth 0.0003 (<< that). The gate head has all the signal (ctx) and
now nothing else can absorb the adaptation. lam kept at 6e-4."""
import time, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from model import SparseSplatVAE, l0_cost
from data import VarFaces
torch.manual_seed(0)
tr = VarFaces(n=1024, size=28, seed=0, tex_vary=False)
dl = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
hold = VarFaces(n=512, size=28, seed=1, tex_vary=False)
hl = DataLoader(hold, batch_size=64, shuffle=False)
m = SparseSplatVAE(28,96,(20,40,80),chunk=20)
m.load_state_dict(torch.load("final_fixed_model.pt"))
for p in m.parameters(): p.requires_grad_(False)
for p in m.dec.gate_net.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam(m.dec.gate_net.parameters(), lr=1e-2)
lam=6e-4; N=m.ren.N; hist=[]

def snap(tag):
    m.eval(); rec=0.0; ap=ab=0.0; npl=nb=0; n=0; perp=perb=None
    with torch.no_grad():
        for x,lab in hl:
            out=m(x); rec+=F.mse_loss(out["recon"],x).item()*len(x); n+=len(x)
            tot,pl,_=m.encode_gate_stats(x)
            ap+=tot[lab==0].sum().item(); npl+=(lab==0).sum().item()
            ab+=tot[lab==1].sum().item(); nb+=(lab==1).sum().item()
            p0=pl[lab==0].sum(0); p1=pl[lab==1].sum(0)
            perp=p0 if perp is None else perp+p0
            perb=p1 if perb is None else perb+p1
    mse=rec/n; psnr=float(10*torch.log10(torch.tensor(1/mse)))
    row=dict(tag=tag,psnr=round(psnr,2),plain=round(ap/npl,1),busy=round(ab/nb,1),
             layers_plain=[round(v,1) for v in (perp/npl).tolist()],
             layers_busy=[round(v,1) for v in (perb/nb).tolist()])
    hist.append(row)
    print(f"  [{tag}] PSNR {psnr:.1f} | plain {row['plain']} busy {row['busy']} /{N} | "
          f"Lp {row['layers_plain']} Lb {row['layers_busy']}", flush=True)

t0=time.time(); snap("start")
for ep in range(30):
    m.train()
    for x,_ in dl:
        out=m(x)
        loss=F.mse_loss(out["recon"],x)+lam*l0_cost(out["p_active"])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep%3==2: snap(f"refine ep{ep+1}")
print(f"total {time.time()-t0:.0f}s", flush=True)
torch.save(m.state_dict(),"refined2_model.pt")
json.dump(hist, open("refine2_results.json","w"), indent=1)
