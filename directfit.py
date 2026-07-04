import torch, torch.nn.functional as F, math
from model import MultiScaleRenderer
from data import VarFaces
torch.manual_seed(0)
ds = VarFaces(n=16, size=28, seed=2)
busy = [ds[i][0] for i in range(16) if ds[i][1]==1][:2]
x = torch.stack(busy)
r = MultiScaleRenderer(28,(20,40,80),chunk=20)
raw = torch.zeros(2, r.N, 11, requires_grad=True)
gate = torch.ones(2, r.N)
opt = torch.optim.Adam([raw], lr=0.05)
def band(im):
    f=torch.fft.rfft2(im.mean(1)); H,W=im.shape[-2:]
    fy=torch.fft.fftfreq(H)*H; fx=torch.fft.rfftfreq(W)*W
    rad=(fy[:,None]**2+fx[None,:]**2).sqrt(); msk=(rad>=8.5)&(rad<=13.5)
    return (f.abs()**2*msk).sum(dim=(-2,-1)).mean().item()
for it in range(400):
    rec = r(raw, gate); loss = F.mse_loss(rec, x)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"direct-fit MSE {loss.item():.4f} PSNR {10*math.log10(1/loss.item()):.1f} | "
      f"tex painted {100*band(rec.detach())/band(x):.0f}%")
