"""Shared utilities for RGB->Thermal experiments: dataset, losses, metrics, palette."""
import os, json, math, random
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F

BASE   = os.environ.get("R2T_BASE", "/home/spant/UMich/umich-hackathon")
CACHE  = os.environ.get("R2T_CACHE", f"{BASE}/rgb2thermal/data_cache")
RES_H, RES_W = 512, 640

def load_split():
    return json.load(open(f"{CACHE}/split.json"))
def load_lut():
    return np.load(f"{CACHE}/lut.npy")   # (256,3) float

def to_color(scalar01):  # scalar HxW [0,1] -> RGB uint8 via recovered palette
    lut=load_lut(); idx=np.clip((np.asarray(scalar01)*255).astype(int),0,255)
    return lut[idx].astype(np.uint8)

# --------------------------------------------------------------------- dataset
class R2TDataset(torch.utils.data.Dataset):
    def __init__(self, names, augment=False, use_depth=True, unreg=False, solar=False):
        self.names=[n.replace('.JPG','') for n in names]
        self.augment=augment; self.use_depth=use_depth; self.unreg=unreg; self.solar=solar
        self.solarmap=json.load(open(f"{CACHE}/solar.json")) if solar else {}
    def __len__(self): return len(self.names)
    def _load(self,n):
        if self.unreg:  # ablation: unregistered full RGB (no crop), SAME scalar target
            rgb=np.asarray(Image.open(f"{BASE}/data/Train_2/RGB/{n}.JPG").convert("RGB").resize((RES_W,RES_H))).astype(np.float32)/255.0
        else:
            rgb=np.asarray(Image.open(f"{CACHE}/reg_rgb/{n}.png").convert("RGB")).astype(np.float32)/255.0
        tgt=np.load(f"{CACHE}/scalar/{n}.npy").astype(np.float32)
        dp_p=f"{CACHE}/depth/{n}.npy"
        dep=np.load(dp_p).astype(np.float32) if os.path.exists(dp_p) else np.zeros((RES_H,RES_W),np.float32)
        al_p=f"{CACHE}/alpha/{n}.npy"
        if os.path.exists(al_p):
            a=np.load(al_p).astype(np.float32); a=np.nan_to_num(a); alpha=a.mean(axis=(1,2))
        else: alpha=np.zeros(64,np.float32)
        return rgb,dep,tgt,alpha
    def __getitem__(self,i):
        n=self.names[i]; rgb,dep,tgt,alpha=self._load(n)
        if self.augment:
            if random.random()<0.5: rgb=rgb[:, ::-1].copy(); dep=dep[:, ::-1].copy(); tgt=tgt[:, ::-1].copy()
            if random.random()<0.5: rgb=rgb[::-1].copy(); dep=dep[::-1].copy(); tgt=tgt[::-1].copy()
            if random.random()<0.7:  # photometric on RGB only
                b=1+ (random.random()-0.5)*0.4; c=1+(random.random()-0.5)*0.4
                m=rgb.mean(); rgb=np.clip((rgb-m)*c+m*b,0,1)
        rgb_t=torch.from_numpy(rgb.transpose(2,0,1))
        dep_t=torch.from_numpy(dep[None])
        tgt_t=torch.from_numpy(tgt[None])
        # normalize rgb with imagenet stats for pretrained encoders
        mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        rgb_n=(rgb_t-mean)/std
        if self.solar:
            el,az=self.solarmap.get(n,[63.0,195.0])
            sol=torch.tensor([el/90.0,az/360.0],dtype=torch.float32).view(2,1,1).expand(2,rgb_n.shape[1],rgb_n.shape[2])
            rgb_n=torch.cat([rgb_n,sol],0)
        return dict(rgb=rgb_n, rgb_raw=rgb_t, depth=dep_t, alpha=torch.from_numpy(alpha),
                    target=tgt_t, name=n)

def make_input(batch, use_depth):
    x=batch['rgb']
    if use_depth: x=torch.cat([x, batch['depth']],1)
    return x

# --------------------------------------------------------------------- ssim / losses
def _gauss(win=11,sigma=1.5):
    c=torch.arange(win).float()-win//2
    g=torch.exp(-(c**2)/(2*sigma**2)); g=(g/g.sum())
    k=g[:,None]@g[None,:]; return k
_K=_gauss()
def ssim(x,y):
    k=_K.to(x.device).view(1,1,11,11).expand(x.shape[1],1,11,11)
    mx=F.conv2d(x,k,padding=5,groups=x.shape[1]); my=F.conv2d(y,k,padding=5,groups=x.shape[1])
    mx2,my2,mxy=mx*mx,my*my,mx*my
    sx=F.conv2d(x*x,k,padding=5,groups=x.shape[1])-mx2
    sy=F.conv2d(y*y,k,padding=5,groups=x.shape[1])-my2
    sxy=F.conv2d(x*y,k,padding=5,groups=x.shape[1])-mxy
    C1,C2=0.01**2,0.03**2
    s=((2*mxy+C1)*(2*sxy+C2))/((mx2+my2+C1)*(sx+sy+C2))
    return s.mean()
def grad_loss(x,y):
    dx=lambda z: z[...,:,1:]-z[...,:,:-1]; dy=lambda z: z[...,1:,:]-z[...,:-1,:]
    return (dx(x)-dx(y)).abs().mean()+(dy(x)-dy(y)).abs().mean()
def combined_loss(pred,tgt):
    return F.l1_loss(pred,tgt) + 0.5*(1-ssim(pred,tgt)) + 0.5*grad_loss(pred,tgt)

# --------------------------------------------------------------------- metrics
@torch.no_grad()
def metrics_np(pred, tgt):
    pred=np.clip(pred,0,1); tgt=np.clip(tgt,0,1)
    mae=float(np.abs(pred-tgt).mean()); rmse=float(np.sqrt(((pred-tgt)**2).mean()))
    mse=((pred-tgt)**2).mean(); psnr=float(10*np.log10(1.0/(mse+1e-10)))
    pf,tf=pred.ravel(),tgt.ravel()
    r=float(np.corrcoef(pf,tf)[0,1]) if pf.std()>1e-6 and tf.std()>1e-6 else 0.0
    return dict(mae=mae,rmse=rmse,psnr=psnr,corr=r)
@torch.no_grad()
def ssim_np(pred,tgt,device='cpu'):
    p=torch.from_numpy(pred[None,None].astype(np.float32)).to(device)
    t=torch.from_numpy(tgt[None,None].astype(np.float32)).to(device)
    return float(ssim(p,t))
