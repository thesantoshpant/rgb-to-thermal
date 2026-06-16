#!/usr/bin/env python3
"""Week 6 pretrained Swin encoder + U-Net decoder baseline."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import r2t_common as C
from unified_dataset import UnifiedR2TDataset
from week2_pix2pix_baseline import _warp_rgb, seed_all


def append_csv(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_base_dataset(args: argparse.Namespace, split: str) -> Dataset:
    if args.dataset == "ann_arbor":
        ds = UnifiedR2TDataset.from_roots(
            ann_arbor_cache=args.ann_arbor_cache,
            split=split,
            size_hw=(args.height, args.width),
            target_norm=args.target_normalization,
            target_norm_stats=args.target_normalization_stats,
        )
    elif args.dataset == "kust4k":
        ds = UnifiedR2TDataset.from_roots(
            kust4k_root=args.kust4k_root,
            split=split,
            size_hw=(args.height, args.width),
            target_norm=args.target_normalization,
            target_norm_stats=args.target_normalization_stats,
        )
    elif args.dataset == "caltech_cart":
        ds = UnifiedR2TDataset.from_roots(
            caltech_root=args.caltech_root,
            split=split,
            size_hw=(args.height, args.width),
            target_norm=args.target_normalization,
            target_norm_stats=args.target_normalization_stats,
        )
    else:
        raise ValueError(args.dataset)
    if len(ds) == 0:
        raise RuntimeError(f"No records found for dataset={args.dataset} split={split}")
    return ds


def maybe_limit(ds: Dataset, limit: int) -> Dataset:
    if limit and limit > 0:
        return Subset(ds, list(range(min(limit, len(ds)))))
    return ds


class PairedMisalignmentDataset(Dataset):
    def __init__(self, base: Dataset, args: argparse.Namespace, train: bool):
        self.base = base
        self.args = args
        self.train = train

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, _thermal, scalar, _tag, _quality = self.base[idx]
        sigma = self.args.train_sigma if self.train else self.args.eval_sigma
        seed = self.args.seed if self.train else self.args.seed + 100000
        rgb = _warp_rgb(
            rgb,
            sigma=sigma,
            seed=seed,
            idx=idx,
            stochastic=self.train,
            max_translation_frac=self.args.max_translation_frac,
            max_rotation_deg=self.args.max_rotation_deg,
            max_scale_frac=self.args.max_scale_frac,
        )
        return rgb, scalar


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwinUNet(nn.Module):
    def __init__(self, encoder: str, height: int, width: int, pretrained: bool = True):
        super().__init__()
        kwargs = {
            "pretrained": pretrained,
            "features_only": True,
            "in_chans": 3,
            "img_size": (height, width),
        }
        try:
            self.enc = timm.create_model(encoder, **kwargs)
        except TypeError:
            kwargs.pop("img_size")
            self.enc = timm.create_model(encoder, **kwargs)
        chs = self.enc.feature_info.channels()
        if len(chs) < 4:
            raise ValueError(f"Expected at least four feature levels from {encoder}, got {chs}")
        chs = chs[-4:]
        self.up3 = ConvBlock(chs[3] + chs[2], chs[2])
        self.up2 = ConvBlock(chs[2] + chs[1], chs[1])
        self.up1 = ConvBlock(chs[1] + chs[0], chs[0])
        self.u0 = ConvBlock(chs[0], 64)
        self.u1 = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def _up(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self._to_nchw(feat) for feat in self.enc(normalize_rgb(x))[-4:]]
        f1, f2, f3, f4 = feats
        d3 = self.up3(torch.cat([self._up(f4, f3), f3], dim=1))
        d2 = self.up2(torch.cat([self._up(d3, f2), f2], dim=1))
        d1 = self.up1(torch.cat([self._up(d2, f1), f1], dim=1))
        u0 = self.u0(F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False))
        u1 = self.u1(F.interpolate(u0, scale_factor=2, mode="bilinear", align_corners=False))
        out = self.head(u1)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(out)

    @staticmethod
    def _to_nchw(feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 4 and feat.shape[1] < feat.shape[-1]:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat


def normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb - mean) / std


def metrics_np(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    mse = float(np.mean((pred - target) ** 2))
    pf = pred.ravel()
    tf = target.ravel()
    corr = float(np.corrcoef(pf, tf)[0, 1]) if pf.std() > 1e-6 and tf.std() > 1e-6 else 0.0
    return {
        "mae": float(np.mean(np.abs(pred - target))),
        "rmse": float(math.sqrt(mse)),
        "psnr": float(10.0 * math.log10(1.0 / (mse + 1e-10))),
        "corr": corr,
        "ssim": C.ssim_np(pred, target),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    for rgb, target in loader:
        rgb = rgb.to(device)
        pred = model(rgb).detach().cpu().numpy()
        target_np = target.numpy()
        for i in range(pred.shape[0]):
            rows.append(metrics_np(pred[i, 0], target_np[i, 0]))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return f"swin_unet_{args.dataset}_robust_sigma{args.train_sigma:.1f}_seed{args.seed}_e{args.epochs}"


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_base = maybe_limit(make_base_dataset(args, "train"), args.max_train)
    eval_base = maybe_limit(make_base_dataset(args, args.eval_split), args.max_eval)
    train_ds = PairedMisalignmentDataset(train_base, args, train=True)
    eval_ds = PairedMisalignmentDataset(eval_base, args, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = SwinUNet(args.encoder, args.height, args.width, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out_dir) / args.dataset / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **vars(args),
        "model": "swin_unet",
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "device": str(device),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    best_psnr = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for rgb, target in train_loader:
            rgb = rgb.to(device)
            target = target.to(device)
            pred = model(rgb)
            loss = C.combined_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        metrics = evaluate(model, eval_loader, device)
        row = {
            "epoch": epoch,
            "train_sigma": args.train_sigma,
            "eval_sigma": args.eval_sigma,
            "loss": float(np.mean(losses)),
            **metrics,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        append_csv(out_dir / "metrics.csv", row)
        print(json.dumps(row), flush=True)
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ann_arbor", "kust4k", "caltech_cart"])
    ap.add_argument("--ann-arbor-cache")
    ap.add_argument("--kust4k-root")
    ap.add_argument("--caltech-root")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--encoder", default="swin_tiny_patch4_window7_224.ms_in1k")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--target-normalization", default="robust", choices=["raw", "robust", "histmatch"])
    ap.add_argument("--target-normalization-stats")
    ap.add_argument("--train-sigma", type=float, default=0.3)
    ap.add_argument("--eval-sigma", type=float, default=0.3)
    ap.add_argument("--max-translation-frac", type=float, default=0.20)
    ap.add_argument("--max-rotation-deg", type=float, default=20.0)
    ap.add_argument("--max-scale-frac", type=float, default=0.25)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="week6_runs")
    ap.add_argument("--run-name")
    ap.add_argument("--device")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
