#!/usr/bin/env python3
"""Week 6 unpaired CycleGAN baseline for RGB -> thermal scalar prediction.

This is intentionally small and self-contained. It uses the same unified
dataset and synthetic misalignment controls as the Week 2 pix2pix harness, but
trains without paired RGB/thermal supervision by shuffling target-domain samples
inside the training split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import r2t_common as C
from unified_dataset import UnifiedR2TDataset
from week2_pix2pix_baseline import _shuffled_indices, _warp_rgb, seed_all


DEFAULT_TRANSLATION_FRAC = 0.20
DEFAULT_ROTATION_DEG = 20.0
DEFAULT_SCALE_FRAC = 0.25


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


class CycleTrainDataset(Dataset):
    def __init__(self, base: Dataset, args: argparse.Namespace):
        self.base = base
        self.args = args
        self.target_indices = _shuffled_indices(len(base), args.seed + 7013)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, _thermal, _scalar, _tag, _quality = self.base[idx]
        _rgb_b, _thermal_b, target, _tag_b, _quality_b = self.base[self.target_indices[idx]]
        rgb = _warp_rgb(
            rgb,
            sigma=self.args.train_sigma,
            seed=self.args.seed,
            idx=idx,
            stochastic=True,
            max_translation_frac=self.args.max_translation_frac,
            max_rotation_deg=self.args.max_rotation_deg,
            max_scale_frac=self.args.max_scale_frac,
        )
        return rgb, target


class CycleEvalDataset(Dataset):
    def __init__(self, base: Dataset, args: argparse.Namespace):
        self.base = base
        self.args = args

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, _thermal, scalar, _tag, _quality = self.base[idx]
        rgb = _warp_rgb(
            rgb,
            sigma=self.args.eval_sigma,
            seed=self.args.seed + 100000,
            idx=idx,
            stochastic=False,
            max_translation_frac=self.args.max_translation_frac,
            max_rotation_deg=self.args.max_rotation_deg,
            max_scale_frac=self.args.max_scale_frac,
        )
        return rgb, scalar


def maybe_limit(ds: Dataset, limit: int) -> Dataset:
    if limit and limit > 0:
        return Subset(ds, list(range(min(limit, len(ds)))))
    return ds


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 32):
        super().__init__()
        self.d1 = DownBlock(in_ch, base, norm=False)
        self.d2 = DownBlock(base, base * 2)
        self.d3 = DownBlock(base * 2, base * 4)
        self.d4 = DownBlock(base * 4, base * 8)
        self.mid = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.InstanceNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.u4 = UpBlock(base * 8, base * 4)
        self.u3 = UpBlock(base * 8, base * 2)
        self.u2 = UpBlock(base * 4, base)
        self.u1 = nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        y = self.mid(d4)
        y = self.u4(y)
        y = self.u3(torch.cat([y, d3], dim=1))
        y = self.u2(torch.cat([y, d2], dim=1))
        y = self.u1(torch.cat([y, d1], dim=1))
        return torch.sigmoid(y)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            DownBlock(in_ch, base, norm=False),
            DownBlock(base, base * 2),
            DownBlock(base * 2, base * 4),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gan_loss(pred: torch.Tensor, is_real: bool) -> torch.Tensor:
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


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
def evaluate(g_ab: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    g_ab.eval()
    rows = []
    for rgb, target in loader:
        rgb = rgb.to(device)
        pred = g_ab(rgb).detach().cpu().numpy()
        target_np = target.numpy()
        for i in range(pred.shape[0]):
            rows.append(metrics_np(pred[i, 0], target_np[i, 0]))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"cyclegan_{args.dataset}_robust_sigma{args.train_sigma:.1f}_"
        f"seed{args.seed}_e{args.epochs}"
    )


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_base = maybe_limit(make_base_dataset(args, "train"), args.max_train)
    eval_base = maybe_limit(make_base_dataset(args, args.eval_split), args.max_eval)
    train_ds = CycleTrainDataset(train_base, args)
    eval_ds = CycleEvalDataset(eval_base, args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    g_ab = UNetGenerator(3, 1, args.base_channels).to(device)
    g_ba = UNetGenerator(1, 3, args.base_channels).to(device)
    d_a = PatchDiscriminator(3, args.base_channels).to(device)
    d_b = PatchDiscriminator(1, args.base_channels).to(device)

    opt_g = torch.optim.AdamW(
        list(g_ab.parameters()) + list(g_ba.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
        weight_decay=1e-4,
    )
    opt_d = torch.optim.AdamW(
        list(d_a.parameters()) + list(d_b.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
        weight_decay=1e-4,
    )

    out_dir = Path(args.out_dir) / args.dataset / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **vars(args),
        "model": "cyclegan",
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "device": str(device),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    best_psnr = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        g_ab.train()
        g_ba.train()
        d_a.train()
        d_b.train()
        g_losses: list[float] = []
        d_losses: list[float] = []
        cycle_losses: list[float] = []
        for rgb_a, thermal_b in train_loader:
            rgb_a = rgb_a.to(device)
            thermal_b = thermal_b.to(device)

            fake_b = g_ab(rgb_a)
            rec_a = g_ba(fake_b)
            fake_a = g_ba(thermal_b)
            rec_b = g_ab(fake_a)

            opt_g.zero_grad(set_to_none=True)
            loss_gan = gan_loss(d_b(fake_b), True) + gan_loss(d_a(fake_a), True)
            loss_cycle = F.l1_loss(rec_a, rgb_a) + F.l1_loss(rec_b, thermal_b)
            loss_g = loss_gan + args.lambda_cycle * loss_cycle
            loss_g.backward()
            opt_g.step()

            opt_d.zero_grad(set_to_none=True)
            loss_d_a = 0.5 * (gan_loss(d_a(rgb_a), True) + gan_loss(d_a(fake_a.detach()), False))
            loss_d_b = 0.5 * (gan_loss(d_b(thermal_b), True) + gan_loss(d_b(fake_b.detach()), False))
            loss_d = loss_d_a + loss_d_b
            loss_d.backward()
            opt_d.step()

            g_losses.append(float(loss_g.detach().cpu()))
            d_losses.append(float(loss_d.detach().cpu()))
            cycle_losses.append(float(loss_cycle.detach().cpu()))

        metrics = evaluate(g_ab, eval_loader, device)
        row = {
            "epoch": epoch,
            "train_sigma": args.train_sigma,
            "eval_sigma": args.eval_sigma,
            "g_loss": float(np.mean(g_losses)),
            "d_loss": float(np.mean(d_losses)),
            "cycle_loss": float(np.mean(cycle_losses)),
            **metrics,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        append_csv(out_dir / "metrics.csv", row)
        print(json.dumps(row), flush=True)
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            torch.save(
                {
                    "g_ab": g_ab.state_dict(),
                    "g_ba": g_ba.state_dict(),
                    "d_a": d_a.state_dict(),
                    "d_b": d_b.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                },
                out_dir / "best.pt",
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ann_arbor", "kust4k", "caltech_cart"])
    ap.add_argument("--ann-arbor-cache")
    ap.add_argument("--kust4k-root")
    ap.add_argument("--caltech-root")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--target-normalization", default="robust", choices=["raw", "robust", "histmatch"])
    ap.add_argument("--target-normalization-stats")
    ap.add_argument("--train-sigma", type=float, default=0.3)
    ap.add_argument("--eval-sigma", type=float, default=0.3)
    ap.add_argument("--max-translation-frac", type=float, default=DEFAULT_TRANSLATION_FRAC)
    ap.add_argument("--max-rotation-deg", type=float, default=DEFAULT_ROTATION_DEG)
    ap.add_argument("--max-scale-frac", type=float, default=DEFAULT_SCALE_FRAC)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--lambda-cycle", type=float, default=10.0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="week6_runs")
    ap.add_argument("--run-name")
    ap.add_argument("--device")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
