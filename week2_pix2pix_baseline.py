#!/usr/bin/env python3
"""Week 2 go/no-go harness: small pix2pix baseline with synthetic misalignment.

Example smoke test on Knox:
  python week2_pix2pix_baseline.py \
    --dataset kust4k \
    --kust4k-root data_cache/external/kust4k \
    --epochs 1 --max-train 16 --max-eval 8 --height 128 --width 160
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
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from unified_dataset import UnifiedR2TDataset


DEFAULT_TRANSLATION_FRAC = 0.08
DEFAULT_ROTATION_DEG = 8.0
DEFAULT_SCALE_FRAC = 0.15


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warp_rgb(
    rgb: torch.Tensor,
    sigma: float,
    seed: int,
    idx: int,
    stochastic: bool,
    max_translation_frac: float,
    max_rotation_deg: float,
    max_scale_frac: float,
) -> torch.Tensor:
    sigma = float(np.clip(sigma, 0.0, 1.0))
    if sigma <= 0:
        return rgb
    salt = random.randint(0, 2**16 - 1) if stochastic else 0
    rng = random.Random(seed + idx * 1009 + salt)
    h, w = rgb.shape[-2:]
    span = min(h, w)
    translate = [
        int(round(rng.uniform(-1.0, 1.0) * max_translation_frac * sigma * span)),
        int(round(rng.uniform(-1.0, 1.0) * max_translation_frac * sigma * span)),
    ]
    angle = rng.uniform(-1.0, 1.0) * max_rotation_deg * sigma
    scale = 1.0 + rng.uniform(-1.0, 1.0) * max_scale_frac * sigma
    return TF.affine(
        rgb,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )


def _shuffled_indices(n: int, seed: int) -> list[int]:
    order = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(order)
    if n > 1:
        for i, j in enumerate(order):
            if i == j:
                order[i], order[(i + 1) % n] = order[(i + 1) % n], order[i]
    return order


class DiagnosticDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        sigma: float,
        train: bool,
        seed: int,
        shuffle_rgb: bool,
        max_translation_frac: float,
        max_rotation_deg: float,
        max_scale_frac: float,
    ):
        self.base = base
        self.sigma = float(np.clip(sigma, 0.0, 1.0))
        self.train = train
        self.seed = seed
        self.shuffle_rgb = shuffle_rgb
        self.max_translation_frac = max_translation_frac
        self.max_rotation_deg = max_rotation_deg
        self.max_scale_frac = max_scale_frac
        self.rgb_indices = _shuffled_indices(len(base), seed + 4242) if shuffle_rgb else None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        rgb, thermal, scalar, tag, quality = self.base[idx]
        if self.rgb_indices is not None:
            rgb_idx = self.rgb_indices[idx]
            rgb, _thermal, _scalar, _tag, _quality = self.base[rgb_idx]
        rgb = _warp_rgb(
            rgb,
            sigma=self.sigma,
            seed=self.seed,
            idx=idx,
            stochastic=self.train,
            max_translation_frac=self.max_translation_frac,
            max_rotation_deg=self.max_rotation_deg,
            max_scale_frac=self.max_scale_frac,
        )
        return rgb, scalar, tag, quality


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Pix2PixGenerator(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.d1 = ConvBlock(3, base, norm=False)
        self.d2 = ConvBlock(base, base * 2)
        self.d3 = ConvBlock(base * 2, base * 4)
        self.d4 = ConvBlock(base * 4, base * 8)
        self.mid = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.u4 = UpBlock(base * 8, base * 4)
        self.u3 = UpBlock(base * 8, base * 2)
        self.u2 = UpBlock(base * 4, base)
        self.u1 = nn.ConvTranspose2d(base * 2, 1, 4, 2, 1)

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
    def __init__(self, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(4, base, norm=False),
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


def metrics_np(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    mse = float(np.mean((pred - target) ** 2))
    return {
        "mae": float(np.mean(np.abs(pred - target))),
        "rmse": float(math.sqrt(mse)),
        "psnr": float(10.0 * math.log10(1.0 / (mse + 1e-10))),
    }


def make_dataset(args: argparse.Namespace, split: str, train: bool) -> Dataset:
    if args.dataset == "kust4k":
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
    elif args.dataset == "ann_arbor":
        ds = UnifiedR2TDataset.from_roots(
            ann_arbor_cache=args.ann_arbor_cache,
            split=split,
            size_hw=(args.height, args.width),
            target_norm=args.target_normalization,
            target_norm_stats=args.target_normalization_stats,
        )
    else:
        raise ValueError(args.dataset)
    limit = args.max_train if train else args.max_eval
    if limit and limit > 0:
        ds = Subset(ds, list(range(min(limit, len(ds)))))
    sigma = args.train_sigma if train else args.eval_sigma
    return DiagnosticDataset(
        ds,
        sigma=sigma,
        train=train,
        seed=args.seed + (0 if train else 100000),
        shuffle_rgb=args.shuffle_rgb,
        max_translation_frac=args.max_translation_frac,
        max_rotation_deg=args.max_rotation_deg,
        max_scale_frac=args.max_scale_frac,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    for rgb, target, _tag, _quality in loader:
        rgb = rgb.to(device)
        target = target.to(device)
        pred = model(rgb).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        for i in range(pred.shape[0]):
            rows.append(metrics_np(pred[i, 0], target_np[i, 0]))
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


def append_csv(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _legacy_run_name(args: argparse.Namespace) -> bool:
    return (
        args.run_name is None
        and args.model == "pix2pix"
        and not args.eval_only
        and not args.shuffle_rgb
        and args.eval_sigma == 0.0
        and args.train_sigma == args.sigma
        and args.max_translation_frac == DEFAULT_TRANSLATION_FRAC
        and args.max_rotation_deg == DEFAULT_ROTATION_DEG
        and args.max_scale_frac == DEFAULT_SCALE_FRAC
    )


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if _legacy_run_name(args):
        return f"sigma_{args.train_sigma:.2f}"
    pieces = [
        args.model,
        f"train_{args.train_sigma:.2f}",
        f"eval_{args.eval_sigma:.2f}",
        f"trans_{args.max_translation_frac:.2f}",
        f"rot_{args.max_rotation_deg:.1f}",
    ]
    if args.shuffle_rgb:
        pieces.append("shuffle_rgb")
    if args.eval_only:
        pieces.append("eval_only")
    pieces.append(f"seed_{args.seed}")
    return "_".join(pieces)


def load_generator(path: str | None, model: nn.Module, device: torch.device) -> None:
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("generator", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)


def metadata_for(args: argparse.Namespace, train_size: int, eval_size: int) -> dict[str, float | int | str | bool]:
    return {
        "dataset": args.dataset,
        "model": args.model,
        "sigma": args.train_sigma,
        "train_sigma": args.train_sigma,
        "eval_sigma": args.eval_sigma,
        "train_size": train_size,
        "eval_size": eval_size,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "shuffle_rgb": args.shuffle_rgb,
        "max_translation_frac": args.max_translation_frac,
        "max_rotation_deg": args.max_rotation_deg,
        "max_scale_frac": args.max_scale_frac,
        "checkpoint": args.checkpoint or "",
    }


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir) / args.dataset / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_split = "val" if args.eval_split == "val" else "test"
    eval_ds = make_dataset(args, eval_split, train=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    gen = Pix2PixGenerator(args.base_channels).to(device)
    load_generator(args.checkpoint, gen, device)

    if args.eval_only:
        metadata = metadata_for(args, train_size=0, eval_size=len(eval_ds))
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
        metrics = evaluate(gen, eval_loader, device)
        row = {
            "epoch": 0,
            "train_sigma": args.train_sigma,
            "eval_sigma": args.eval_sigma,
            "g_loss": 0.0,
            "d_loss": 0.0,
            **metrics,
            "elapsed_sec": 0.0,
        }
        append_csv(out_dir / "metrics.csv", row)
        print(json.dumps(row), flush=True)
        return

    train_ds = make_dataset(args, "train", train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    disc = PatchDiscriminator(args.base_channels).to(device) if args.model == "pix2pix" else None
    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.5, 0.999)) if disc is not None else None

    metadata = metadata_for(args, train_size=len(train_ds), eval_size=len(eval_ds))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    best_psnr = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        gen.train()
        if disc is not None:
            disc.train()
        g_losses = []
        d_losses = []
        for rgb, target, _tag, _quality in train_loader:
            rgb = rgb.to(device)
            target = target.to(device)

            fake = gen(rgb)
            if disc is not None and opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
                pred_real = disc(rgb, target)
                pred_fake = disc(rgb, fake.detach())
                loss_d = 0.5 * (
                    F.mse_loss(pred_real, torch.ones_like(pred_real))
                    + F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
                )
                loss_d.backward()
                opt_d.step()
            else:
                loss_d = torch.zeros((), device=device)

            opt_g.zero_grad(set_to_none=True)
            if disc is not None:
                pred_fake_g = disc(rgb, fake)
                loss_g = F.mse_loss(pred_fake_g, torch.ones_like(pred_fake_g)) + args.l1_weight * F.l1_loss(fake, target)
            else:
                loss_g = F.l1_loss(fake, target)
            loss_g.backward()
            opt_g.step()

            g_losses.append(float(loss_g.detach().cpu()))
            d_losses.append(float(loss_d.detach().cpu()))

        metrics = evaluate(gen, eval_loader, device)
        row = {
            "epoch": epoch,
            "sigma": args.train_sigma,
            "train_sigma": args.train_sigma,
            "eval_sigma": args.eval_sigma,
            "g_loss": float(np.mean(g_losses)),
            "d_loss": float(np.mean(d_losses)),
            **metrics,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        append_csv(out_dir / "metrics.csv", row)
        print(json.dumps(row), flush=True)
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            checkpoint = {
                "generator": gen.state_dict(),
                "args": vars(args),
                "metrics": metrics,
            }
            if disc is not None:
                checkpoint["discriminator"] = disc.state_dict()
            torch.save(checkpoint, out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kust4k", "caltech_cart", "ann_arbor"])
    ap.add_argument("--ann-arbor-cache")
    ap.add_argument("--caltech-root")
    ap.add_argument("--kust4k-root")
    ap.add_argument("--model", default="pix2pix", choices=["pix2pix", "l1"])
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--train-sigma", type=float, default=None)
    ap.add_argument("--eval-sigma", type=float, default=0.0)
    ap.add_argument("--max-translation-frac", type=float, default=DEFAULT_TRANSLATION_FRAC)
    ap.add_argument("--max-rotation-deg", type=float, default=DEFAULT_ROTATION_DEG)
    ap.add_argument("--max-scale-frac", type=float, default=DEFAULT_SCALE_FRAC)
    ap.add_argument("--shuffle-rgb", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--run-name")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--target-normalization", default="raw", choices=["raw", "robust", "histmatch"])
    ap.add_argument("--target-normalization-stats")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1-weight", type=float, default=50.0)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--out-dir", default="week2_runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.train_sigma = args.sigma if args.train_sigma is None else args.train_sigma
    if args.eval_only and not args.checkpoint:
        raise SystemExit("--eval-only requires --checkpoint")
    return args


if __name__ == "__main__":
    train(parse_args())
