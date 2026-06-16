#!/usr/bin/env python3
"""Week 3 learned registration v0.

This is a sanity-check trainer for the registration-bottleneck story. It trains
on Ann Arbor first: synthetic RGB misalignment is applied while the scalar
thermal target remains aligned, then a small registration head predicts an
affine correction and uncertainty map before the existing ConvNeXt+U-Net
translator predicts scalar thermal.
"""
from __future__ import annotations

import argparse
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
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

import r2t_common as C
from train_a1 import ConvBlock, UNetReg
from unified_dataset import UnifiedR2TDataset


DEFAULT_TRANSLATION_FRAC = 0.20
DEFAULT_ROTATION_DEG = 20.0
DEFAULT_SCALE_FRAC = 0.25
DEFAULT_MAX_FLOW = 0.15
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rgb(rgb_raw: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(rgb_raw.device, rgb_raw.dtype)
    std = IMAGENET_STD.to(rgb_raw.device, rgb_raw.dtype)
    return (rgb_raw - mean) / std


def warp_rgb(
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


class MisalignedAnnArbor(Dataset):
    def __init__(
        self,
        names: list[str],
        sigma: float,
        augment: bool,
        seed: int,
        max_translation_frac: float,
        max_rotation_deg: float,
        max_scale_frac: float,
    ):
        self.base = C.R2TDataset(names, augment=augment, use_depth=False)
        self.sigma = sigma
        self.augment = augment
        self.seed = seed
        self.max_translation_frac = max_translation_frac
        self.max_rotation_deg = max_rotation_deg
        self.max_scale_frac = max_scale_frac

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.base[idx]
        rgb_raw = row["rgb_raw"]
        misaligned = warp_rgb(
            rgb_raw,
            sigma=self.sigma,
            seed=self.seed,
            idx=idx,
            stochastic=self.augment,
            max_translation_frac=self.max_translation_frac,
            max_rotation_deg=self.max_rotation_deg,
            max_scale_frac=self.max_scale_frac,
        )
        row["rgb_input_raw"] = misaligned
        row["rgb_input"] = normalize_rgb(misaligned.unsqueeze(0))[0]
        return row


class MisalignedUnifiedDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        sigma: float,
        augment: bool,
        seed: int,
        max_translation_frac: float,
        max_rotation_deg: float,
        max_scale_frac: float,
    ):
        self.base = base
        self.sigma = sigma
        self.augment = augment
        self.seed = seed
        self.max_translation_frac = max_translation_frac
        self.max_rotation_deg = max_rotation_deg
        self.max_scale_frac = max_scale_frac

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rgb_raw, _thermal, scalar, dataset_tag, quality = self.base[idx]
        misaligned = warp_rgb(
            rgb_raw,
            sigma=self.sigma,
            seed=self.seed,
            idx=idx,
            stochastic=self.augment,
            max_translation_frac=self.max_translation_frac,
            max_rotation_deg=self.max_rotation_deg,
            max_scale_frac=self.max_scale_frac,
        )
        return {
            "rgb_raw": rgb_raw,
            "rgb_input_raw": misaligned,
            "rgb_input": normalize_rgb(misaligned.unsqueeze(0))[0],
            "target": scalar,
            "dataset_tag": dataset_tag,
            "alignment_quality": quality,
        }


def make_external_base_dataset(
    args: argparse.Namespace,
    split: str,
    limit: int,
    dataset: str | None = None,
) -> Dataset:
    dataset = dataset or args.dataset
    if dataset == "kust4k":
        ds = UnifiedR2TDataset.from_roots(
            kust4k_root=args.kust4k_root,
            split=split,
            size_hw=(C.RES_H, C.RES_W),
            target_norm=args.target_normalization,
            target_norm_stats=args.target_normalization_stats,
        )
    elif dataset == "caltech_cart":
        ds = UnifiedR2TDataset.from_roots(
            caltech_root=args.caltech_root,
            split=split,
            size_hw=(C.RES_H, C.RES_W),
            target_norm=args.target_normalization,
            target_norm_stats=args.target_normalization_stats,
        )
    else:
        raise ValueError(dataset)
    if len(ds) == 0:
        raise RuntimeError(f"No records found for dataset={dataset} split={split}")
    if limit and limit > 0:
        return Subset(ds, list(range(min(limit, len(ds)))))
    return ds


def make_registration_dataset(
    args: argparse.Namespace,
    split_name: str,
    sigma: float,
    augment: bool,
    seed: int,
    limit: int,
    dataset: str | None = None,
) -> Dataset:
    dataset = dataset or args.dataset
    if dataset == "ann_arbor":
        split = C.load_split()
        names = split[split_name][: limit or None]
        return MisalignedAnnArbor(
            names,
            sigma=sigma,
            augment=augment,
            seed=seed,
            max_translation_frac=args.max_translation_frac,
            max_rotation_deg=args.max_rotation_deg,
            max_scale_frac=args.max_scale_frac,
        )
    base = make_external_base_dataset(args, split_name, limit, dataset=dataset)
    return MisalignedUnifiedDataset(
        base,
        sigma=sigma,
        augment=augment,
        seed=seed,
        max_translation_frac=args.max_translation_frac,
        max_rotation_deg=args.max_rotation_deg,
        max_scale_frac=args.max_scale_frac,
    )


def resize_registration_batch(batch: dict[str, torch.Tensor], res: int) -> dict[str, torch.Tensor]:
    if res >= C.RES_H:
        return batch
    h = res
    w = int(round(res * C.RES_W / C.RES_H / 2) * 2)
    for key in ("target", "rgb_input_raw", "rgb_input", "rgb_raw", "rgb", "depth"):
        if key in batch and torch.is_tensor(batch[key]):
            batch[key] = F.interpolate(batch[key], size=(h, w), mode="bilinear", align_corners=False)
    return batch


class RegistrationHead(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, base, 5, stride=2, padding=2),
            nn.GroupNorm(8, base),
            nn.GELU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.GELU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.theta = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 4, base * 4),
            nn.GELU(),
            nn.Linear(base * 4, 6),
        )
        self.unc = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base * 2, 1, 1),
        )
        nn.init.zeros_(self.theta[-1].weight)
        with torch.no_grad():
            self.theta[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, rgb_raw: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([rgb_raw, target], dim=1)
        feat = self.enc(x)
        theta = self.theta(self.pool(feat)).view(-1, 2, 3)
        uncertainty = F.softplus(F.interpolate(self.unc(feat), size=target.shape[-2:], mode="bilinear", align_corners=False))
        return theta, uncertainty


class RegistrationTranslator(nn.Module):
    def __init__(self, encoder: str, reg_base: int = 32):
        super().__init__()
        self.reg = RegistrationHead(reg_base)
        self.translator = UNetReg(encoder=encoder, in_ch=3, use_alpha=False)

    def forward(self, rgb_raw: torch.Tensor, target_for_reg: torch.Tensor) -> dict[str, torch.Tensor]:
        theta, uncertainty = self.reg(rgb_raw, target_for_reg)
        warped_raw = apply_affine(rgb_raw, theta)
        pred = self.translator(normalize_rgb(warped_raw))
        return {
            "pred": pred,
            "theta": theta,
            "uncertainty": uncertainty,
            "warped_raw": warped_raw,
        }


class SharedFeatureRegistrationTranslator(nn.Module):
    """Deployable RGB-only variant: one encoder feeds registration and decoding."""

    def __init__(self, encoder: str):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=True, features_only=True, in_chans=3)
        chs = self.enc.feature_info.channels()
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chs[-1], 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )
        nn.init.zeros_(self.theta[-1].weight)
        with torch.no_grad():
            self.theta[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.unc = nn.Sequential(
            nn.Conv2d(chs[0], 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        self.up3 = ConvBlock(chs[3] + chs[2], chs[2])
        self.up2 = ConvBlock(chs[2] + chs[1], chs[1])
        self.up1 = ConvBlock(chs[1] + chs[0], chs[0])
        self.u0 = ConvBlock(chs[0], 64)
        self.u_1 = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def _up(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def _up2(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, rgb_raw: torch.Tensor, target_for_reg: torch.Tensor) -> dict[str, torch.Tensor]:
        f1, f2, f3, f4 = self.enc(normalize_rgb(rgb_raw))
        theta = self.theta(f4).view(-1, 2, 3)
        wf1, wf2, wf3, wf4 = [apply_affine(feat, theta) for feat in (f1, f2, f3, f4)]
        d3 = self.up3(torch.cat([self._up(wf4, wf3), wf3], 1))
        d2 = self.up2(torch.cat([self._up(d3, wf2), wf2], 1))
        d1 = self.up1(torch.cat([self._up(d2, wf1), wf1], 1))
        u0 = self.u0(self._up2(d1))
        u1 = self.u_1(self._up2(u0))
        pred = torch.sigmoid(self.head(u1))
        uncertainty = F.softplus(F.interpolate(self.unc(wf1), size=target_for_reg.shape[-2:], mode="bilinear", align_corners=False))
        return {
            "pred": pred,
            "theta": theta,
            "uncertainty": uncertainty,
            "warped_raw": apply_affine(rgb_raw, theta),
        }


class NoRegistrationTranslator(nn.Module):
    """ConvNeXt+U-Net baseline on misaligned RGB with no learned warp."""

    def __init__(self, encoder: str):
        super().__init__()
        self.translator = UNetReg(encoder=encoder, in_ch=3, use_alpha=False)

    def forward(self, rgb_raw: torch.Tensor, target_for_reg: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.translator(normalize_rgb(rgb_raw))
        b = rgb_raw.shape[0]
        theta = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=rgb_raw.dtype,
            device=rgb_raw.device,
        ).view(1, 2, 3).expand(b, -1, -1)
        uncertainty = torch.zeros(
            b,
            1,
            target_for_reg.shape[-2],
            target_for_reg.shape[-1],
            dtype=rgb_raw.dtype,
            device=rgb_raw.device,
        )
        return {
            "pred": pred,
            "theta": theta,
            "uncertainty": uncertainty,
            "warped_raw": rgb_raw,
        }


class RGBInputAffineRegistrationTranslator(nn.Module):
    """Deployable RGB-only affine predictor that warps input pixels."""

    def __init__(self, encoder: str, reg_base: int = 32):
        super().__init__()
        self.reg_enc = nn.Sequential(
            nn.Conv2d(3, reg_base, 5, stride=2, padding=2),
            nn.GroupNorm(8, reg_base),
            nn.GELU(),
            nn.Conv2d(reg_base, reg_base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, reg_base * 2),
            nn.GELU(),
            nn.Conv2d(reg_base * 2, reg_base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, reg_base * 4),
            nn.GELU(),
            nn.Conv2d(reg_base * 4, reg_base * 4, 3, padding=1),
            nn.GroupNorm(8, reg_base * 4),
            nn.GELU(),
        )
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(reg_base * 4, reg_base * 4),
            nn.GELU(),
            nn.Linear(reg_base * 4, 6),
        )
        self.unc = nn.Sequential(
            nn.Conv2d(reg_base * 4, reg_base * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(reg_base * 2, 1, 1),
        )
        nn.init.zeros_(self.theta[-1].weight)
        with torch.no_grad():
            self.theta[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.translator = UNetReg(encoder=encoder, in_ch=3, use_alpha=False)

    def forward(self, rgb_raw: torch.Tensor, target_for_reg: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.reg_enc(rgb_raw)
        theta = self.theta(feat).view(-1, 2, 3)
        warped_raw = apply_affine(rgb_raw, theta)
        pred = self.translator(normalize_rgb(warped_raw))
        uncertainty = F.softplus(F.interpolate(self.unc(feat), size=target_for_reg.shape[-2:], mode="bilinear", align_corners=False))
        return {
            "pred": pred,
            "theta": theta,
            "uncertainty": uncertainty,
            "warped_raw": warped_raw,
        }


class RGBDenseFlowRegistrationTranslator(nn.Module):
    """Deployable RGB-only dense-flow predictor for small non-rigid correction."""

    def __init__(self, encoder: str, max_flow: float = DEFAULT_MAX_FLOW, base: int = 32):
        super().__init__()
        self.max_flow = max_flow
        self.e1 = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.GELU(),
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.GELU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.GELU(),
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(base * 6, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
        )
        self.d1 = nn.Sequential(
            nn.Conv2d(base * 3, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.GELU(),
        )
        self.flow_head = nn.Conv2d(base, 2, 3, padding=1)
        self.unc_head = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        self.translator = UNetReg(encoder=encoder, in_ch=3, use_alpha=False)

    def forward(self, rgb_raw: torch.Tensor, target_for_reg: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.e1(rgb_raw)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        flow = torch.tanh(self.flow_head(d1)) * self.max_flow
        warped_raw = apply_flow(rgb_raw, flow)
        pred = self.translator(normalize_rgb(warped_raw))
        uncertainty = F.softplus(F.interpolate(self.unc_head(d1), size=target_for_reg.shape[-2:], mode="bilinear", align_corners=False))
        b = rgb_raw.shape[0]
        theta = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=rgb_raw.dtype,
            device=rgb_raw.device,
        ).view(1, 2, 3).expand(b, -1, -1)
        return {
            "pred": pred,
            "theta": theta,
            "uncertainty": uncertainty,
            "warped_raw": warped_raw,
            "flow": flow,
        }


def apply_affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_flow(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _c, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, dtype=x.dtype, device=x.device),
        torch.linspace(-1.0, 1.0, w, dtype=x.dtype, device=x.device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, h, w, 2)
    grid = base + flow.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    sy = sx.transpose(2, 3)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + 1e-6)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def affine_identity_loss(theta: torch.Tensor) -> torch.Tensor:
    ident = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=theta.dtype, device=theta.device)
    return (theta - ident.view(1, 2, 3)).square().mean()


def flow_regularization(flow: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if flow is None:
        zero = torch.tensor(0.0)
        return zero, zero
    mag = flow.square().mean()
    smooth = tv_loss(flow)
    return mag, smooth


def loss_terms(
    out: dict[str, torch.Tensor],
    target: torch.Tensor,
    aligned_rgb: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    pred = out["pred"]
    uncertainty = out["uncertainty"]
    weight = 1.0 / (1.0 + uncertainty)
    l1 = (weight * (pred - target).abs()).sum() / (weight.sum() + 1e-6)
    ssim_loss = 1.0 - C.ssim(pred, target)
    pred_edge = edge_magnitude(out["warped_raw"])
    target_edge = edge_magnitude(target)
    edge = F.l1_loss(normalize_per_sample(pred_edge), normalize_per_sample(target_edge))
    affine = affine_identity_loss(out["theta"])
    flow_mag, flow_tv = flow_regularization(out.get("flow"))
    flow_mag = flow_mag.to(pred.device)
    flow_tv = flow_tv.to(pred.device)
    if aligned_rgb is None:
        warp_rgb = pred.new_tensor(0.0)
    else:
        warp_rgb = F.l1_loss(out["warped_raw"], aligned_rgb)
    unc_tv = tv_loss(uncertainty)
    total = (
        l1
        + args.lambda_ssim * ssim_loss
        + args.lambda_edge * edge
        + args.lambda_affine * affine
        + args.lambda_flow * flow_mag
        + args.lambda_flow_tv * flow_tv
        + args.lambda_warp_rgb * warp_rgb
        + args.lambda_uncertainty * uncertainty.mean()
        + args.lambda_uncertainty_tv * unc_tv
    )
    return {
        "total": total,
        "l1": l1.detach(),
        "ssim": ssim_loss.detach(),
        "edge": edge.detach(),
        "affine": affine.detach(),
        "flow_mag": flow_mag.detach(),
        "flow_tv": flow_tv.detach(),
        "warp_rgb": warp_rgb.detach(),
        "uncertainty": uncertainty.mean().detach(),
        "unc_tv": unc_tv.detach(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, res: int) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    aux: list[dict[str, float]] = []
    for batch in loader:
        batch = resize_registration_batch(batch, res)
        rgb_raw = batch["rgb_input_raw"].to(device)
        target = batch["target"].to(device)
        aligned_rgb = batch.get("rgb_raw")
        aligned_rgb = aligned_rgb.to(device) if torch.is_tensor(aligned_rgb) else None
        out = model(rgb_raw, target)
        pred = out["pred"].cpu().numpy()
        tgt = target.cpu().numpy()
        theta = out["theta"].detach().cpu()
        uncertainty = out["uncertainty"].detach().cpu()
        flow = out.get("flow")
        flow_cpu = flow.detach().cpu() if flow is not None else None
        for i in range(len(pred)):
            m = C.metrics_np(pred[i, 0], tgt[i, 0])
            m["ssim"] = C.ssim_np(pred[i, 0], tgt[i, 0])
            rows.append(m)
            ident = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            aux.append(
                {
                    "theta_l2": float((theta[i] - ident).square().mean().sqrt()),
                    "uncertainty": float(uncertainty[i].mean()),
                    "flow_l2": float(flow_cpu[i].square().mean().sqrt()) if flow_cpu is not None else 0.0,
                    "flow_tv": float(tv_loss(flow_cpu[i : i + 1])) if flow_cpu is not None else 0.0,
                    "warp_rgb_mae": (
                        float(F.l1_loss(out["warped_raw"][i : i + 1], aligned_rgb[i : i + 1]).detach().cpu())
                        if aligned_rgb is not None
                        else 0.0
                    ),
                }
            )
    keys = rows[0].keys()
    out = {k: float(np.mean([m[k] for m in rows])) for k in keys}
    out.update({k: float(np.mean([m[k] for m in aux])) for k in aux[0].keys()})
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="week3_reg_v0_ann_arbor")
    parser.add_argument("--dataset", default="ann_arbor", choices=["ann_arbor", "kust4k", "caltech_cart"])
    parser.add_argument("--eval-dataset", default=None, choices=["ann_arbor", "kust4k", "caltech_cart"])
    parser.add_argument("--ann-arbor-cache", default=None, help="Kept for metadata compatibility; R2T_CACHE controls Ann Arbor loading.")
    parser.add_argument("--kust4k-root", default=None)
    parser.add_argument("--caltech-root", default=None)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--target-normalization", default="raw", choices=["raw", "robust", "histmatch"])
    parser.add_argument("--target-normalization-stats")
    parser.add_argument(
        "--arch",
        default="target_conditioned",
        choices=["target_conditioned", "shared_rgb", "no_registration", "input_rgb_affine", "input_rgb_flow"],
    )
    parser.add_argument("--encoder", default="convnext_tiny")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sigma", type=float, default=0.3)
    parser.add_argument("--eval-sigma", type=float, default=0.3)
    parser.add_argument("--max-translation-frac", type=float, default=DEFAULT_TRANSLATION_FRAC)
    parser.add_argument("--max-rotation-deg", type=float, default=DEFAULT_ROTATION_DEG)
    parser.add_argument("--max-scale-frac", type=float, default=DEFAULT_SCALE_FRAC)
    parser.add_argument("--lambda-ssim", type=float, default=0.25)
    parser.add_argument("--lambda-edge", type=float, default=0.10)
    parser.add_argument("--lambda-affine", type=float, default=0.02)
    parser.add_argument("--lambda-flow", type=float, default=0.02)
    parser.add_argument("--lambda-flow-tv", type=float, default=0.01)
    parser.add_argument("--lambda-warp-rgb", type=float, default=0.0)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.01)
    parser.add_argument("--lambda-uncertainty-tv", type=float, default=0.005)
    parser.add_argument("--max-flow", type=float, default=DEFAULT_MAX_FLOW)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    args.eval_dataset = args.eval_dataset or args.dataset

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = make_registration_dataset(
        args,
        "train",
        sigma=args.train_sigma,
        augment=True,
        seed=args.seed,
        limit=args.max_train,
        dataset=args.dataset,
    )
    val_ds = make_registration_dataset(
        args,
        args.eval_split,
        sigma=args.eval_sigma,
        augment=False,
        seed=args.seed + 100000,
        limit=args.max_val,
        dataset=args.eval_dataset,
    )
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    if args.arch == "shared_rgb":
        model = SharedFeatureRegistrationTranslator(args.encoder).to(device)
    elif args.arch == "no_registration":
        model = NoRegistrationTranslator(args.encoder).to(device)
    elif args.arch == "input_rgb_affine":
        model = RGBInputAffineRegistrationTranslator(args.encoder).to(device)
    elif args.arch == "input_rgb_flow":
        model = RGBDenseFlowRegistrationTranslator(args.encoder, max_flow=args.max_flow).to(device)
    else:
        model = RegistrationTranslator(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out_dir or f"{C.BASE}/rgb2thermal_wacv/week3_runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **vars(args),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "device": str(device),
    }
    write_json(out_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)

    best_mae = math.inf
    history: list[dict[str, float | int]] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        accum: dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            batch = resize_registration_batch(batch, args.res)
            rgb_raw = batch["rgb_input_raw"].to(device)
            target = batch["target"].to(device)
            aligned_rgb = batch.get("rgb_raw")
            aligned_rgb = aligned_rgb.to(device) if torch.is_tensor(aligned_rgb) else None
            out = model(rgb_raw, target)
            losses = loss_terms(out, target, aligned_rgb, args)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batches += 1
            for key, value in losses.items():
                accum[key] = accum.get(key, 0.0) + float(value.detach().cpu())
        sched.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, device, args.res)
            row = {
                "epoch": epoch,
                **{f"train_{k}": v / max(1, batches) for k, v in accum.items()},
                **{f"val_{k}": v for k, v in val.items()},
                "elapsed_sec": round(time.time() - t0, 1),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if val["mae"] < best_mae:
                best_mae = val["mae"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "val": val,
                    },
                    out_dir / "best.pth",
                )
                write_json(out_dir / "best_metrics.json", {"epoch": epoch, **val})

    write_json(out_dir / "metrics.json", {"history": history, "best_mae": best_mae, "args": vars(args)})
    print(f"[{args.name}] DONE best_val_mae={best_mae:.4f}", flush=True)


if __name__ == "__main__":
    main()
