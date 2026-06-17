#!/usr/bin/env python3
"""Generate final Week 8 qualitative figure candidates.

The figures are qualitative artifacts only. Ann Arbor uses the locked Week 7
uncertainty-decoupled affine protocol; Kust4K/CART rows use the best available
within-dataset checkpoints from earlier weeks and are labeled as external
qualitative evidence rather than a unified quantitative comparison.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from week3_registration_v0 import (
    DEFAULT_ROTATION_DEG,
    DEFAULT_SCALE_FRAC,
    DEFAULT_TRANSLATION_FRAC,
    make_registration_dataset,
    resize_registration_batch,
    seed_all,
)
from week8_make_qualitative_figures import (
    IDENTITY,
    build_model,
    colorize_heat,
    colorize_scalar,
    metric_row,
    tile,
)


AA_OURS = "week7_runs/week7_convnext_affine_deterministic_lam0p5_ann_arbor_robust_sigma03_seed42_e50/best.pth"
AA_BASE = "week6_runs/week6_convnext_unet_no_registration_ann_arbor_robust_sigma03_seed42_e50/best.pth"
K4K_OURS = "week4_runs/week4_input_rgb_affine_warprgb1_kust4k_sigma03_amp_seed42_e20/best.pth"
K4K_BASE = "week4_runs/week4_no_registration_kust4k_sigma03_amp_seed42_e20/best.pth"
CART_OURS = "week5_runs/week5_input_rgb_affine_warprgb0p5_caltech_cart_sigma03_amp_seed42_e20/best.pth"
CART_BASE = "week4_runs/week4_no_registration_caltech_cart_sigma03_amp_seed42_e20/best.pth"


DATASETS = {
    "ann_arbor": {
        "label": "Ann Arbor",
        "target_norm": "robust",
        "ours": AA_OURS,
        "baseline": AA_BASE,
        "index": None,
        "note": "Week 7 robust protocol",
    },
    "kust4k": {
        "label": "Kust4K",
        "target_norm": "raw",
        "ours": K4K_OURS,
        "baseline": K4K_BASE,
        "index": None,
        "note": "legacy within-dataset checkpoint",
    },
    "caltech_cart": {
        "label": "CART",
        "target_norm": "raw",
        "ours": CART_OURS,
        "baseline": CART_BASE,
        "index": None,
        "note": "legacy within-dataset checkpoint",
    },
}


def save_canvas(canvas: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, dpi=(300, 300))


def paste_fit(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    canvas.paste(img.resize((x1 - x0, y1 - y0), Image.BILINEAR), (x0, y0))


def make_args(base: argparse.Namespace, dataset: str, target_norm: str, sigma: float) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        eval_dataset=dataset,
        eval_split=base.eval_split,
        kust4k_root=base.kust4k_root,
        caltech_root=base.caltech_root,
        target_normalization=target_norm,
        target_normalization_stats=None,
        max_translation_frac=base.max_translation_frac,
        max_rotation_deg=base.max_rotation_deg,
        max_scale_frac=base.max_scale_frac,
    )


def dataset_for(base: argparse.Namespace, dataset: str, target_norm: str, sigma: float, limit: int = 0):
    ds_args = make_args(base, dataset, target_norm, sigma)
    return make_registration_dataset(
        ds_args,
        base.eval_split,
        sigma=sigma,
        augment=False,
        seed=base.seed + 100000,
        limit=limit,
        dataset=dataset,
    )


def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = build_model(str(ckpt_args.get("arch")), str(ckpt_args.get("encoder", "convnext_tiny"))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def find_index(ds, name: str | None, fallback: int) -> int:
    if name is None:
        return min(max(0, fallback), len(ds) - 1)
    for idx in range(len(ds)):
        row = ds[idx]
        if isinstance(row, dict) and str(row.get("name")) == str(name):
            return idx
    return min(max(0, fallback), len(ds) - 1)


@torch.no_grad()
def predict_one(model: torch.nn.Module, ds, idx: int, device: torch.device, res: int) -> dict:
    loader = DataLoader(Subset(ds, [idx]), batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = resize_registration_batch(batch, res)
    rgb_input = batch["rgb_input_raw"].to(device)
    target = batch["target"].to(device)
    aligned = batch.get("rgb_raw")
    aligned = aligned.to(device) if torch.is_tensor(aligned) else rgb_input
    out = model(rgb_input, target)
    pred = out["pred"].clamp(0, 1).detach().cpu().numpy()[0, 0]
    tgt = target.detach().cpu().numpy()[0, 0]
    metrics = metric_row(pred, tgt)
    warped = out["warped_raw"].clamp(0, 1).detach().cpu().numpy()[0]
    rgb_np = rgb_input.detach().cpu().numpy()[0]
    aligned_np = aligned.detach().cpu().numpy()[0]
    uncertainty = out["uncertainty"].detach().cpu().numpy()[0, 0]
    theta = out["theta"].detach()
    theta_l2 = float(((theta - IDENTITY.to(device).view(1, 2, 3)) ** 2).mean(dim=(1, 2)).sqrt().cpu()[0])
    name = batch.get("name", [f"idx_{idx}"])
    if isinstance(name, (list, tuple)):
        name = str(name[0])
    return {
        "idx": idx,
        "name": str(name),
        "rgb": rgb_np,
        "aligned": aligned_np,
        "warped": warped,
        "target": tgt,
        "pred": pred,
        "uncertainty": uncertainty,
        "error": np.abs(pred - tgt),
        "before": np.mean(np.abs(rgb_np - aligned_np), axis=0),
        "after": np.mean(np.abs(warped - aligned_np), axis=0),
        "theta_l2": theta_l2,
        "warp_rgb_mae": float(np.mean(np.abs(warped - aligned_np))),
        **metrics,
    }


@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    ds,
    device: torch.device,
    res: int,
    max_items: int,
) -> list[dict]:
    limit = min(len(ds), max_items or len(ds))
    rows = []
    for idx in range(limit):
        pred = predict_one(model, ds, idx, device, res)
        rows.append({"idx": idx, "name": pred["name"], "psnr": pred["psnr"], "corr": pred["corr"]})
    return rows


def choose_median_psnr(scored: list[dict]) -> int:
    scored = sorted(scored, key=lambda r: r["psnr"])
    return int(scored[len(scored) // 2]["idx"])


def choose_failure_indices(scored: list[dict], n: int) -> list[int]:
    return [int(r["idx"]) for r in sorted(scored, key=lambda r: r["psnr"])[:n]]


def draw_labeled_grid(
    path: Path,
    title: str,
    rows: list[dict],
    columns: list[tuple[str, str]],
    row_labels: list[str],
    tile_w: int,
    left_w: int = 210,
) -> None:
    h, w = rows[0]["target"].shape
    tile_h = max(1, round(tile_w * h / w))
    pad = 10
    header_h = 52
    row_h = tile_h + 42
    canvas = Image.new("RGB", (left_w + len(columns) * (tile_w + pad) + pad, header_h + len(rows) * row_h + pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(0, 0, 0))
    for col, (label, _key) in enumerate(columns):
        draw.text((left_w + col * (tile_w + pad), 30), label, fill=(0, 0, 0))
    for row_idx, row in enumerate(rows):
        y = header_h + row_idx * row_h
        draw.text((pad, y + 4), row_labels[row_idx], fill=(0, 0, 0))
        draw.text((pad, y + 20), f"ours {row.get('ours_psnr', row['psnr']):.2f} dB", fill=(45, 45, 45))
        if "baseline_psnr" in row:
            draw.text((pad, y + 36), f"base {row['baseline_psnr']:.2f} dB", fill=(45, 45, 45))
        for col, (_label, key) in enumerate(columns):
            paste_fit(canvas, row[key], (left_w + col * (tile_w + pad), y, left_w + col * (tile_w + pad) + tile_w, y + tile_h))
    save_canvas(canvas, path)


def row_images(row: dict, prefix: str = "") -> dict:
    return {
        f"{prefix}rgb_img": tile(row["rgb"]),
        f"{prefix}warped_img": tile(row["warped"]),
        f"{prefix}target_img": Image.fromarray(colorize_scalar(row["target"])),
        f"{prefix}pred_img": Image.fromarray(colorize_scalar(row["pred"])),
        f"{prefix}error_img": Image.fromarray(colorize_heat(row["error"])),
        f"{prefix}uncertainty_img": Image.fromarray(colorize_heat(row["uncertainty"])),
        f"{prefix}before_img": Image.fromarray(colorize_heat(row["before"])),
        f"{prefix}after_img": Image.fromarray(colorize_heat(row["after"])),
    }


def write_metrics(path: Path, rows: list[dict]) -> None:
    keys = [
        "figure",
        "dataset",
        "sample",
        "idx",
        "model",
        "sigma",
        "psnr",
        "ssim",
        "corr",
        "mae",
        "rmse",
        "theta_l2",
        "warp_rgb_mae",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in keys} for row in rows)


def metric_record(figure: str, dataset: str, model_name: str, sigma: float, pred: dict, note: str = "") -> dict:
    return {
        "figure": figure,
        "dataset": dataset,
        "sample": pred["name"],
        "idx": pred["idx"],
        "model": model_name,
        "sigma": sigma,
        "psnr": pred["psnr"],
        "ssim": pred["ssim"],
        "corr": pred["corr"],
        "mae": pred["mae"],
        "rmse": pred["rmse"],
        "theta_l2": pred["theta_l2"],
        "warp_rgb_mae": pred["warp_rgb_mae"],
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures/week8")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--kust4k-root", default="data_cache/external/kust4k")
    parser.add_argument("--caltech-root", default="data_cache/external/caltech")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--tile-width", type=int, default=230)
    parser.add_argument("--hero-name", default="326")
    parser.add_argument("--max-score-items", type=int, default=0)
    parser.add_argument("--max-translation-frac", type=float, default=DEFAULT_TRANSLATION_FRAC)
    parser.add_argument("--max-rotation-deg", type=float, default=DEFAULT_ROTATION_DEG)
    parser.add_argument("--max-scale-frac", type=float, default=DEFAULT_SCALE_FRAC)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_all(args.seed)
    out_dir = Path(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    metrics: list[dict] = []

    aa_ds = dataset_for(args, "ann_arbor", "robust", sigma=0.3)
    aa_ours = load_model(AA_OURS, device)
    aa_base = load_model(AA_BASE, device)
    hero_idx = find_index(aa_ds, args.hero_name, fallback=0)
    hero = predict_one(aa_ours, aa_ds, hero_idx, device, args.res)
    hero_base = predict_one(aa_base, aa_ds, hero_idx, device, args.res)
    hero.update(row_images(hero))
    hero["baseline_psnr"] = hero_base["psnr"]
    metrics.append(metric_record("hero", "ann_arbor", "ours", 0.3, hero, "locked Week 7 checkpoint"))
    metrics.append(metric_record("hero", "ann_arbor", "baseline", 0.3, hero_base, "ConvNeXt no-registration"))
    draw_labeled_grid(
        out_dir / "hero_ann_arbor_seed42.png",
        "Hero candidate: Ann Arbor robust target, seed 42",
        [hero],
        [
            ("RGB input", "rgb_img"),
            ("Target", "target_img"),
            ("Ours", "pred_img"),
            ("Uncertainty", "uncertainty_img"),
            ("Abs error", "error_img"),
        ],
        [f"{hero['name']}"],
        args.tile_width,
    )

    scored_aa = score_dataset(aa_ours, aa_ds, device, args.res, args.max_score_items)
    failure_indices = choose_failure_indices(scored_aa, 2)
    failure_rows = []
    failure_labels = []
    for idx in failure_indices:
        ours = predict_one(aa_ours, aa_ds, idx, device, args.res)
        base = predict_one(aa_base, aa_ds, idx, device, args.res)
        row = {**ours, **row_images(ours), "baseline_img": Image.fromarray(colorize_scalar(base["pred"])), "baseline_psnr": base["psnr"]}
        failure_rows.append(row)
        failure_labels.append(f"{ours['name']} / low PSNR")
        metrics.append(metric_record("failure", "ann_arbor", "ours", 0.3, ours, "selected among two lowest validation PSNR rows"))
        metrics.append(metric_record("failure", "ann_arbor", "baseline", 0.3, base, "same sample"))
    draw_labeled_grid(
        out_dir / "failure_cases_ann_arbor_seed42.png",
        "Failure cases: lowest Ann Arbor validation PSNR rows",
        failure_rows,
        [
            ("RGB input", "rgb_img"),
            ("Target", "target_img"),
            ("Ours", "pred_img"),
            ("Baseline", "baseline_img"),
            ("Abs error", "error_img"),
            ("Uncertainty", "uncertainty_img"),
        ],
        failure_labels,
        args.tile_width,
    )

    recovery_rows = []
    recovery_labels = []
    for sigma in (0.0, 0.2, 0.5):
        ds = dataset_for(args, "ann_arbor", "robust", sigma=sigma)
        idx = find_index(ds, args.hero_name, fallback=hero_idx)
        ours = predict_one(aa_ours, ds, idx, device, args.res)
        base = predict_one(aa_base, ds, idx, device, args.res)
        row = {
            **ours,
            **row_images(ours),
            "baseline_img": Image.fromarray(colorize_scalar(base["pred"])),
            "baseline_psnr": base["psnr"],
        }
        recovery_rows.append(row)
        recovery_labels.append(f"sigma {sigma:.1f}")
        metrics.append(metric_record("recovery", "ann_arbor", "ours", sigma, ours, "same scene across synthetic severity"))
        metrics.append(metric_record("recovery", "ann_arbor", "baseline", sigma, base, "same scene across synthetic severity"))
    draw_labeled_grid(
        out_dir / "misalignment_recovery_multisigma_seed42.png",
        "Misalignment recovery: same Ann Arbor scene across severity",
        recovery_rows,
        [
            ("RGB input", "rgb_img"),
            ("Baseline", "baseline_img"),
            ("Ours", "pred_img"),
            ("Target", "target_img"),
            ("Uncertainty", "uncertainty_img"),
            ("Abs error", "error_img"),
        ],
        recovery_labels,
        args.tile_width,
    )

    cross_rows = []
    cross_labels = []
    for dataset, cfg in DATASETS.items():
        ds = dataset_for(args, dataset, str(cfg["target_norm"]), sigma=0.3)
        ours_model = aa_ours if cfg["ours"] == AA_OURS else load_model(str(cfg["ours"]), device)
        base_model = aa_base if cfg["baseline"] == AA_BASE else load_model(str(cfg["baseline"]), device)
        scored = score_dataset(ours_model, ds, device, args.res, args.max_score_items)
        idx = choose_median_psnr(scored)
        ours = predict_one(ours_model, ds, idx, device, args.res)
        base = predict_one(base_model, ds, idx, device, args.res)
        row = {
            **ours,
            **row_images(ours),
            "baseline_img": Image.fromarray(colorize_scalar(base["pred"])),
            "baseline_psnr": base["psnr"],
        }
        cross_rows.append(row)
        cross_labels.append(f"{cfg['label']} / {cfg['note']}")
        metrics.append(metric_record("cross_dataset", dataset, "ours", 0.3, ours, str(cfg["note"])))
        metrics.append(metric_record("cross_dataset", dataset, "baseline", 0.3, base, str(cfg["note"])))
    draw_labeled_grid(
        out_dir / "cross_dataset_gallery_seed42.png",
        "Cross-dataset qualitative gallery (representative median-PSNR rows)",
        cross_rows,
        [
            ("RGB input", "rgb_img"),
            ("Target", "target_img"),
            ("Ours", "pred_img"),
            ("Baseline", "baseline_img"),
        ],
        cross_labels,
        args.tile_width,
        left_w=275,
    )

    write_metrics(out_dir / "week8_final_figure_metrics.csv", metrics)
    for path in (
        "hero_ann_arbor_seed42.png",
        "failure_cases_ann_arbor_seed42.png",
        "misalignment_recovery_multisigma_seed42.png",
        "cross_dataset_gallery_seed42.png",
        "week8_final_figure_metrics.csv",
    ):
        print(f"wrote {out_dir / path}")


if __name__ == "__main__":
    main()
