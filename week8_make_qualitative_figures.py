#!/usr/bin/env python3
"""Generate Week 8 qualitative figures for the locked registration model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import r2t_common as C
from week3_registration_v0 import (
    DEFAULT_ROTATION_DEG,
    DEFAULT_SCALE_FRAC,
    DEFAULT_TRANSLATION_FRAC,
    NoRegistrationTranslator,
    RGBInputAffineRegistrationTranslator,
    make_registration_dataset,
    resize_registration_batch,
    seed_all,
)


IDENTITY = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
DEFAULT_CHECKPOINT = (
    "week7_runs/"
    "week7_convnext_affine_deterministic_lam0p5_ann_arbor_robust_sigma03_seed42_e50/"
    "best.pth"
)


def to_rgb_uint8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=2)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def colorize_scalar(x: np.ndarray) -> np.ndarray:
    x = np.squeeze(np.asarray(x))
    try:
        return C.to_color(np.clip(x, 0.0, 1.0))
    except Exception:
        return to_rgb_uint8(x)


def colorize_heat(x: np.ndarray, vmax: float | None = None) -> np.ndarray:
    x = np.squeeze(np.asarray(x))
    if vmax is None:
        vmax = float(np.percentile(x, 99)) if x.size else 1.0
    z = np.clip(x / max(vmax, 1e-6), 0.0, 1.0)
    anchors = np.array(
        [
            [0, 0, 20],
            [0, 60, 160],
            [220, 90, 20],
            [255, 230, 120],
        ],
        dtype=np.float32,
    )
    pos = z * (len(anchors) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(anchors) - 1)
    frac = (pos - lo)[..., None]
    rgb = anchors[lo] * (1.0 - frac) + anchors[hi] * frac
    return rgb.astype(np.uint8)


def tile(img: np.ndarray) -> Image.Image:
    return Image.fromarray(to_rgb_uint8(img))


def image_tile(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def draw_grid(
    rows: list[dict],
    columns: list[tuple[str, str]],
    out_path: Path,
    title: str,
    tile_w: int,
) -> None:
    if not rows:
        raise RuntimeError("No rows available for figure")
    h, w = rows[0]["target"].shape
    tile_h = max(1, round(tile_w * h / w))
    left_w = 185
    header_h = 46
    row_h = tile_h + 36
    pad = 10
    width = left_w + len(columns) * (tile_w + pad) + pad
    height = header_h + len(rows) * row_h + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(0, 0, 0))
    for c, (label, _key) in enumerate(columns):
        x = left_w + c * (tile_w + pad)
        draw.text((x, 26), label, fill=(0, 0, 0))
    for r, row in enumerate(rows):
        y = header_h + r * row_h
        name = str(row["name"])[:24]
        draw.text((pad, y + 4), name, fill=(0, 0, 0))
        draw.text((pad, y + 20), f"PSNR {row['psnr']:.2f} / r {row['corr']:.2f}", fill=(40, 40, 40))
        draw.text((pad, y + 36), f"warp MAE {row['warp_rgb_mae']:.3f}", fill=(40, 40, 40))
        for c, (_label, key) in enumerate(columns):
            x = left_w + c * (tile_w + pad)
            canvas.paste(row[key].resize((tile_w, tile_h), Image.BILINEAR), (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))


def build_model(arch: str, encoder: str) -> torch.nn.Module:
    if arch == "input_rgb_affine":
        return RGBInputAffineRegistrationTranslator(encoder)
    if arch == "no_registration":
        return NoRegistrationTranslator(encoder)
    raise ValueError(f"Unsupported qualitative arch: {arch}")


def metric_row(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    base = C.metrics_np(pred, target)
    base["ssim"] = C.ssim_np(pred, target)
    return base


def select_samples(rows: list[dict], n: int) -> list[dict]:
    rows = sorted(rows, key=lambda x: x["psnr"])
    if len(rows) <= n:
        selected = rows
    else:
        picks = np.linspace(0, len(rows) - 1, n).round().astype(int).tolist()
        selected = [rows[i] for i in picks]
    for i, row in enumerate(selected):
        row["selection_role"] = f"psnr_quantile_{i + 1}_of_{len(selected)}"
    return selected


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default="figures/week8")
    parser.add_argument("--dataset", default="ann_arbor", choices=["ann_arbor", "kust4k", "caltech_cart"])
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--kust4k-root", default=None)
    parser.add_argument("--caltech-root", default=None)
    parser.add_argument("--target-normalization", default="robust", choices=["raw", "robust", "histmatch"])
    parser.add_argument("--target-normalization-stats", default=None)
    parser.add_argument("--arch", default="input_rgb_affine", choices=["input_rgb_affine", "no_registration"])
    parser.add_argument("--encoder", default="convnext_tiny")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-sigma", type=float, default=0.3)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=220)
    parser.add_argument("--max-translation-frac", type=float, default=DEFAULT_TRANSLATION_FRAC)
    parser.add_argument("--max-rotation-deg", type=float, default=DEFAULT_ROTATION_DEG)
    parser.add_argument("--max-scale-frac", type=float, default=DEFAULT_SCALE_FRAC)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    ds = make_registration_dataset(
        args,
        args.eval_split,
        sigma=args.eval_sigma,
        augment=False,
        seed=args.seed + 100000,
        limit=args.max_val,
        dataset=args.dataset,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    arch = str(ckpt_args.get("arch") or args.arch)
    encoder = str(ckpt_args.get("encoder") or args.encoder)
    model = build_model(arch, encoder).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows: list[dict] = []
    csv_rows: list[dict[str, float | str | int]] = []
    sample_idx = 0
    identity = IDENTITY.to(device)
    for batch in loader:
        batch = resize_registration_batch(batch, args.res)
        rgb_input = batch["rgb_input_raw"].to(device)
        target = batch["target"].to(device)
        aligned = batch.get("rgb_raw")
        aligned = aligned.to(device) if torch.is_tensor(aligned) else rgb_input
        out = model(rgb_input, target)

        pred = out["pred"].clamp(0, 1).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        rgb_input_np = rgb_input.detach().cpu().numpy()
        aligned_np = aligned.detach().cpu().numpy()
        warped_np = out["warped_raw"].clamp(0, 1).detach().cpu().numpy()
        uncertainty_np = out["uncertainty"].detach().cpu().numpy()
        theta = out["theta"].detach()
        theta_l2 = ((theta - identity.view(1, 2, 3)) ** 2).mean(dim=(1, 2)).sqrt().cpu().numpy()
        names = batch.get("name")
        if names is None:
            names = [f"{args.dataset}_{sample_idx + i:04d}" for i in range(pred.shape[0])]

        for i in range(pred.shape[0]):
            p = pred[i, 0]
            t = target_np[i, 0]
            m = metric_row(p, t)
            before = np.mean(np.abs(rgb_input_np[i] - aligned_np[i]), axis=0)
            after = np.mean(np.abs(warped_np[i] - aligned_np[i]), axis=0)
            err = np.abs(p - t)
            warp_mae = float(np.mean(np.abs(warped_np[i] - aligned_np[i])))
            name = str(names[i])
            csv_row = {
                "index": sample_idx,
                "name": name,
                "psnr": m["psnr"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "ssim": m["ssim"],
                "corr": m["corr"],
                "theta_l2": float(theta_l2[i]),
                "uncertainty": float(np.mean(uncertainty_np[i])),
                "warp_rgb_mae": warp_mae,
                "selection_role": "",
            }
            rows.append(
                {
                    **csv_row,
                    "input_rgb": tile(rgb_input_np[i]),
                    "aligned_rgb": tile(aligned_np[i]),
                    "warped_rgb": tile(warped_np[i]),
                    "target": t,
                    "target_img": image_tile(colorize_scalar(t)),
                    "pred_img": image_tile(colorize_scalar(p)),
                    "error_img": image_tile(colorize_heat(err)),
                    "uncertainty_img": image_tile(colorize_heat(uncertainty_np[i, 0])),
                    "before_img": image_tile(colorize_heat(before)),
                    "after_img": image_tile(colorize_heat(after)),
                }
            )
            csv_rows.append(csv_row)
            sample_idx += 1

    selected = select_samples(rows, max(1, args.num_samples))
    role_by_index = {row["index"]: row["selection_role"] for row in selected}
    for row in csv_rows:
        row["selection_role"] = role_by_index.get(row["index"], "")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ann_arbor_candidate_metrics_seed42.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    draw_grid(
        selected,
        [
            ("RGB input", "input_rgb"),
            ("Warped RGB", "warped_rgb"),
            ("Target", "target_img"),
            ("Prediction", "pred_img"),
            ("Abs error", "error_img"),
            ("Uncertainty", "uncertainty_img"),
        ],
        out_dir / "ann_arbor_candidate_grid_seed42.png",
        "Week 8 Ann Arbor qualitative candidates",
        args.tile_width,
    )
    draw_grid(
        selected[: min(4, len(selected))],
        [
            ("Misaligned RGB", "input_rgb"),
            ("Aligned RGB", "aligned_rgb"),
            ("Predicted warp", "warped_rgb"),
            ("Before residual", "before_img"),
            ("After residual", "after_img"),
            ("Target", "target_img"),
        ],
        out_dir / "misalignment_recovery_seed42.png",
        "Week 8 misalignment recovery candidates",
        args.tile_width,
    )
    print(f"wrote {out_dir / 'ann_arbor_candidate_grid_seed42.png'}")
    print(f"wrote {out_dir / 'misalignment_recovery_seed42.png'}")
    print(f"wrote {out_dir / 'ann_arbor_candidate_metrics_seed42.csv'}")


if __name__ == "__main__":
    main()
