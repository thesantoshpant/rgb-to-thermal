#!/usr/bin/env python3
"""Apply controlled synthetic misalignment to an aligned RGB/thermal pair.

The RGB image is warped while the thermal target is copied unchanged. The severity parameter
sigma is in [0, 1] and scales translation, rotation, and zoom perturbations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def sample_transform(
    sigma: float,
    width: int,
    height: int,
    rng: np.random.Generator,
    max_translation_frac: float = 0.08,
    max_rotation_deg: float = 8.0,
    max_scale_delta: float = 0.15,
) -> dict[str, float]:
    sigma = float(np.clip(sigma, 0.0, 1.0))
    span = min(width, height)
    return {
        "sigma": sigma,
        "tx_px": float(rng.uniform(-1.0, 1.0) * max_translation_frac * sigma * span),
        "ty_px": float(rng.uniform(-1.0, 1.0) * max_translation_frac * sigma * span),
        "rotation_deg": float(rng.uniform(-1.0, 1.0) * max_rotation_deg * sigma),
        "scale": float(1.0 + rng.uniform(-1.0, 1.0) * max_scale_delta * sigma),
    }


def _fill_for_mode(img: Image.Image):
    if img.mode == "RGB":
        return (0, 0, 0)
    if img.mode == "RGBA":
        return (0, 0, 0, 0)
    return 0


def warp_rgb(rgb: Image.Image, params: dict[str, float]) -> Image.Image:
    """Warp RGB with a center-based similarity transform."""
    w, h = rgb.size
    cx, cy = w / 2.0, h / 2.0
    theta = math.radians(params["rotation_deg"])
    scale = params["scale"]
    tx, ty = params["tx_px"], params["ty_px"]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # Inverse map from output pixel to input pixel for PIL.Image.transform.
    inv_scale = 1.0 / max(scale, 1e-6)
    a = cos_t * inv_scale
    b = sin_t * inv_scale
    d = -sin_t * inv_scale
    e = cos_t * inv_scale
    c = cx - a * (cx - tx) - b * (cy - ty)
    f = cy - d * (cx - tx) - e * (cy - ty)

    return rgb.transform(
        rgb.size,
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BICUBIC,
        fillcolor=_fill_for_mode(rgb),
    )


def apply_pair(
    rgb_path: Path,
    thermal_path: Path,
    out_rgb_path: Path,
    out_thermal_path: Path,
    sigma: float,
    seed: int,
) -> dict[str, float | str]:
    rgb = Image.open(rgb_path).convert("RGB")
    rng = np.random.default_rng(seed)
    params = sample_transform(sigma, rgb.width, rgb.height, rng)
    out_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    out_thermal_path.parent.mkdir(parents=True, exist_ok=True)
    warp_rgb(rgb, params).save(out_rgb_path)
    if thermal_path.resolve() != out_thermal_path.resolve():
        shutil.copy2(thermal_path, out_thermal_path)
    return {
        "rgb": str(rgb_path),
        "thermal": str(thermal_path),
        "out_rgb": str(out_rgb_path),
        "out_thermal": str(out_thermal_path),
        "seed": seed,
        **params,
    }


def run_manifest(manifest_path: Path, output_root: Path, sigma: float, seed: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rgb = Path(row.get("rgb") or row.get("rgb_path") or "")
            thermal = Path(row.get("thermal") or row.get("thermal_path") or "")
            if not rgb.exists() or not thermal.exists():
                continue
            stem = rgb.stem
            out_rgb = output_root / "rgb" / f"{stem}_sigma{sigma:.2f}{rgb.suffix.lower()}"
            out_thermal = output_root / "thermal" / thermal.name
            rows.append(apply_pair(rgb, thermal, out_rgb, out_thermal, sigma, seed + i))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", help="Aligned RGB image path")
    ap.add_argument("--thermal", help="Aligned thermal target path")
    ap.add_argument("--out-rgb", help="Output misaligned RGB path")
    ap.add_argument("--out-thermal", help="Output copied thermal path")
    ap.add_argument("--pairs-csv", help="Optional CSV with rgb/rgb_path and thermal/thermal_path columns")
    ap.add_argument("--output-root", default="synthetic_misalignment")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params-out", default=None)
    args = ap.parse_args()

    if args.pairs_csv:
        results = run_manifest(Path(args.pairs_csv), Path(args.output_root), args.sigma, args.seed)
    else:
        missing = [name for name in ["rgb", "thermal", "out_rgb", "out_thermal"] if getattr(args, name.replace("-", "_"), None) is None]
        if missing:
            raise SystemExit(f"Missing required args for single-pair mode: {', '.join(missing)}")
        results = [
            apply_pair(
                Path(args.rgb),
                Path(args.thermal),
                Path(args.out_rgb),
                Path(args.out_thermal),
                args.sigma,
                args.seed,
            )
        ]

    payload = {"count": len(results), "results": results}
    text = json.dumps(payload, indent=2)
    if args.params_out:
        Path(args.params_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.params_out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
