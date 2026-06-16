#!/usr/bin/env python3
"""Week 4 target-normalization audit for Ann Arbor, Kust4K, and Caltech CART.

The goal is to quantify how comparable the scalar targets are across datasets.
Ann Arbor uses a palette-inverted scalar target; Kust4K/CART currently use raw
normalized thermal grayscale in unified_dataset.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unified_dataset import UnifiedR2TDataset


def edge_stats(x: torch.Tensor) -> tuple[float, float]:
    x = x.float().view(1, 1, *x.shape[-2:])
    sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype).view(1, 1, 3, 3) / 8.0
    sy = sx.transpose(2, 3)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8).flatten().numpy()
    return float(mag.mean()), float(np.quantile(mag, 0.90))


def entropy01(x: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(np.clip(x, 0.0, 1.0), bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p = p / max(1.0, p.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def summarize_dataset(tag: str, ds: UnifiedR2TDataset, max_items: int) -> dict[str, float | int | str]:
    rows = []
    n = min(len(ds), max_items or len(ds))
    for i in range(n):
        _rgb, _thermal, scalar, dataset_tag, _quality = ds[i]
        arr = scalar[0].numpy().astype(np.float32)
        e_mean, e_p90 = edge_stats(scalar[0])
        rows.append(
            {
                "dataset": dataset_tag or tag,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "p01": float(np.quantile(arr, 0.01)),
                "p50": float(np.quantile(arr, 0.50)),
                "p99": float(np.quantile(arr, 0.99)),
                "edge_mean": e_mean,
                "edge_p90": e_p90,
                "entropy": entropy01(arr),
            }
        )
    out: dict[str, float | int | str] = {"dataset": tag, "count": n}
    for key in ("mean", "std", "p01", "p50", "p99", "edge_mean", "edge_p90", "entropy"):
        vals = [r[key] for r in rows]
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-arbor-cache")
    parser.add_argument("--kust4k-root")
    parser.add_argument("--caltech-root")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--target-normalization", default="raw", choices=["raw", "robust", "histmatch"])
    parser.add_argument("--target-normalization-stats")
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--out", default="results/week4_target_normalization_audit.csv")
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    if args.ann_arbor_cache:
        rows.append(
            summarize_dataset(
                "ann_arbor",
                UnifiedR2TDataset.from_roots(
                    ann_arbor_cache=args.ann_arbor_cache,
                    split=args.split,
                    size_hw=(args.height, args.width),
                    target_norm=args.target_normalization,
                    target_norm_stats=args.target_normalization_stats,
                ),
                args.max_items,
            )
        )
    if args.kust4k_root:
        rows.append(
            summarize_dataset(
                "kust4k",
                UnifiedR2TDataset.from_roots(
                    kust4k_root=args.kust4k_root,
                    split=args.split,
                    size_hw=(args.height, args.width),
                    target_norm=args.target_normalization,
                    target_norm_stats=args.target_normalization_stats,
                ),
                args.max_items,
            )
        )
    if args.caltech_root:
        rows.append(
            summarize_dataset(
                "caltech_cart",
                UnifiedR2TDataset.from_roots(
                    caltech_root=args.caltech_root,
                    split=args.split,
                    size_hw=(args.height, args.width),
                    target_norm=args.target_normalization,
                    target_norm_stats=args.target_normalization_stats,
                ),
                args.max_items,
            )
        )
    if not rows:
        raise SystemExit("No datasets were provided")
    out = Path(args.out)
    write_csv(out, rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
