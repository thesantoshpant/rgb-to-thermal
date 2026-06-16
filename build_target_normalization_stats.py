#!/usr/bin/env python3
"""Build target-distribution stats for cross-dataset target normalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unified_dataset import UnifiedR2TDataset


def sample_dataset(ds: UnifiedR2TDataset, max_items: int, pixels_per_item: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    n = min(len(ds), max_items or len(ds))
    for i in range(n):
        _rgb, _thermal, scalar, _tag, _quality = ds[i]
        arr = scalar[0].numpy().astype(np.float32).reshape(-1)
        if pixels_per_item and arr.size > pixels_per_item:
            idx = rng.choice(arr.size, size=pixels_per_item, replace=False)
            arr = arr[idx]
        samples.append(arr)
    if not samples:
        raise RuntimeError("No samples found")
    return np.concatenate(samples)


def summarize(name: str, ds: UnifiedR2TDataset, args: argparse.Namespace, seed_offset: int) -> dict:
    pixels = sample_dataset(ds, args.max_items, args.pixels_per_item, args.seed + seed_offset)
    levels = np.linspace(0.0, 1.0, args.quantiles)
    quantiles = np.quantile(pixels, levels).astype(float)
    quantiles = np.maximum.accumulate(quantiles)
    return {
        "dataset": name,
        "count": len(ds),
        "sampled_pixels": int(pixels.size),
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "quantiles": [float(x) for x in quantiles],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-arbor-cache")
    parser.add_argument("--kust4k-root")
    parser.add_argument("--caltech-root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--pixels-per-item", type=int, default=4096)
    parser.add_argument("--quantiles", type=int, default=257)
    parser.add_argument("--reference", default="ann_arbor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/week5_target_normalization_stats.json")
    args = parser.parse_args()

    datasets: dict[str, dict] = {}
    if args.ann_arbor_cache:
        datasets["ann_arbor"] = summarize(
            "ann_arbor",
            UnifiedR2TDataset.from_roots(
                ann_arbor_cache=args.ann_arbor_cache,
                split=args.split,
                size_hw=(args.height, args.width),
            ),
            args,
            0,
        )
    if args.kust4k_root:
        datasets["kust4k"] = summarize(
            "kust4k",
            UnifiedR2TDataset.from_roots(
                kust4k_root=args.kust4k_root,
                split=args.split,
                size_hw=(args.height, args.width),
            ),
            args,
            1000,
        )
    if args.caltech_root:
        datasets["caltech_cart"] = summarize(
            "caltech_cart",
            UnifiedR2TDataset.from_roots(
                caltech_root=args.caltech_root,
                split=args.split,
                size_hw=(args.height, args.width),
            ),
            args,
            2000,
        )
    if args.reference not in datasets:
        raise SystemExit(f"Reference dataset {args.reference!r} was not provided")

    payload = {
        "reference": args.reference,
        "split": args.split,
        "height": args.height,
        "width": args.width,
        "quantile_levels": [float(x) for x in np.linspace(0.0, 1.0, args.quantiles)],
        "datasets": datasets,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "quantiles"} for k, v in datasets.items()}, indent=2))


if __name__ == "__main__":
    main()
