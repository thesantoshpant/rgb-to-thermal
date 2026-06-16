#!/usr/bin/env python3
"""Collect final registration metrics from run directories."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read_run(root: Path, run: str, baseline_psnr: dict[tuple[str, int], float]) -> dict[str, float | int | str]:
    run_dir = root / run
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    final = metrics["history"][-1]
    dataset = metadata.get("dataset", "ann_arbor")
    seed = int(metadata["seed"])
    psnr = float(final["val_psnr"])
    arch = metadata["arch"]
    baseline = baseline_psnr.get((dataset, seed))
    return {
        "run": run,
        "arch": arch,
        "dataset": dataset,
        "seed": seed,
        "train_sigma": float(metadata["train_sigma"]),
        "eval_sigma": float(metadata["eval_sigma"]),
        "translation_frac": float(metadata["max_translation_frac"]),
        "rotation_deg": float(metadata["max_rotation_deg"]),
        "scale_frac": float(metadata["max_scale_frac"]),
        "epochs": int(metadata["epochs"]),
        "res": int(metadata["res"]),
        "batch_size": int(metadata["bs"]),
        "train_count": int(metadata["train_size"]),
        "val_count": int(metadata["val_size"]),
        "final_val_mae": float(final["val_mae"]),
        "final_val_psnr": psnr,
        "final_val_ssim": float(final["val_ssim"]),
        "final_val_corr": float(final["val_corr"]),
        "final_theta_l2": float(final.get("val_theta_l2", 0.0)),
        "final_flow_l2": float(final.get("val_flow_l2", 0.0)),
        "final_flow_tv": float(final.get("val_flow_tv", 0.0)),
        "final_warp_rgb_mae": float(final.get("val_warp_rgb_mae", 0.0)),
        "final_uncertainty": float(final.get("val_uncertainty", 0.0)),
        "delta_vs_no_registration": "" if arch == "no_registration" or baseline is None else psnr - baseline,
        "knox_path": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="Run root to search, e.g. week4_runs")
    parser.add_argument("--run", action="append", required=True, help="Run directory name to collect")
    args = parser.parse_args()

    roots = [Path(p) for p in args.root]
    run_to_root: dict[str, Path] = {}
    for run in args.run:
        for root in roots:
            if (root / run / "metrics.json").exists():
                run_to_root[run] = root
                break
        if run not in run_to_root:
            raise FileNotFoundError(run)

    baseline_psnr: dict[tuple[str, int], float] = {}
    for run, root in run_to_root.items():
        metadata = json.loads((root / run / "metadata.json").read_text(encoding="utf-8"))
        if metadata["arch"] == "no_registration":
            final = json.loads((root / run / "metrics.json").read_text(encoding="utf-8"))["history"][-1]
            baseline_psnr[(metadata.get("dataset", "ann_arbor"), int(metadata["seed"]))] = float(final["val_psnr"])

    rows = [read_run(run_to_root[run], run, baseline_psnr) for run in args.run]
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
