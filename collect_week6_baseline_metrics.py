#!/usr/bin/env python3
"""Collect Week 6 baseline metrics into one comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[-1]


def _float(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return default


def _maybe_float(row: dict[str, Any], *keys: str) -> float | str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return ""


def _int(row: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    return default


def read_metrics_csv(label: str, family: str, run_dir: Path, note: str) -> dict[str, Any]:
    meta = _load_json(run_dir / "metadata.json")
    final = _last_csv_row(run_dir / "metrics.csv")
    target_normalization = meta.get("target_normalization", "")
    if not target_normalization and "robust" in str(run_dir):
        target_normalization = "robust"
    return {
        "label": label,
        "family": family,
        "run": run_dir.name,
        "dataset": meta.get("dataset", ""),
        "eval_dataset": meta.get("eval_dataset", meta.get("dataset", "")),
        "seed": _int(meta, "seed"),
        "target_normalization": target_normalization,
        "train_sigma": _float(meta, "train_sigma", "sigma"),
        "eval_sigma": _float(meta, "eval_sigma"),
        "translation_frac": _float(meta, "max_translation_frac"),
        "rotation_deg": _float(meta, "max_rotation_deg"),
        "scale_frac": _float(meta, "max_scale_frac"),
        "epochs": _int(meta, "epochs", default=_int(final, "epoch")),
        "train_count": _int(meta, "train_size"),
        "val_count": _int(meta, "eval_size"),
        "final_mae": _float(final, "mae"),
        "final_rmse": _float(final, "rmse"),
        "final_psnr": _float(final, "psnr"),
        "final_ssim": _maybe_float(final, "ssim"),
        "final_corr": _maybe_float(final, "corr"),
        "note": note,
        "path": str(run_dir),
    }


def read_registration_json(label: str, family: str, run_dir: Path, note: str) -> dict[str, Any]:
    meta = _load_json(run_dir / "metadata.json")
    metrics = _load_json(run_dir / "metrics.json")
    history = metrics.get("history") or []
    if not history:
        raise RuntimeError(f"No history in {run_dir / 'metrics.json'}")
    final = history[-1]
    return {
        "label": label,
        "family": family,
        "run": run_dir.name,
        "dataset": meta.get("dataset", ""),
        "eval_dataset": meta.get("eval_dataset", meta.get("dataset", "")),
        "seed": _int(meta, "seed"),
        "target_normalization": meta.get("target_normalization", ""),
        "train_sigma": _float(meta, "train_sigma"),
        "eval_sigma": _float(meta, "eval_sigma"),
        "translation_frac": _float(meta, "max_translation_frac"),
        "rotation_deg": _float(meta, "max_rotation_deg"),
        "scale_frac": _float(meta, "max_scale_frac"),
        "epochs": _int(meta, "epochs"),
        "train_count": _int(meta, "train_size"),
        "val_count": _int(meta, "val_size"),
        "final_mae": _float(final, "val_mae"),
        "final_rmse": _float(final, "val_rmse"),
        "final_psnr": _float(final, "val_psnr"),
        "final_ssim": _float(final, "val_ssim"),
        "final_corr": _float(final, "val_corr"),
        "note": note,
        "path": str(run_dir),
    }


def parse_entry(text: str) -> tuple[str, str, str, Path, str]:
    parts = text.split("|", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--entry must be 'label|family|kind|path|note', where kind is metrics_csv or registration_json"
        )
    label, family, kind, path, note = parts
    if kind not in {"metrics_csv", "registration_json"}:
        raise argparse.ArgumentTypeError(f"Unknown entry kind: {kind}")
    return label, family, kind, Path(path), note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", required=True, type=parse_entry)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for label, family, kind, path, note in args.entry:
        if kind == "metrics_csv":
            rows.append(read_metrics_csv(label, family, path, note))
        else:
            rows.append(read_registration_json(label, family, path, note))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
