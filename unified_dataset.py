#!/usr/bin/env python3
"""Unified RGB->thermal dataset adapters for Ann Arbor, Caltech CART, and Kust4K-like roots.

The dataset returns the Week 1 contract:
    (rgb_tensor, thermal_tensor, scalar_target, dataset_tag, alignment_quality_score)

Tensors are float32 in [0, 1], channel-first, and resized to 512x640 by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


DEFAULT_SIZE = (512, 640)  # height, width
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class PairRecord:
    rgb_path: Path
    thermal_path: Path | None
    scalar_path: Path | None
    dataset_tag: str
    alignment_quality: float = 1.0
    split: str = "all"


def _stable_split(key: str, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def _iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _find_first_dir(root: Path, candidates: Iterable[str]) -> Path | None:
    ordered = [c.lower() for c in candidates]
    lowered = set(ordered)
    found: dict[str, Path] = {}
    for p in chain([root], root.rglob("*")):
        if p.is_dir():
            name = p.name.lower()
            if name in lowered and name not in found:
                found[name] = p
    for name in ordered:
        if name in found:
            return found[name]
    return None


def _read_manifest(root: Path, manifest: Path, dataset_tag: str, split: str) -> list[PairRecord]:
    records: list[PairRecord] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = row.get("split") or _stable_split(row.get("rgb", ""))
            if split != "all" and row_split != split:
                continue
            rgb = root / (row.get("rgb") or row.get("rgb_path") or row.get("color") or "")
            thermal = root / (row.get("thermal") or row.get("thermal_path") or row.get("ir") or "")
            scalar = row.get("scalar") or row.get("scalar_path")
            quality = float(row.get("alignment_quality") or row.get("quality") or 1.0)
            if rgb.exists() and thermal.exists():
                records.append(
                    PairRecord(
                        rgb_path=rgb,
                        thermal_path=thermal,
                        scalar_path=(root / scalar) if scalar else None,
                        dataset_tag=dataset_tag,
                        alignment_quality=quality,
                        split=row_split,
                    )
                )
    return records


def _read_split_txt(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().rsplit(".", 1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _split_map_from_txts(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for stem in _read_split_txt(root / f"{split_name}.txt"):
            out[stem] = split_name
    return out


def _match_thermal(rgb_path: Path, thermal_by_name: dict[str, Path], thermal_by_stem: dict[str, Path]) -> Path | None:
    name = rgb_path.name
    stem = rgb_path.stem
    candidates = [
        name,
        name.replace("eo", "thermal"),
        name.replace("color", "thermal"),
        name.replace("rgb", "thermal"),
        name.replace("RGB", "thermal"),
        f"{stem}.png",
        f"{stem}.jpg",
        f"{stem}.tif",
        f"{stem}.tiff",
    ]
    for cand in candidates:
        if cand in thermal_by_name:
            return thermal_by_name[cand]
    stem_candidates = [
        stem,
        stem.replace("eo", "thermal"),
        stem.replace("color", "thermal"),
        stem.replace("rgb", "thermal"),
        stem.replace("RGB", "thermal"),
    ]
    for cand in stem_candidates:
        if cand in thermal_by_stem:
            return thermal_by_stem[cand]
    return None


def discover_caltech(root: Path, split: str = "all", manifest: Path | None = None) -> list[PairRecord]:
    """Discover Caltech CART pairs from a labeled_rgbt_pairs extraction."""
    if manifest:
        return _read_manifest(root, manifest, "caltech_cart", split)
    pair_root = root / "labeled_rgbt_pairs" if (root / "labeled_rgbt_pairs").exists() else root
    rgb_dir = _find_first_dir(pair_root, ["color", "rgb", "images", "eo"])
    thermal_dir = _find_first_dir(pair_root, ["thermal16", "thermal8", "thermal", "ir"])
    if not rgb_dir or not thermal_dir:
        return []
    thermal_files = _iter_images(thermal_dir)
    by_name = {p.name: p for p in thermal_files}
    by_stem = {p.stem: p for p in thermal_files}
    records: list[PairRecord] = []
    for rgb_path in _iter_images(rgb_dir):
        thermal_path = _match_thermal(rgb_path, by_name, by_stem)
        if not thermal_path:
            continue
        row_split = _stable_split(str(rgb_path.relative_to(pair_root)))
        if split == "all" or split == row_split:
            records.append(PairRecord(rgb_path, thermal_path, None, "caltech_cart", 1.0, row_split))
    return records


def discover_kust4k(root: Path, split: str = "all", manifest: Path | None = None) -> list[PairRecord]:
    """Discover Kust4K-style aligned RGB/Thermal folders.

    The public source/name still needs confirmation, so this intentionally accepts a generic
    layout with rgb/color/visible and thermal/ir/lwir folders, or a CSV manifest.
    """
    if manifest:
        return _read_manifest(root, manifest, "kust4k", split)
    rgb_dir = _find_first_dir(root, ["rgb", "color", "visible", "vis", "images"])
    thermal_dir = _find_first_dir(root, ["thermal", "ir", "tir", "lwir", "thermal8", "thermal16"])
    if not rgb_dir or not thermal_dir:
        return []
    thermal_files = _iter_images(thermal_dir)
    by_name = {p.name: p for p in thermal_files}
    by_stem = {p.stem: p for p in thermal_files}
    split_by_stem = _split_map_from_txts(root)
    records: list[PairRecord] = []
    for rgb_path in _iter_images(rgb_dir):
        thermal_path = _match_thermal(rgb_path, by_name, by_stem)
        if not thermal_path:
            continue
        row_split = split_by_stem.get(rgb_path.stem, _stable_split(str(rgb_path.relative_to(root))))
        if split == "all" or split == row_split:
            records.append(PairRecord(rgb_path, thermal_path, None, "kust4k", 1.0, row_split))
    return records


def discover_ann_arbor(cache_root: Path, split: str = "all") -> list[PairRecord]:
    split_path = cache_root / "split.json"
    rgb_dir = cache_root / "reg_rgb"
    scalar_dir = cache_root / "scalar"
    if not split_path.exists() or not rgb_dir.exists() or not scalar_dir.exists():
        return []
    split_map = json.loads(split_path.read_text(encoding="utf-8"))
    meta_path = cache_root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    qualities = meta.get("meta", {})
    names: list[tuple[str, str]] = []
    for row_split, row_names in split_map.items():
        if split == "all" or split == row_split:
            names.extend((row_split, n.replace(".JPG", "")) for n in row_names)
    records: list[PairRecord] = []
    for row_split, name in names:
        rgb_path = rgb_dir / f"{name}.png"
        scalar_path = scalar_dir / f"{name}.npy"
        if not rgb_path.exists() or not scalar_path.exists():
            continue
        q = float(qualities.get(f"{name}.JPG", {}).get("quality", 1.0))
        records.append(PairRecord(rgb_path, None, scalar_path, "ann_arbor", q, row_split))
    return records


def _load_rgb(path: Path, size_hw: tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _load_gray_image(path: Path, size_hw: tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = np.asarray(img.convert("L"))
    orig_dtype = arr.dtype
    arr = arr.astype(np.float32)
    if arr.size == 0:
        arr = np.zeros((h, w), dtype=np.float32)
    elif arr.max() > arr.min():
        if orig_dtype == np.uint8:
            arr = arr / 255.0
        elif np.issubdtype(orig_dtype, np.integer) and arr.max() > 255:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        elif np.issubdtype(orig_dtype, np.floating) and 0.0 <= arr.min() and arr.max() <= 1.0:
            arr = arr
        else:
            arr = arr / 255.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    img01 = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    out = np.asarray(img01, dtype=np.float32) / 255.0
    return torch.from_numpy(out[None])


def _load_scalar(path: Path, size_hw: tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(out[None])


class UnifiedR2TDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[PairRecord],
        size_hw: tuple[int, int] = DEFAULT_SIZE,
    ):
        self.records = records
        self.size_hw = size_hw

    @classmethod
    def from_roots(
        cls,
        ann_arbor_cache: str | os.PathLike[str] | None = None,
        caltech_root: str | os.PathLike[str] | None = None,
        kust4k_root: str | os.PathLike[str] | None = None,
        caltech_manifest: str | os.PathLike[str] | None = None,
        kust4k_manifest: str | os.PathLike[str] | None = None,
        split: str = "train",
        size_hw: tuple[int, int] = DEFAULT_SIZE,
    ) -> "UnifiedR2TDataset":
        records: list[PairRecord] = []
        if ann_arbor_cache:
            records.extend(discover_ann_arbor(Path(ann_arbor_cache), split))
        if caltech_root:
            records.extend(discover_caltech(Path(caltech_root), split, Path(caltech_manifest) if caltech_manifest else None))
        if kust4k_root:
            records.extend(discover_kust4k(Path(kust4k_root), split, Path(kust4k_manifest) if kust4k_manifest else None))
        return cls(records, size_hw=size_hw)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        rgb = _load_rgb(rec.rgb_path, self.size_hw)
        if rec.scalar_path is not None:
            scalar = _load_scalar(rec.scalar_path, self.size_hw)
            thermal = scalar.clone()
        elif rec.thermal_path is not None:
            thermal = _load_gray_image(rec.thermal_path, self.size_hw)
            scalar = thermal.clone()
        else:
            raise FileNotFoundError(f"No thermal/scalar target for {rec.rgb_path}")
        quality = torch.tensor(float(rec.alignment_quality), dtype=torch.float32)
        return rgb, thermal, scalar, rec.dataset_tag, quality

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.records:
            out[rec.dataset_tag] = out.get(rec.dataset_tag, 0) + 1
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-arbor-cache", default=os.environ.get("R2T_CACHE"))
    ap.add_argument("--caltech-root")
    ap.add_argument("--kust4k-root")
    ap.add_argument("--caltech-manifest")
    ap.add_argument("--kust4k-manifest")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--height", type=int, default=DEFAULT_SIZE[0])
    ap.add_argument("--width", type=int, default=DEFAULT_SIZE[1])
    ap.add_argument("--check-first", action="store_true")
    args = ap.parse_args()

    ds = UnifiedR2TDataset.from_roots(
        ann_arbor_cache=args.ann_arbor_cache,
        caltech_root=args.caltech_root,
        kust4k_root=args.kust4k_root,
        caltech_manifest=args.caltech_manifest,
        kust4k_manifest=args.kust4k_manifest,
        split=args.split,
        size_hw=(args.height, args.width),
    )
    print(json.dumps({"total": len(ds), "by_dataset": ds.summary()}, indent=2))
    if args.check_first and len(ds):
        rgb, thermal, scalar, tag, quality = ds[0]
        print(
            json.dumps(
                {
                    "first_tag": tag,
                    "rgb_shape": list(rgb.shape),
                    "thermal_shape": list(thermal.shape),
                    "scalar_shape": list(scalar.shape),
                    "alignment_quality": float(quality),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
