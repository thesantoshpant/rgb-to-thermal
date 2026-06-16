# Week 2 Preliminary Checkpoint

Date: 2026-06-15

Superseded by `WEEK2_GO_NO_GO_MEMO.md` after the full severity sweep.

This is not the final Week 2 go/no-go memo. It records the first aligned vs
synthetic-misalignment checkpoint after the Week 1 blockers were fixed.

## Setup

- Host: Knox, `/home/spant/UMich/umich-hackathon/rgb2thermal_wacv`
- Script: `week2_pix2pix_baseline.py`
- Python: `../weather_experiments/.venv/bin/python`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Resolution: `256x320`
- Epochs: `20`
- Batch size: `8`
- Eval split: `val`
- Output root: `week2_runs/`

## Results

| Dataset | Sigma | Train | Val | Final PSNR | Best PSNR |
|---|---:|---:|---:|---:|---:|
| Kust4K | 0.0 | 1970 | 283 | 16.354 | 16.354 |
| Kust4K | 0.3 | 1970 | 283 | 16.007 | 16.235 |
| Caltech CART | 0.0 | 1822 | 222 | 16.456 | 16.676 |
| Caltech CART | 0.3 | 1822 | 222 | 16.270 | 16.476 |

## Read

The initial `sigma=0.3` checkpoint does not meet the Week 2 go criterion:

- Kust4K final drop: `0.347 dB`
- Caltech CART final drop: `0.185 dB`

This is only a two-point check. Before making the final decision, run the full
severity sweep at `sigma = {0.1, 0.2, 0.3, 0.5}` on both datasets, plot PSNR vs
severity, and inspect whether the current pix2pix baseline is relying on RGB
structure or mostly learning dataset-level thermal priors.
