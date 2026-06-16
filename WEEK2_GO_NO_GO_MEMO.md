# Week 2 Go/No-Go Memo

Date: 2026-06-15

## Decision

**NO-GO for the WACV registration-bottleneck story as currently tested.**

The Week 2 criterion was: PSNR should drop by at least `2 dB` at
`sigma = 0.3` on at least one external dataset. It did not. The drop was below
`1 dB` on both datasets.

## Experiment

- Model: small pix2pix baseline from `week2_pix2pix_baseline.py`
- Training perturbation: synthetic RGB-only affine misalignment
- Evaluation: aligned validation pairs
- Resolution: `256x320`
- Epochs: `20`
- Datasets: usable Kust4K and Caltech CART supervised pairs
- Knox output roots: `week2_runs/` for `sigma=0.0,0.3`; `week2_sweep_runs/`
  for `sigma=0.1,0.2,0.5`

## Results

| Dataset | Sigma 0.0 | Sigma 0.1 | Sigma 0.2 | Sigma 0.3 | Sigma 0.5 | Drop at 0.3 |
|---|---:|---:|---:|---:|---:|---:|
| Kust4K | 16.354 | 16.077 | 15.689 | 16.007 | 15.408 | 0.347 |
| Caltech CART | 16.456 | 16.409 | 15.762 | 16.270 | 15.989 | 0.185 |

Best-epoch PSNR shows the same conclusion:

| Dataset | Sigma 0.0 | Sigma 0.3 | Drop at 0.3 |
|---|---:|---:|---:|
| Kust4K | 16.354 | 16.235 | 0.119 |
| Caltech CART | 16.676 | 16.476 | 0.200 |

## Interpretation

This does not support the claim that alignment uncertainty is the dominant
bottleneck for this pix2pix setup. The model may be relying heavily on
dataset-level thermal priors, or the current training-only misalignment
protocol may be too weak to expose the failure mode.

Before fully pivoting the paper, run two small diagnostics:

1. Evaluate trained aligned models with RGB misalignment applied at validation
   time, to test sensitivity when the model cannot adapt during training.
2. Train/evaluate a shuffled-RGB control, to measure how much PSNR is possible
   from thermal priors alone.

Unless those diagnostics reverse the conclusion, follow the plan's no-go branch:
re-scope toward a workshop-first contribution around palette inversion,
multi-dataset RGB-to-thermal benchmarking, and failure analysis.

