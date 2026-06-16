# Week 5 Preflight

Date: 2026-06-16

## Why This Exists

The Week 4 audit correctly narrowed the claim. The current method is not
unsupervised registration discovering the RGB/TIR alignment bottleneck. The
working mechanism is synthetic warp augmentation plus direct RGB warp-recovery
supervision.

Before cross-dataset transfer experiments, Week 5 must check three things:

- Kust4K and CART seed stability for no-registration vs supervised affine.
- CART sensitivity to `lambda_warp_rgb`.
- Target-normalization convention for external datasets.

## Target Normalization

Implemented in `unified_dataset.py`:

- `raw`: current behavior, default for reproducibility.
- `robust`: per-sample 1st-to-99th percentile scaling.
- `histmatch`: map a dataset's target quantiles to Ann Arbor reference quantiles.

Stats builder:

- `build_target_normalization_stats.py`
- output: `results/week5_target_normalization_stats.json`

Validation split audit, edge mean:

| Mode | Ann Arbor | Kust4K | CART |
|---|---:|---:|---:|
| raw | 0.0330 | 0.0196 | 0.0166 |
| robust | 0.0345 | 0.0295 | 0.0257 |
| histmatch | 0.0330 | 0.0284 | 0.0215 |

Week 5 choice: use `robust` for external target-normalized experiments because
it is the most edge-preserving immediate fix. Keep `histmatch` available as an
ablation, but do not make it the default until it improves CART edge content.

## Seed Confirmation

| Dataset | Seed | No-reg PSNR | Supervised affine PSNR | Delta |
|---|---:|---:|---:|---:|
| Kust4K | 42 | 19.031 | 19.178 | +0.147 |
| Kust4K | 7 | 18.993 | 19.013 | +0.020 |
| Kust4K | 123 | 18.995 | 19.115 | +0.120 |
| CART | 42 | 20.068 | 21.264 | +1.196 |
| CART | 7 | 20.203 | 20.857 | +0.654 |
| CART | 123 | 20.580 | 21.074 | +0.494 |

Kust4K three-seed delta: `+0.096 +/- 0.067 dB`. This fails the
`0.3 dB` registration-help threshold and should not be framed as a positive
Kust4K method result unless a later target-normalized experiment changes the
outcome.

CART three-seed delta: `+0.782 +/- 0.368 dB`. This passes the threshold on
average, but the variance is high enough that CART should be described as
positive but loss-sensitive.

## Lambda Sweep

CART seed-42 sweep:

| lambda_warp_rgb | CART PSNR | Delta vs no-reg |
|---:|---:|---:|
| 0.1 | 20.290 | +0.222 |
| 0.5 | 21.016 | +0.948 |
| 1.0 | 21.264 | +1.196 |
| 2.0 | 21.324 | +1.257 |

The CART gain is loss-balance-sensitive. At `lambda_warp_rgb=0.1`, the gain
falls below the `0.3 dB` threshold; stronger warp-recovery supervision restores
the gain. This supports the audit's concern that CART's large Week 4 result is
not clean evidence of unsupervised registration. The honest mechanism remains
synthetic warp recovery as auxiliary supervision.

## Decision

Week 5 transfer may proceed only as a diagnostic, using `robust` target
normalization and conservative language:

- Do not claim Kust4K benefits from supervised affine registration under the
  current raw-target protocol.
- CART is positive over three seeds, but the effect depends strongly on
  `lambda_warp_rgb`.
- Cross-dataset experiments should test whether robust target normalization
  improves transfer stability, not whether the original WACV bottleneck claim
  is already proven.

## Transfer Harness Start

Added `--eval-dataset` to `week3_registration_v0.py` so training and validation
can use different dataset roots while keeping the same misalignment protocol and
model code. Knox smoke test passed:

- run: `week5_smoke_transfer_kust4k_to_cart_robust`
- train: Kust4K train, `max_train=16`
- eval: CART val, `max_val=8`
- normalization: `target_normalization=robust`
- result: one epoch completed and wrote metadata/metrics.

The full Week 5 transfer matrix is not run yet.
