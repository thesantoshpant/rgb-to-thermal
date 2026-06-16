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

Week 5 transfer runs use `lambda_warp_rgb=0.5` for the supervised-affine
variant. This is the locked compromise setting: it remains above the CART
seed-42 threshold (`+0.948 dB`) while avoiding the more loss-dominant `1.0` and
`2.0` settings. The Week 4 default `1.0` remains documented as an ablation, not
the default transfer-matrix setting.

The Kust4K row should be interpreted conservatively:

- AA->Kust4K is still useful as a transfer/null check.
- Kust4K->AA tests whether Kust4K is a useful source domain despite its
  within-dataset null result.
- Kust4K->CART and CART->Kust4K are lower-priority completeness checks.
- The first priority experiment is Kust4K+CART pretraining followed by Ann
  Arbor fine-tuning.

## Transfer Harness Start

Added `--eval-dataset` to `week3_registration_v0.py` so training and validation
can use different dataset roots while keeping the same misalignment protocol and
model code. Knox smoke test passed:

- run: `week5_smoke_transfer_kust4k_to_cart_robust`
- train: Kust4K train, `max_train=16`
- eval: CART val, `max_val=8`
- normalization: `target_normalization=robust`
- result: one epoch completed and wrote metadata/metrics.

The full Week 5 transfer matrix is now complete; see
`WEEK5_TRANSFER_RESULT.md` and `results/week5_transfer_matrix_summary.csv`.
The matrix is weak overall: affine wins 2/4 cells, only Ann Arbor -> Kust4K
clears `+0.3 dB`, and both external-to-external transfer cells are negative.
Follow-up runs in `WEEK5_FOLLOWUP_RESULT.md` show that Ann Arbor -> Kust4K
survives a three-seed audit at `+0.474 +/- 0.061 dB`.

## Week 5 Follow-Up Runs

Ann Arbor robust diagonal control, locked `lambda_warp_rgb=0.5`:

| Seed | No-reg PSNR | Supervised affine PSNR | Delta |
|---:|---:|---:|---:|
| 42 | 15.218 | 15.681 | +0.463 |
| 7 | 15.656 | 15.721 | +0.065 |
| 123 | 15.510 | 15.674 | +0.164 |
| Mean +/- std | 15.461 | 15.692 | +0.230 +/- 0.207 |

This is directionally positive but weaker than the Week 4 raw-target
`lambda_warp_rgb=1.0` result. Under the locked Week 5 protocol, Ann Arbor no
longer clears the old `0.3 dB` threshold.

First-priority pretrain/fine-tune diagnostic:

| Run | Train | Eval | PSNR | Note |
|---|---|---|---:|---|
| from scratch | Ann Arbor | Ann Arbor | 15.681 | robust, seed 42, lambda 0.5 |
| external pretrain only | Kust4K+CART | Ann Arbor | 9.983 | poor direct transfer |
| external pretrain -> AA fine-tune | Ann Arbor | Ann Arbor | 15.845 | +0.165 vs from scratch |

External pretraining gives a small seed-42 initialization gain after Ann Arbor
fine-tuning under the matched robust protocol. Do not compare the absolute
`15.x dB` robust PSNR values directly to the old `19.28 dB` raw-target
ensemble/TTA result; the target normalization and model protocol changed.

Matched-compute follow-up:

| Run | Source epochs | AA epochs | PSNR |
|---|---:|---:|---:|
| AA from scratch | 0 | 30 | 15.681 |
| Kust4K+CART pretrain -> AA fine-tune | 20 | 30 | 15.845 |
| AA from scratch matched compute | 0 | 50 | 15.920 |

The pretrain/fine-tune gain does not survive matched compute; drop it as a
positive claim unless later controls reverse this.

Final transfer matrix result:

| Train -> Eval | No-reg PSNR | Affine PSNR | Delta |
|---|---:|---:|---:|
| Ann Arbor -> Kust4K | 9.902 | 10.306 | +0.404 |
| Kust4K -> Ann Arbor | 10.168 | 10.250 | +0.082 |
| Kust4K -> CART | 10.054 | 9.859 | -0.195 |
| CART -> Kust4K | 9.672 | 9.264 | -0.408 |

Week 5 is complete, but it does not support a robust cross-dataset
registration claim. The one supported transfer result is Ann Arbor -> Kust4K;
the broader cross-dataset and external-pretraining claims remain unsupported.
