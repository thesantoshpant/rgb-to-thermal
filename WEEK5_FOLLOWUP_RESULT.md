# Week 5 Follow-Up Result

Date: 2026-06-16

These runs address the audit items raised after the initial Week 5 transfer
matrix.

## Protocol Framing Fix

The old `19.28 dB` local result used raw targets plus ensemble/TTA. Week 5 uses
robust per-sample target normalization and a single supervised-affine model.
Those absolute PSNR values are not directly comparable; Week 5 claims should use
matched-protocol deltas.

## AA -> Kust4K Three-Seed Transfer Check

Locked Week 5 protocol: robust targets, amplified `sigma=0.3`, `lambda_warp_rgb=0.5`,
Ann Arbor train, Kust4K validation.

| Seed | No-reg PSNR | Affine PSNR | Delta | No-reg SSIM | Affine SSIM | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9.902 | 10.306 | +0.404 | 0.386 | 0.415 | +0.028 |
| 7 | 10.052 | 10.559 | +0.507 | 0.390 | 0.413 | +0.023 |
| 123 | 9.712 | 10.223 | +0.511 | 0.394 | 0.416 | +0.022 |
| Mean +/- std | 9.889 | 10.363 | +0.474 +/- 0.061 | 0.390 | 0.415 | +0.024 +/- 0.003 |

Full table: `results/week5_aa_to_kust4k_3seed_summary.csv`.

This is the only Week 5 transfer result that survives a three-seed audit. It is
directionally consistent and clears the `+0.3 dB` threshold.

## Matched-Compute Pretrain/Fine-Tune Control

Seed 42, robust targets, Ann Arbor validation.

| Run | Source epochs | AA epochs | PSNR | SSIM | Pearson r |
|---|---:|---:|---:|---:|---:|
| AA from scratch | 0 | 30 | 15.681 | 0.540 | 0.814 |
| Kust4K+CART pretrain -> AA fine-tune | 20 | 30 | 15.845 | 0.546 | 0.821 |
| AA from scratch matched compute | 0 | 50 | 15.920 | 0.549 | 0.828 |

Full table: `results/week5_matched_compute_control_summary.csv`.

The earlier `+0.165 dB` pretrain/fine-tune gain does not survive the
matched-compute control: from-scratch 50 epochs reaches `15.920 dB`, which is
`+0.075 dB` above the external-pretrain fine-tune run.

## Decision

Week 5 remains complete, with narrower conclusions:

- AA -> Kust4K supervised-affine transfer is real under this protocol
  (`+0.474 +/- 0.061 dB`, three seeds).
- The external-to-external transfer cells remain weak/negative and single-seed;
  do not claim broad cross-dataset generalization.
- The Kust4K+CART pretrain -> AA fine-tune gain should be dropped as a positive
  claim unless future controls show an advantage over matched compute.
- Week 6 can proceed, but the paper framing should stay diagnostic and
  conservative.
