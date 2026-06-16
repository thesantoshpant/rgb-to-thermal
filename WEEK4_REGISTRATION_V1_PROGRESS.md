# Week 4 Registration v1 Result

Date: 2026-06-16

## Question

Can a learned registration path beat the same ConvNeXt+U-Net translator without
registration under amplified synthetic misalignment?

## Protocol

All decision runs use:

- `train_sigma=0.3`, `eval_sigma=0.3`
- `max_translation_frac=0.20`, `max_rotation_deg=20`,
  `max_scale_frac=0.25`
- validation split evaluation
- same model, seed, resolution, epoch count, and batch size within each
  no-registration vs registration pair

Ann Arbor uses `30` epochs, `bs=6`, `res=256`. Kust4K/CART use full train/val
splits for `20` epochs, `bs=8`, `res=256`.

## Architecture Sweep

Seed-42 Ann Arbor runs showed that unsupervised warp capacity was not enough:

| Run | Mechanism | PSNR | Warp metric | Delta vs no-reg |
|---|---|---:|---:|---:|
| `week3_no_registration_ann_arbor_sigma03_amp_seed42` | no registration | 15.886 | 0.000 | 0.000 |
| `week3_reg_shared_rgb_ann_arbor_sigma03_amp_seed42` | shared feature-space affine | 15.814 | theta_l2 0.036 | -0.072 |
| `week4_input_rgb_affine_ann_arbor_sigma03_amp_seed42` | input-space RGB affine | 15.772 | theta_l2 0.013 | -0.114 |
| `week4_input_rgb_flow_ann_arbor_sigma03_amp_seed42` | input-space dense flow | 15.547 | flow_l2 0.0066 | -0.339 |
| `week4_input_rgb_flow_edge1_ann_arbor_sigma03_amp_seed42` | dense flow, stronger edge loss | 15.384 | flow_l2 0.0214 | -0.502 |
| `week4_input_rgb_affine_warprgb1_ann_arbor_sigma03_amp_seed42` | input-space affine + synthetic RGB warp supervision | 16.199 | theta_l2 0.038 | +0.313 |

The important design result is negative for unsupervised affine/flow and
positive for synthetic RGB warp supervision. Blindly adding warp capacity did
not help; adding direct known-warp supervision did.

## Three-Seed Ann Arbor Check

| Seed | No-reg PSNR | Supervised affine PSNR | Delta |
|---:|---:|---:|---:|
| 42 | 15.886 | 16.199 | +0.313 |
| 7 | 16.172 | 16.349 | +0.177 |
| 123 | 15.765 | 16.202 | +0.437 |
| Mean +/- std | 15.941 | 16.250 | +0.309 +/- 0.130 |

This meets the Week 4 Ann Arbor bar, but the gain is still moderate and should
be treated as a synthetic-supervised signal, not a final real-world registration
claim.

## External Same-Dataset Comparisons

| Dataset | Train / Val | No-reg PSNR | Supervised affine PSNR | Delta | RGB warp MAE change |
|---|---:|---:|---:|---:|---:|
| Kust4K | 1970 / 283 | 19.031 | 19.178 | +0.147 | 0.0888 -> 0.0574 |
| Caltech CART | 1822 / 222 | 20.068 | 21.264 | +1.196 | 0.0888 -> 0.0582 |

The external result is mixed but useful. CART gives a strong win, Kust4K gives a
small positive win, and both show the registration head is actually reducing the
RGB warp error.

## Target-Normalization Decision

The target-normalization audit found Kust4K/CART raw grayscale targets have much
lower edge energy than Ann Arbor palette-inverted scalar targets. For Week 4,
the conservative choice is:

- report only within-dataset no-registration vs registration comparisons;
- do not compare PSNR-drop magnitudes across Ann Arbor, Kust4K, and CART as if
  the target conventions were identical;
- keep Kust4K/CART target normalization as a Week 5 prerequisite before
  cross-dataset transfer claims.

## Decision

Week 4 is complete and passes the continuation criterion with caveats:

- learned registration beats fixed-crop/no-registration by at least `0.3 dB` on
  two datasets: Ann Arbor mean `+0.309 dB`, CART `+1.196 dB`;
- Kust4K is positive but below the `0.3 dB` threshold at `+0.147 dB`;
- the primary Week 5 mechanism should be input-space affine with synthetic RGB
  warp supervision, likely as pretraining or an auxiliary loss;
- target-conditioned registration remains an internal oracle only because it
  mostly predicts identity.

See `results/week4_registration_v1_summary.csv` for the full machine-readable
run table.
