# Week 3 Registration v0 Result

Date: 2026-06-15

## What Landed

`week3_registration_v0.py` implements the first learned registration sanity
check:

- synthetic RGB-only misalignment on Ann Arbor;
- a small registration head that consumes misaligned RGB plus scalar thermal
  target and predicts an affine warp plus uncertainty map;
- a shared RGB feature-pyramid variant that predicts affine + uncertainty from
  the same ConvNeXt encoder features used by the decoder;
- warping before the existing ConvNeXt+U-Net translator from `train_a1.py`;
- losses for uncertainty-weighted reconstruction, RGB/thermal edge alignment,
  affine identity regularization, and uncertainty smoothness.

## Ann Arbor Passes

Target-conditioned sanity run:

`/home/spant/UMich/umich-hackathon/rgb2thermal_wacv/week3_runs/week3_reg_v0_ann_arbor_sigma03_amp_seed42`

Shared RGB feature run:

`/home/spant/UMich/umich-hackathon/rgb2thermal_wacv/week3_runs/week3_reg_shared_rgb_ann_arbor_sigma03_amp_seed42`

Settings:

- `train_sigma=0.3`, `eval_sigma=0.3`
- `max_translation_frac=0.20`, `max_rotation_deg=20`, `max_scale_frac=0.25`
- `res=256`, `epochs=30`, `batch_size=6`
- train/val: `336 / 41`

Final validation metrics:

| Architecture | MAE | PSNR | SSIM | Corr | theta_l2 | uncertainty |
|---|---:|---:|---:|---:|---:|---:|
| target-conditioned v0 | 0.1043 | 15.452 | 0.544 | 0.787 | 0.0139 | 1.085 |
| shared RGB feature v0 | 0.1028 | 15.814 | 0.549 | 0.805 | 0.0360 | 0.922 |

## Read

This is a successful Week 3 result: both learned affine correction variants
train stably on the Ann Arbor amplified misalignment task, and the shared RGB
feature variant gives a deployable path that no longer requires thermal/target
input at inference.

This is still v0, not the final paper architecture. Week 4 should decide whether
affine is enough or whether TPS/dense flow is needed.

## Open Items

- Audit/improve Kust4K and CART thermal target normalization before using them
  for final cross-dataset claims.
- Compare against fixed-crop/no-registration baselines under the same amplified
  validation protocol.
- Add qualitative warp/uncertainty visualizations in Week 4 or Week 8.
