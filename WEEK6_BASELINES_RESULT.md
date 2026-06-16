# Week 6 Baselines Result

Date: 2026-06-16

## Protocol

Main fair-compute table:

- dataset: Ann Arbor train -> Ann Arbor val
- target normalization: `robust`
- synthetic misalignment: `train_sigma=0.3`, `eval_sigma=0.3`
- amplified range: translation `0.20`, rotation `20 deg`, scale `0.25`
- resolution: `256 x 320`
- seed: `42`
- training budget: `50` epochs, `336` train samples, `41` validation samples

Full machine-readable table: `results/week6_baseline_summary.csv`.

The legacy `19.28 dB` ensemble+TTA result remains valid only for the older
raw-target local protocol. It is documented as a reproduced legacy result, not
included as a same-y-axis row in this robust-normalized Week 6 table.

The pretrained-backbone row uses the available pretrained timm Swin-T image
encoder with a U-Net decoder. It is a transformer-backbone baseline, not a true
SwinIR/Restormer restoration checkpoint.

## Baseline Table

| Method | Family | PSNR | MAE | SSIM | Pearson r | Note |
|---|---|---:|---:|---:|---:|---|
| CycleGAN | unpaired | 8.598 | 0.302 | 0.269 | 0.180 | unpaired, no paired L1 |
| pix2pix | paired GAN | 11.835 | 0.183 | - | - | Week 2 pix2pix harness |
| Small U-Net L1 | paired L1 | 12.714 | 0.177 | - | - | pix2pix generator, L1 only |
| ConvNeXt+U-Net | paired regression | 15.637 | 0.105 | 0.536 | 0.812 | no registration |
| Ours supervised affine | registration | 15.920 | 0.099 | 0.549 | 0.828 | Week 5 method, matched compute |
| Swin-T+U-Net | pretrained transformer | 16.123 | 0.096 | 0.566 | 0.834 | pretrained timm Swin-T encoder |

## Decision

Week 6 is complete as a seed-42 matched-compute baseline pass, but it weakens
the method story:

- the learned supervised-affine method beats ConvNeXt+U-Net by `+0.284 dB`;
- the pretrained Swin-T+U-Net baseline beats the supervised-affine method by
  `+0.203 dB`;
- the unpaired CycleGAN baseline is not competitive;
- pix2pix and small L1 U-Net are far below the pretrained/backbone baselines.

The current paper cannot claim state-of-the-art performance under the Week 6
robust protocol. The defensible claim is narrower: synthetic warp supervision
adds a modest gain over the matched ConvNeXt+U-Net baseline, but a stronger
pretrained transformer backbone is currently better. Week 7 should prioritize:

- adding the supervised-affine registration head on top of the Swin-T backbone,
  or
- reframing the registration result as an ablation/diagnostic rather than the
  headline method.
