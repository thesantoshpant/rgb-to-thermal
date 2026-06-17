# CCAI Short-Version Cut List

This file maps the Week 9 WACV-style draft into a four-page CCAI workshop
version. The short version is a subset, not a separate paper.

## Keep As Core

- Climate motivation from the introduction: aerial thermal proxies for urban
  heat and adaptation workflows.
- Dataset/protocol warning: target representation and alignment conventions
  change the apparent difficulty of RGB-to-thermal translation.
- Main method: ConvNeXt-tiny U-Net plus RGB-only input-space affine head with
  synthetic warp-recovery supervision.
- Primary result: `+0.571 +/- 0.157 dB` over matched no-registration ConvNeXt
  on Ann Arbor under amplified synthetic misalignment.
- Main ablation lesson: uncertainty-weighted reconstruction hurt; the primary
  method is uncertainty-decoupled.
- Conservative external claim: Kust4K does not show a meaningful
  within-dataset registration gain; CART is loss-balance-sensitive.

## Compress

- Related Work: reduce to one paragraph covering paired RGB-to-thermal,
  multimodal registration, and climate remote sensing.
- Dataset table: convert to prose unless space allows a small table.
- Baselines: keep only ConvNeXt no-reg, ConvNeXt affine uncertainty-decoupled,
  Swin-T no-reg, and one weak-baseline sentence for pix2pix/CycleGAN.
- External results: keep only Kust4K null, CART caveat, and Ann Arbor-to-Kust4K
  transfer as a short paragraph.
- Qualitative results: keep one hero/method panel or move all figures to
  appendix depending on page pressure.

## Cut From Main Text

- Full Week 2.5 diagnostic walkthrough.
- Full severity sweep table.
- Detailed Swin-T affine stacking seed discussion.
- Full loss-term ablation table beyond the uncertainty/warp supervision
  message.
- Cross-dataset gallery as a main figure.
- Appendix placeholders from the WACV draft.

## Caption Guardrails

- Do not present the hero `+0.93 dB` example as typical; cite the three-seed
  mean `+0.571 +/- 0.157 dB`.
- Do not imply Kust4K registration benefit from any single qualitative sample.
- Do not call the method unsupervised registration.
- Do not call uncertainty maps calibrated or beneficial for reconstruction.

## CCAI Pitch Sentence

Reliable use of aerial RGB-to-thermal translation for urban heat analysis
requires evaluating not only model architecture, but also alignment, target
normalization, and dataset-specific thermal conventions.
