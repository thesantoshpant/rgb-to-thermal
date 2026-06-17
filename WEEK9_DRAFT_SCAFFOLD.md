# Week 9 Paper Draft Scaffold

This is the starting scaffold for draft 1. It intentionally narrows the paper
around claims that survived the Week 7/8 audits.

## Working Title

Synthetic Warp Supervision for Robust Aerial RGB-to-Thermal Translation

## Core Claim

A lightweight RGB-only affine registration head, trained with synthetic
warp-recovery supervision, gives a small but repeatable robustness gain for
aerial RGB-to-thermal translation under controlled synthetic misalignment.

This is not a general unsupervised-registration claim and not a broad
cross-dataset generalization claim.

## Abstract Skeleton

RGB-to-thermal translation from aerial imagery is sensitive to camera alignment,
target representation, and dataset conventions. We study this sensitivity on an
Ann Arbor UAV RGB/thermal dataset and two external RGB-TIR datasets, Kust4K and
CART. Starting from a ConvNeXt-tiny U-Net translator, we add a lightweight
RGB-only affine registration head trained with synthetic warp-recovery
supervision. After ablations, the strongest variant decouples uncertainty from
the reconstruction loss and treats uncertainty maps as diagnostics. On Ann
Arbor under amplified synthetic misalignment, the method improves over a matched
no-registration ConvNeXt baseline by `+0.571 +/- 0.157 dB` across three seeds
and outperforms a Swin-T U-Net baseline by `+0.215 +/- 0.113 dB`. External
results are mixed: Kust4K shows no statistically meaningful within-dataset gain,
while CART gains are sensitive to loss balance. These findings support a
conservative conclusion: synthetic warp supervision can improve robustness on
the source dataset, but target normalization and dataset-specific alignment
limit cross-dataset claims.

## Contributions

1. A reproducible multi-dataset evaluation harness for aerial RGB-to-thermal
   translation with Ann Arbor, Kust4K, and CART.
2. A synthetic misalignment diagnostic showing that Ann Arbor is alignment
   sensitive while Kust4K/CART are weaker and target-representation-confounded.
3. A lightweight input-space affine registration module with direct RGB
   warp-recovery supervision.
4. A controlled ablation showing that uncertainty-weighted reconstruction hurts
   in this protocol, while uncertainty-decoupled affine registration gives a
   repeatable but modest gain.
5. Qualitative figures showing both useful behavior and failure modes, including
   cases where the method underperforms the no-registration baseline.

## Section Plan

### 1. Introduction

- Motivate aerial RGB-to-thermal translation for urban heat, infrastructure, and
  UAV sensing.
- State the practical issue: RGB/TIR pairs are often imperfectly aligned, and
  naive translation models can either ignore RGB detail or overfit dataset
  target conventions.
- Frame the paper as an empirical + method study, not as a universal
  registration solution.
- End with the conservative contributions above.

### 2. Related Work

- Paired RGB-to-thermal / RGB-TIR translation.
- Aerial and UAV multimodal datasets.
- Cross-modal registration and spatial transformer-style alignment.
- Uncertainty in image translation, with care to note that our uncertainty
  branch is diagnostic after ablation.

### 3. Datasets and Target Representation

- Ann Arbor: local registered RGB/scalar thermal target.
- Kust4K: official splits, broken-sample exclusions, qualitative-only caution
  for registration benefit.
- CART/Caltech: labeled paired subset, target normalization caveat.
- Explain `raw`, `robust`, and `histmatch`; justify robust for the locked Week
  5-8 protocol.

### 4. Method

- Baseline: ConvNeXt-tiny + U-Net translator.
- Synthetic misalignment generation: translation/rotation/scale, thermal target
  stays fixed.
- Registration module: RGB-only affine predictor, input-space warp, identity
  initialization.
- Training losses: reconstruction, SSIM, edge, affine regularizer, RGB
  warp-recovery supervision.
- Final design choice: uncertainty maps are generated/logged but not used to
  weight reconstruction in the primary model.

### 5. Experiments

- Week 2/2.5 diagnostics: why the initial no-go was not trusted; Ann Arbor
  control establishes alignment sensitivity.
- Week 4/5 registration controls: supervised warp signal is necessary; external
  results are limited.
- Week 6/7 baselines and ablations: ConvNeXt no-reg, Swin-T no-reg, Swin-T
  affine, uncertainty on/off, loss ablations, severity diagnostics.
- Week 8 qualitative figures: hero, multi-sigma recovery, failure cases,
  cross-dataset qualitative context.

### 6. Discussion and Limitations

- Gain is real but modest.
- Synthetic warp supervision is not real-world registration ground truth.
- Kust4K does not support a positive registration-benefit claim across seeds.
- CART gains are partly loss-balance-sensitive.
- Qualitative figures show smoothing of high-frequency thermal structure.
- Uncertainty maps are not calibrated enough to drive reconstruction weighting.

## Figure and Table Map

| Slot | Artifact | Status |
|---|---|---|
| Figure 1 | Hero Ann Arbor scene | `figures/week8/hero_ann_arbor_seed42.png` |
| Figure 2 | Method diagram | TODO in Week 9/11 |
| Figure 3 | Multi-sigma recovery | `figures/week8/misalignment_recovery_multisigma_seed42.png` |
| Figure 4 | Failure cases | `figures/week8/failure_cases_ann_arbor_seed42.png` |
| Appendix Fig. A | Candidate grid | `figures/week8/ann_arbor_candidate_grid_seed42.png` |
| Appendix Fig. B | Cross-dataset qualitative gallery | `figures/week8/cross_dataset_gallery_seed42.png` |
| Table 1 | Dataset counts and normalization | TODO |
| Table 2 | Baselines: ConvNeXt, Swin-T, pix2pix, CycleGAN | use Week 6/7 CSVs |
| Table 3 | Main ablation: no-reg vs affine + uncertainty vs uncertainty-decoupled affine | use Week 7 CSV |
| Table 4 | External/cross-dataset results | use Week 5, with Kust4K caveat |

## Claim Guardrails

- Do not claim Kust4K registration benefit. The three-seed within-dataset gain
  is `+0.096 +/- 0.067 dB`, consistent with no effect.
- Do not claim the hero `+0.93 dB` is typical. The main method gain is
  `+0.571 +/- 0.157 dB` across three seeds.
- Do not describe the final method as deterministic. Use
  "uncertainty-decoupled" or "unweighted-L1 primary variant."
- Do not claim uncertainty helps reconstruction. It hurt in the locked
  protocol and should be described as diagnostic only.
- Do not report the cross-dataset gallery as a unified quantitative protocol.
  It mixes the locked Ann Arbor checkpoint with older within-dataset external
  checkpoints.
- Do not use the single-seed severity curve as a main statistical claim.

## Immediate Week 9 Tasks

- Convert this scaffold into a full prose draft.
- Build the dataset-count and main-ablation tables from committed CSVs.
- Draft figure captions with the guardrails above.
- Create a simple method diagram.
- Run a no-overclaiming pass on the abstract before sending to collaborators.
