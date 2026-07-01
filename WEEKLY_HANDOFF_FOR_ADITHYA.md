# RGB-to-Thermal Project — Complete Handoff for Adithya

**From:** Santosh
**For:** Adithya — full review pass + paper-writing partner
**Date:** 2026-06-18
**Repo:** `rgb-to-thermal` (local; not yet pushed). 14 commits ahead of `origin/main` (plus this one).
**Status:** Week 11 polish done locally. Paper draft ready for your paragraph-by-paragraph review. Week 12 (WACV submission, Aug 28 2026 AoE deadline) is the next step.

---

## TL;DR — where we are right now

We finished the hackathon in March 2026 with the ResNet18+FiLM + ThermalGen model that hit **19.28 dB PSNR** on the official 202 unseen test images (2nd place, "Game of Drones"). After Prof. Siwo asked us to push toward publication, we did a full from-scratch rebuild that found two foundational data issues the hackathon pipeline missed, then ran **eleven weeks of disciplined experiments** with weekly Claude Code audits to lock in a defensible paper-grade method.

The headline result, after seven weeks of registration experiments and three-seed audits:

> Our final ConvNeXt-tiny + UNet + RGB-only affine head + synthetic warp-recovery supervision + uncertainty-decoupled reconstruction gives a **+0.571 ± 0.157 dB three-seed gain** over a matched no-registration baseline on Ann Arbor under amplified synthetic misalignment, and **+0.215 ± 0.113 dB** above a Swin-T U-Net no-registration baseline. External datasets (Kust4K, CART) do not support a broad cross-dataset registration claim.

Paper draft is locked, 42-entry bibliography in place, figures have honest caveats baked in, no-overclaiming checklist guards against the things audits caught us nearly doing.

**What I need from you:** paragraph-by-paragraph review of `paper_draft/main_draft.md` using the marks in `paper_draft/review_packet.md` (Clear / Unclear / Wrong / Too strong / Needs citation). Not asking for line edits yet — claim-shape first.

---

## Table of Contents

1. [Project context](#1-project-context)
2. [The two foundational data discoveries](#2-the-two-foundational-data-discoveries)
3. [Week-by-week, with full numbers](#3-week-by-week-with-full-numbers)
4. [The locked Week 7 method](#4-the-locked-week-7-method)
5. [Headline numbers — single reference table](#5-headline-numbers--single-reference-table)
6. [Full ablation tables](#6-full-ablation-tables)
7. [Datasets — complete documentation](#7-datasets--complete-documentation)
8. [Architecture details](#8-architecture-details)
9. [Loss equations in full](#9-loss-equations-in-full)
10. [Reproducing key experiments — exact commands](#10-reproducing-key-experiments--exact-commands)
11. [Complete repo file inventory](#11-complete-repo-file-inventory)
12. [Full CSV inventory with column descriptions](#12-full-csv-inventory-with-column-descriptions)
13. [Bibliography — all 42 entries](#13-bibliography--all-42-entries)
14. [How to write each paper section](#14-how-to-write-each-paper-section)
15. [Anti-overclaim phrasebook](#15-anti-overclaim-phrasebook)
16. [Reviewer Q&A preparation](#16-reviewer-qa-preparation)
17. [Submission checklist](#17-submission-checklist)
18. [Glossary](#18-glossary)
19. [What I'm asking from you](#19-what-im-asking-from-you)

---

## 1. Project context

### 1.1 Origin

- **Hackathon (March 2026):** UMich Heat Resilience Hackathon, hosted by Prof. Geoffrey H. Siwo. We submitted a ResNet18+FiLM refiner on top of a frozen ThermalGen generator. Team placed 2nd. Best v2 ensemble metrics on the internal held-out set (from `results/leaderboard_v2.csv` row `[ens_weighted]`): **PSNR 19.28 dB, SSIM 0.712, CLPIPS 0.364**. Official 202-hidden-test ensemble (`results/leaderboard_official.csv`): PSNR 19.11, SSIM 0.705, CLPIPS 0.350.
- **Prof. Siwo's follow-up:** two asks. (1) Weather-perturbation probe for a Detroit deployment (separate `weather_experiments/` track — outside this paper). (2) Push the work to peer review.
- **Publication path:**
  - **Primary:** WACV 2027 main conference. R2 register Aug 21, **submit Aug 28 2026 AoE**, decisions Oct 9, camera-ready Nov 2, conference Jan 5-9 2027.
  - **Safety net:** CCAI @ NeurIPS 2026 workshop, December 2026. CCAI 2026 CFP not posted yet as of June 18. Planning date Aug 29 2026 AoE.

### 1.2 Team and roles

- **Santosh Pant (lead):** code, experiments, audit cadence, draft writing.
- **Adithya (collaborator):** hackathon co-author; now reviewing the draft and co-authoring the paper.
- **Prof. Geoffrey H. Siwo (senior author):** mentor; framing review; institutional / climate-and-health expertise.

### 1.3 Multi-AI verification workflow

Every week ends with a Claude Code deep audit of the prior week's claims and code. Significant plan decisions cross-checked with `codex` CLI and `gemini` CLI. Each week ends with a written memo (`WEEK<n>_*_RESULT.md`). The audit cadence has overturned weak findings multiple times (Week 2 NO-GO, Week 6 "Swin-T wins" headline, Week 8 K4K overclaim risk).

---

## 2. The two foundational data discoveries

These are the most important things in the project. Everything downstream rests on them.

### 2.1 Discovery 1: target was a colormapped JPEG, not a thermal scalar

The thermal images in `data/Train_2/Thermal/*.JPG` are a single colormapped palette (DJI-style false-color thermal), **not** a clean inferno colormap and **not** a raw 16-bit thermal scalar. The hackathon pipeline trained on the **red channel** of this colormapped JPEG, which is not monotonic in temperature — the same red intensity appears in both cool-purple and hot-yellow regions.

PCA of the thermal pixels shows the colors lie on one curve (PC1 ≈ 0.76 of variance, thickened by JPEG compression). A learned 1-D LUT built from the principal curve plus quantile binning recovers the true scalar heat field with a residual of **~5/255** vs matplotlib inferno's ~42.

**Code:** `data_prep.py` produces `lut.npy` and `to_scalar()` / `to_color()` helpers. These live in `r2t_common.py`.

### 2.2 Discovery 2: RGB and thermal are not spatially aligned

| Sensor | Resolution | Aspect |
|---|---|---|
| RGB camera | 4000 × 3000 (12 MP) | 4:3 |
| Thermal camera | 640 × 512 | 5:4 |

The thermal sensor has a narrower field of view than the RGB sensor. Edge-correlation between RGB-grayscale and thermal at native framing is ≈ 0. Preprocessing (in `data_prep.py`) uses a two-stage registration: (1) `estimate_global_c` finds the best global crop-width fraction `c` (measured mean edge-NCC ~0.65) by maximizing mean thermal-edge NCC across the dataset; (2) `refine_offset` then does a **per-image** small-translation search over (fx, fy) ∈ {−0.08, −0.04, 0, 0.04, 0.08}² around that center crop, choosing the offset that maximizes edge-NCC against the recovered thermal scalar for that image. Each image's registered RGB is saved to `data_cache/reg_rgb/<name>.png`. This target-informed preprocessing is a documented protocol choice; the paper should describe it as such in §7.

### 2.3 Why these matter

Before these fixes, any RGB-to-thermal model was being trained on:

1. A target channel that does not monotonically encode temperature.
2. Spatially misaligned pixel correspondences.

Pixel-loss metrics (PSNR, MAE) were effectively measuring the wrong thing. The data fixes alone account for a large fraction of the practical accuracy improvement — model architecture choices became second-order. **This is the strongest empirical contribution of the paper.**

---

## 3. Week-by-week, with full numbers

### Week 1 — Data foundation

**Goal:** make all three datasets loadable through one consistent interface.

**Done:**
- `unified_dataset.py` returns `(rgb, thermal, scalar, dataset_tag, quality)` for Ann Arbor, Kust4K, and Caltech CART.
- Ann Arbor: 336 train / 41 val / 42 test (from our recovered-scalar protocol).
- Kust4K: official `train/val/test.txt` splits applied; `broke_RGB.txt` and `broke_TIR.txt` stems excluded. Final 1970 / 283 / 565.
- Caltech CART: 1822 / 222 / 238 labeled RGB/thermal pairs.

**Audit:** I initially flagged that Caltech might be under-paired. False alarm — 2282 is the labeled subset and is correct.

### Week 2 — Initial misalignment sweep

**Goal:** test whether synthetic misalignment hurts pix2pix-style translation.

**Done:** ran the pix2pix harness across sigma ∈ {0.0, 0.1, 0.3, 0.5} on all three datasets, single seed. Initial conclusion: **NO-GO** — drops small (<1 dB on externals, <2 dB on AA).

**Audit verdict:** the NO-GO was a protocol artifact. Perturbation range was small (0.08 translation, 8° rotation, 0.15 scale), targets were low-frequency, one seed. Decision: do not pivot; run Week 2.5 diagnostics.

### Week 2.5 — Diagnostics that overturned the NO-GO

**Goal:** harder protocol — bigger perturbations, more seeds, control experiments.

**Done:** Ann Arbor control, validation-time misalignment, shuffled-RGB controls, amplified sigma (translation 0.20, rotation 20°, scale 0.25), three seeds, L1-only baseline.

**Numbers (from `results/week2_5_diagnostics_summary.csv`):**

| Test | AA | K4K | CART |
|---|---:|---:|---:|
| Amplified train sigma=0.3, 3-seed mean drop | **−2.61 dB** | −1.08 dB | −0.52 dB |
| Shuffled-RGB drop (seed 42) | −6.89 dB | −3.59 dB | −4.61 dB |
| Training-time misalignment sigma=0.3 (seed 42, default range) | −1.98 dB | – | – |
| Training-time misalignment sigma=0.5 (seed 42) | −3.82 dB | – | – |
| Eval-time misalignment sigma=0.3 (aligned model) | −1.70 dB | −1.04 dB | −0.58 dB |
| Eval-time misalignment sigma=0.5 (aligned model) | −2.43 dB | −1.65 dB | −0.98 dB |
| L1-only training-time sigma=0.3 (AA seed 42) | −1.54 dB | – | – |

**Audit verdict:** Ann Arbor *is* alignment-sensitive. The Week 2 NO-GO was overturned. Pivot approved: the registration story is worth pursuing on Ann Arbor specifically.

### Week 3 — Registration v0

**Goal:** first registration variants.

**Done:** `week3_registration_v0.py` introduces 2 architectures:
- `target_conditioned` — oracle that sees the target during warp prediction (cheats by construction; theta_l2 ≈ 0).
- `shared_rgb` — shared feature encoder, feature-space warp.

**Audit verdict:** missing the no-registration baseline. Cannot tell if registration helps without it.

### Week 4 — Registration v1 + the no-reg baseline

**Goal:** add missing controls and try input-space registration.

**Done:** added three more architectures:
- `no_registration` — UNetReg only, identity theta. **The control baseline.**
- `input_rgb_affine` — small CNN predicts affine, warps the RGB input before the translator.
- `input_rgb_flow` — dense flow predictor.

Plus the direct synthetic-warp-recovery loss: `L1(warped_RGB, aligned_RGB)`.

**Numbers (seed 42, single-seed at this point, from `results/week4_registration_v1_summary.csv`; Week 4 protocol was raw target normalization, 30 epochs — different from the Week 7 locked robust/50-epoch protocol, so the absolute PSNRs are NOT directly comparable to later tables):**

| Dataset | no-reg PSNR | affine PSNR | Δ | lambda_warp |
|---|---:|---:|---:|---:|
| Ann Arbor | 15.89 | 16.20 | +0.313 | 1.0 |
| Kust4K | 19.03 | 19.18 | +0.147 | 1.0 |
| CART | 20.07 | 21.26 | **+1.196** | 1.0 |

**Target-normalization audit (from `results/week4_target_normalization_audit.csv`):**

| Dataset | raw edge mean | robust edge mean | histmatch edge mean |
|---|---:|---:|---:|
| Ann Arbor | 0.033 | 0.035 | 0.033 |
| Kust4K | 0.020 (60% of AA) | 0.029 | 0.028 |
| CART | 0.017 (50% of AA) | 0.026 | 0.022 |

**Audit verdict:** AA and CART look promising, but **single-seed and lambda-sensitive**. Need 3-seed audit and lambda sweep. Unsupervised variants (shared_rgb, input_rgb_flow) all underperformed no_reg.

### Week 5 — Cross-dataset transfer + the registration story narrows

**Goal:** test cross-dataset transfer; lock target normalization.

**Pre-flight 3-seed audit (from `results/week5_preflight_registration_summary.csv`):**

Kust4K:
| Seed | no-reg | affine | Δ |
|---:|---:|---:|---:|
| 42 | 19.031 | 19.178 | +0.147 |
| 7 | 18.993 | 19.013 | +0.020 |
| 123 | 18.995 | 19.115 | +0.120 |
| **Mean ± std** | – | – | **+0.096 ± 0.067 dB (NULL)** |

CART:
| Seed | no-reg | affine | Δ |
|---:|---:|---:|---:|
| 42 | 20.068 | 21.264 | +1.196 |
| 7 | 20.203 | 20.857 | +0.654 |
| 123 | 20.580 | 21.074 | +0.494 |
| **Mean ± std** | – | – | **+0.782 ± 0.368 dB** |

**CART lambda_warp_rgb sweep (seed 42 only):**
| λ_warp | CART Δ |
|---:|---:|
| 0.1 | +0.222 (below threshold) |
| 0.5 | +0.948 |
| 1.0 | +1.196 |
| 2.0 | +1.257 |

CART gain depends strongly on the warp-recovery loss weight.

**Cross-dataset transfer matrix (single-seed each, from `results/week5_transfer_matrix_summary.csv`):**

| Train → Eval | no-reg | affine | Δ | no-reg corr | affine corr |
|---|---:|---:|---:|---:|---:|
| AA → K4K | 9.90 | 10.31 | +0.404 | 0.291 | 0.369 |
| K4K → AA | 10.17 | 10.25 | +0.082 | 0.260 | 0.304 |
| K4K → CART | 10.05 | 9.86 | **−0.195** | −0.023 | −0.004 |
| CART → K4K | 9.67 | 9.26 | **−0.408** | −0.127 | **−0.151** |

External-to-external transfer fails. Some cells have negative Pearson correlations (model is anti-predictive).

**AA → K4K 3-seed follow-up (from `results/week5_aa_to_kust4k_3seed_summary.csv`):**

| Seed | no-reg | affine | Δ |
|---:|---:|---:|---:|
| 42 | 9.902 | 10.306 | +0.404 |
| 7 | 10.052 | 10.559 | +0.507 |
| 123 | 9.712 | 10.223 | +0.511 |
| **Mean ± std** | – | – | **+0.474 ± 0.061 dB** |

t = 13.5, df=2, **p ≈ 0.003** (one-tailed). Only transfer cell that survives 3-seed audit.

**Pretrain K4K+CART → AA fine-tune (seed 42, from `results/week5_pretrain_finetune_summary.csv`):** 15.85 dB vs 15.68 dB from-scratch = **+0.165 dB at 1.67× compute**. Tiny, possibly compute artifact.

**Audit verdict:** the "alignment is the universal bottleneck" thesis is dead. Only AA → K4K is real. Narrow the paper claim.

### Week 6 — Baselines (seed 42, matched 50 epochs, robust normalization)

**Goal:** all required baselines on the same protocol.

**Numbers (from `results/week6_baseline_summary.csv`):**

| Method | Family | PSNR | MAE | SSIM | Pearson r | Note |
|---|---|---:|---:|---:|---:|---|
| CycleGAN | unpaired | **8.598** | 0.302 | 0.269 | 0.180 | no paired L1 |
| pix2pix | paired GAN | 11.835 | 0.183 | – | – | Week 2 harness |
| Small U-Net L1 | paired L1 | 12.714 | 0.177 | – | – | pix2pix generator, L1 only |
| ConvNeXt + U-Net | paired regression | 15.637 | 0.105 | 0.536 | 0.812 | no-registration baseline |
| **Ours (supervised affine)** | registration | 15.920 | 0.099 | 0.549 | 0.828 | Week 5 method |
| Swin-T + U-Net | pretrained transformer | **16.123** | 0.096 | 0.566 | 0.834 | pretrained timm Swin-T encoder |

**Audit verdict:** Swin-T appeared to beat ours by 0.20 dB single seed. **But the margin is within the typical seed std.** Flagged for 3-seed audit.

### Week 7 (first half) — 3-seed seed audit

**Goal:** test whether Week 6 margins survive multi-seed.

**Numbers (from `results/week7_seed_audit_summary.csv`):**

| Family | Seed 42 | Seed 7 | Seed 123 | Mean ± std |
|---|---:|---:|---:|---:|
| ConvNeXt no-reg | 15.637 | 16.062 | 15.918 | 15.872 ± 0.217 |
| ConvNeXt affine (uncertainty-weighted) | 15.920 | 16.315 | 16.162 | 16.133 ± 0.199 |
| Swin-T no-reg | 16.123 | 16.335 | 16.228 | 16.228 ± 0.106 |

**Paired deltas:**

| Comparison | Mean ± std | t-stat | p (1-tailed) |
|---|---:|---:|---:|
| ConvNeXt affine − no-reg | **+0.260 ± 0.021 dB** | 21.5 | < 0.001 (significant) |
| Swin-T no-reg − ConvNeXt affine | +0.096 ± 0.096 dB | 1.73 | 0.11 (**NOT significant**) |

**Swin-T affine stacking (seed 42 only):** 16.001 vs 16.123 = **−0.122 dB**.

**Audit verdict:** Week 6 "Swin-T wins" collapses under 3-seed. Registration on ConvNeXt is real, small, and stable. Need 3-seed for Swin-T affine too.

### Week 7 (second half) — Ablations and the headline pivot

**Goal:** loss-term ablation, severity sweep, finalize Swin-T stacking with 3 seeds, lock the primary method.

**The single biggest finding:** disabling the uncertainty-weighted L1 (and zeroing λ_unc and λ_unc_tv) jumps the method from +0.260 dB to +0.571 dB over no-reg.

**3-seed table (from `results/week7_ablation_summary.csv`):**

| Family | Seed 42 | Seed 7 | Seed 123 | Mean ± std | Same-family Δ |
|---|---:|---:|---:|---:|---:|
| ConvNeXt no-reg | 15.637 | 16.062 | 15.918 | 15.872 ± 0.217 | — |
| ConvNeXt affine + uncertainty | 15.920 | 16.315 | 16.162 | 16.133 ± 0.199 | +0.260 ± 0.021 |
| **ConvNeXt affine, uncertainty-decoupled** | **16.312** | **16.453** | **16.566** | **16.444 ± 0.127** | **+0.571 ± 0.157** |
| Swin-T no-reg | 16.123 | 16.335 | 16.228 | 16.228 ± 0.106 | — |
| Swin-T affine | 16.001 | 16.507 | 15.984 | 16.164 ± 0.297 | **−0.064 ± 0.214 (null)** |

**Significance:**

| Comparison | t | df | p (1-tailed) | Verdict |
|---|---:|---:|---:|---|
| Method − ConvNeXt no-reg | 6.30 | 2 | 0.012 | **significant** |
| Method vs +0.3 dB threshold | 2.99 | 2 | 0.048 | clears threshold |
| Method − Swin-T no-reg | 3.30 | 2 | 0.040 | marginally significant |
| Swin-T affine − Swin-T no-reg | −0.52 | 2 | 0.66 | null (not negative) |

**Loss-term ablations (seed 42 only, primary = 16.312 dB):**

| Variant | PSNR | Δ vs primary |
|---|---:|---:|
| **Primary (uncertainty-decoupled)** | **16.312** | — |
| Add uncertainty weighting back | 15.920 | **−0.392** |
| Remove RGB warp-recovery loss | 15.814 | **−0.498** |
| Remove edge loss | 16.072 | −0.240 |
| Remove SSIM loss | 16.187 | −0.125 |
| Remove affine identity regularization | 16.083 | −0.229 |

**Load-bearing pieces:** direct RGB warp-recovery loss and disabling uncertainty weighting. Edge/SSIM/affine reg are smaller and within seed noise.

**Severity sweep (seed 42, matched train/eval sigma):**

| σ | no-reg | affine (det) | Δ |
|---:|---:|---:|---:|
| 0.0 | 17.154 | 17.278 | +0.124 |
| 0.1 | 16.698 | 16.304 | **−0.394** (anomaly) |
| 0.2 | 16.096 | 16.414 | +0.318 |
| 0.3 | 15.637 | 16.312 | +0.675 |
| 0.5 | 15.308 | 15.644 | +0.336 |

Non-monotonic at σ=0.1. Curve downgraded to diagnostic table, not polished paper figure.

**Loss-confound bound (seed 42, addresses cross-baseline loss-recipe difference):**

| Variant | PSNR | Δ |
|---|---:|---:|
| ConvNeXt no-reg (registration loss recipe) | 15.637 | — |
| ConvNeXt no-reg with Swin combined loss + uncertainty regularizers | 15.875 | +0.238 |
| ConvNeXt no-reg with Swin combined loss + λ_unc=0 | 15.838 | +0.201 (pure loss-recipe effect) |

Pure loss-recipe effect: +0.201 dB. The +0.571 dB method gain is comfortably larger.

**Audit verdict:** +0.571 ± 0.157 dB is statistically solid. Swin-T affine stacking is null. Method locked.

### Week 8 — Qualitative figures

**Goal:** hero scene, multi-sigma recovery, cross-dataset gallery, failure cases.

**Done:** `week8_make_final_figures.py` generates all 4 figures from the locked Week 7 checkpoint plus baseline checkpoints. All figures committed under `figures/week8/`.

**Numbers from `figures/week8/week8_final_figure_metrics.csv`:**

| Figure | Sample | Ours | Baseline | Δ |
|---|---|---:|---:|---:|
| Hero | AA 326 | 16.94 | 16.01 | **+0.93** (single sample, upper-tail) |
| Recovery σ=0.0 | AA 326 | 17.03 | 16.84 | +0.19 |
| Recovery σ=0.2 | AA 326 | 18.39 | 17.37 | +1.02 (best showing) |
| Recovery σ=0.5 | AA 326 | 13.84 | 13.96 | **−0.12** (ours loses, disclosed honestly) |
| Cross-dataset AA | sample 70 | 16.52 | 15.78 | +0.74 |
| Cross-dataset K4K | idx 90 | 19.13 | 18.67 | +0.46 (legacy checkpoint, see caveat) |
| Cross-dataset CART | idx 12 | 21.12 | 20.62 | +0.50 (legacy checkpoint, see caveat) |
| Failure AA 329 | – | 15.32 | 15.95 | **−0.64 (ours loses)** — selected by most negative paired delta |
| Failure AA 386 | – | 17.24 | 17.67 | **−0.43 (ours loses)** — selected by most negative paired delta |

**Audit issues caught and fixed (commit `680311b`):**

- **Failure-case selection bug.** Initial logic picked samples with lowest *ours-PSNR*. Fixed to pick samples with most negative *paired delta* (ours − baseline).
- **K4K cross-dataset overclaim risk.** Added in-figure footnote disclaimer "+0.096 ± 0.067 dB across seeds, not significant."
- **Hero gain caveat.** Disclosed in §5.6 prose, table caption, and (after Week 11) embedded as text inside the polished PNG itself.

### Week 9 — Full paper draft

**Goal:** write the WACV draft end-to-end.

**Done:** 8 sections + 2 Appendix placeholders in `paper_draft/main_draft.md`. Tables in `paper_draft/tables_and_captions.md` auto-generated from CSVs via `build_week9_tables.py`. Method diagram in `paper_draft/method_diagram.svg`. No-overclaiming checklist in `paper_draft/no_overclaiming_checklist.md` with claim/evidence/status table and wording rules.

**Audit issues caught and fixed in polish commit `d3d937e`:**

- **§6 misstatement about uncertainty regularizers.** Original text claimed "removes it from the reconstruction weighting path and sets the uncertainty regularizers to zero." Verified at CSV level — the primary deterministic rows DO have `lambda_uncertainty=0` and `lambda_uncertainty_tv=0`. Paper text now accurate.
- **§5.4 "ConvNeXt family fixed."** Table 3 includes both ConvNeXt and Swin-T rows. Rephrased to "across both ConvNeXt and Swin-T families."
- **Table 3 Swin-T deltas were confounding encoder change with registration.** Added a same-family registration delta column.
- **CycleGAN single-seed instability note** added.
- **Test count spot-check** via `UnifiedR2TDataset.from_roots(..., split="test")` on Knox: K4K test = 565, CART test = 238.
- **Method diagram** gained an inference-time annotation.
- **Hero PNG regenerated** with embedded "+0.93 dB; mean +0.571 ± 0.157" caveat.

### Week 10 — CCAI workshop scaffold

**Goal:** prepare a 4-page version for the climate workshop.

**Done:** `WEEK10_CCAI_WORKSHOP_PLAN.md`, `paper_draft/ccai_short_draft.md` (137 lines, climate-reframed), `paper_draft/ccai_cut_list.md` (explicit keep/compress/cut map). Added loss-confound refinement to the WACV main draft.

**Blocker:** CCAI 2026 CFP not officially posted as of 2026-06-17. Planning under conservative date Aug 29 2026 AoE.

### Week 11 — Bibliography, polished figures, review packet

**Goal:** prepare the draft for internal review and Prof. Siwo handoff.

**Done:**

- **`paper_draft/references.bib`** — 42 BibTeX entries spanning urban heat / thermal remote sensing (9), I2I translation + architecture + uncertainty + registration (16), RGB-T / aerial thermal datasets (17).
- **57 inline Pandoc-style citations** in `main_draft.md` covering **40 unique** references (cross-checked: 0 missing keys; 2 dead entries).
- **`paper_draft/citation_audit.md`** — coverage doc + camera-ready TODO list.
- **`paper_draft/review_packet.md`** — paragraph-by-paragraph review instructions + 5 top-level framing questions for Prof. Siwo.
- **`week11_polish_figures.py`** — non-destructive wrapper for the Week 8 PNGs. Uses TrueType (Arial/Segoe/Calibri fallback chain), wraps subtitle text. Outputs `figures/week8/*_polished_seed42.png`.
- **No-overclaiming checklist** extended with sentence-by-sentence abstract audit.

**Audit issues to fix before LaTeX conversion (none blocking review):**

- 2 dead bib entries — `vertens2020heatnet`, `zhang2018lpips`
- 3 BibTeX type errors — `detone2016homography` (`@inproceedings` → `@article`), `ipcc2022wgii` (`@article` → `@techreport`), `ouyang2025kust4k` (`@article` → `@misc`)
- 5-6 recent arXiv IDs (2025/2026) need web-verification
- `sherpa2026conditional` — Prof. Siwo is a co-author; he should personally verify
- No DOI fields yet (deferred to camera-ready)

---

## 4. The locked Week 7 method

### 4.1 Architecture

```
RGB input (256 × 320 × 3, normalized to [0, 1])
        │
        ▼
┌──────────────────────────┐
│  RGB-only affine head    │  4 conv blocks (32→64→128→128) + AdaptiveAvgPool + MLP(128→128→6)
│  Output: theta ∈ R^{2×3} │  Identity-initialized (zero weight, bias = [1,0,0,0,1,0])
└──────────────────────────┘
        │
        ▼  apply affine (grid_sample)
┌──────────────────────────┐
│  Warped RGB x_warp       │  256 × 320 × 3
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  ConvNeXt-tiny encoder   │  timm convnext_tiny.fb_in22k pretrained
│  + UNet decoder          │  UNetReg from train_a1
│  + sigmoid head          │  Output: scalar in [0, 1]
└──────────────────────────┘
        │
        ▼
Predicted thermal scalar y_hat (256 × 320 × 1)
```

Also produces:
- `theta` (the predicted warp parameters)
- `warped_raw` (the warped RGB; used for warp-recovery loss)
- `uncertainty` (per-pixel scalar; **diagnostic only, not used in loss**)

### 4.2 Training protocol

- **Optimizer:** AdamW, lr = 2e-4, weight decay = 1e-4.
- **Schedule:** Cosine annealing over the training epochs.
- **Epochs:** 50.
- **Batch size:** 6 for Ann Arbor (336 train); 8 for external datasets.
- **Resolution:** 256 × 320.
- **Gradient clip:** 1.0.
- **Seeds:** {42, 7, 123} (the 3-seed audit standard).
- **Target normalization:** robust (per-sample 1st-99th percentile rescale to [0, 1]).
- **Synthetic misalignment:** sigma=0.3 at amplified range (max_translation_frac=0.20, max_rotation_deg=20°, max_scale_frac=0.25).
- **Lambda weights:** see §9.

### 4.3 Why each choice

| Choice | Reason | Audit evidence |
|---|---|---|
| ConvNeXt-tiny | Strongest non-transformer baseline; ImageNet pretrained | Week 6 baseline table |
| Robust normalization | Boosts external dataset edge content uniformly | Week 4 target-norm audit |
| Sigma=0.3 amplified | Strong enough to expose alignment sensitivity (2.61 dB AA drop) | Week 2.5 diagnostics |
| 3 seeds | Single-seed margins were within seed std | Week 5 AA diagonal std ±0.207 |
| 50 epochs | Matches all baselines for fair compute | Week 6 protocol lock |
| λ_warp = 0.5 | Compromise between λ=0.1 (below threshold) and λ=2.0 (loss-dominant) | Week 5 CART sweep |
| Identity-init theta | Prevents random destruction of input at start | standard STN practice |
| Uncertainty-decoupled | Adding uncertainty weighting hurts by 0.392 dB at seed 42 | Week 7 ablation |

---

## 5. Headline numbers — single reference table

Memorize this table. Every paper claim traces back to one of these rows.

| Statement | Number | Status |
|---|---:|---|
| **Method vs ConvNeXt no-reg (paired, 3 seeds)** | **+0.571 ± 0.157 dB** | **p ≈ 0.012, significant** |
| **Method vs Swin-T no-reg (paired, 3 seeds)** | **+0.215 ± 0.113 dB** | **p ≈ 0.04, marginally significant** |
| Swin-T affine stacking (3 seeds) | −0.064 ± 0.214 dB | null (not negative) |
| Uncertainty weighting added (seed 42) | −0.392 dB | strong negative |
| Warp-recovery loss removed (seed 42) | −0.498 dB | strong negative |
| Kust4K within-dataset (3 seeds) | +0.096 ± 0.067 dB | **null** |
| CART within-dataset (3 seeds) | +0.782 ± 0.368 dB | passes mean but loss-balance sensitive |
| CART λ=0.1 single-seed | +0.222 dB | loss-balance confound |
| CART λ=2.0 single-seed | +1.257 dB | loss-balance confound |
| AA → Kust4K transfer (3 seeds) | +0.474 ± 0.061 dB | only transfer cell that survives audit |
| Other 3 transfer cells | mixed/negative single-seed | diagnostic only |
| Hero scene Δ | +0.93 dB | upper-tail single sample, disclosed |
| Recovery σ=0.5 (hero scene) | −0.12 dB | ours loses, disclosed |
| Pure loss-recipe confound (seed 42) | +0.201 dB | comfortably smaller than +0.571 dB |
| AA 3-seed amplified σ=0.3 drop | −2.61 dB | alignment-sensitivity proof |
| K4K 3-seed amplified σ=0.3 drop | −1.08 dB | weaker effect |
| CART 3-seed amplified σ=0.3 drop | −0.52 dB | weakest effect |
| Hackathon ensemble (legacy, raw protocol) | 19.28 dB | not comparable to robust protocol |
| Locked Week 7 AA from-scratch (robust) | 16.444 ± 0.127 dB | paper headline (Δ vs no-reg in row 1) |

---

## 6. Full ablation tables

These come straight from `results/week7_ablation_summary.csv`. The full CSV has 28 columns and 28 rows; the most-cited subset is below.

### 6.1 Architecture × seed (the headline)

| Run | Seed | PSNR | SSIM | Pearson r | theta_l2 | final_unc | warp_rgb_mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt no-reg | 42 | 15.637 | 0.536 | 0.812 | 0.000 | 0.000 | 0.152 |
| ConvNeXt no-reg | 7 | 16.062 | 0.550 | 0.830 | 0.000 | 0.000 | 0.156 |
| ConvNeXt no-reg | 123 | 15.918 | 0.544 | 0.822 | 0.000 | 0.000 | 0.151 |
| ConvNeXt affine + unc | 42 | 15.920 | 0.549 | 0.828 | 0.038 | 0.568 | 0.106 |
| ConvNeXt affine + unc | 7 | 16.315 | 0.556 | 0.839 | 0.040 | 0.552 | 0.100 |
| ConvNeXt affine + unc | 123 | 16.162 | 0.554 | 0.834 | 0.038 | 0.572 | 0.098 |
| **ConvNeXt affine det.** | **42** | **16.312** | 0.557 | 0.842 | 0.037 | 0.700 | 0.108 |
| **ConvNeXt affine det.** | **7** | **16.453** | 0.560 | 0.845 | 0.039 | 0.694 | 0.101 |
| **ConvNeXt affine det.** | **123** | **16.566** | 0.562 | 0.849 | 0.039 | 0.651 | 0.096 |
| Swin-T no-reg | 42 | 16.123 | 0.566 | 0.834 | 0.000 | – | 0.000 |
| Swin-T no-reg | 7 | 16.335 | 0.567 | 0.841 | 0.000 | – | 0.000 |
| Swin-T no-reg | 123 | 16.228 | 0.565 | 0.836 | 0.000 | – | 0.000 |
| Swin-T affine | 42 | 16.001 | 0.567 | 0.833 | 0.036 | – | 0.110 |
| Swin-T affine | 7 | 16.507 | 0.576 | 0.846 | 0.039 | – | 0.102 |
| Swin-T affine | 123 | 15.984 | 0.562 | 0.830 | 0.026 | – | 0.129 |

### 6.2 Loss-term ablations (seed 42 only — primary = 16.312 dB)

| Variant | PSNR | SSIM | Pearson r | Δ vs primary |
|---|---:|---:|---:|---:|
| **Primary (det., λ_warp=0.5, all on)** | **16.312** | **0.557** | **0.842** | — |
| Add uncertainty weighting back | 15.920 | 0.549 | 0.828 | −0.392 |
| Remove warp-recovery loss (λ_warp=0) | 15.814 | 0.545 | 0.820 | **−0.498** |
| Remove edge loss (λ_edge=0) | 16.072 | 0.553 | 0.836 | −0.240 |
| Remove SSIM loss (λ_ssim=0) | 16.187 | 0.549 | 0.834 | −0.125 |
| Remove affine reg (λ_affine=0) | 16.083 | 0.550 | 0.834 | −0.229 |

### 6.3 Loss-recipe confound (seed 42)

| Variant | PSNR | Δ |
|---|---:|---:|
| ConvNeXt no-reg (registration recipe, default λ_unc=0.01) | 15.637 | — |
| ConvNeXt no-reg + Swin combined loss + λ_unc=0.01 | 15.875 | +0.238 |
| ConvNeXt no-reg + Swin combined loss + λ_unc=0 | 15.838 | +0.201 (pure loss effect) |

Decomposition: +0.201 dB pure loss, +0.037 dB pure regularizer removal, +0.238 dB combined. The +0.571 dB method gain is comfortably larger than +0.238 dB.

### 6.4 Severity sweep (seed 42)

| σ | no-reg PSNR | det. affine PSNR | Δ |
|---:|---:|---:|---:|
| 0.0 | 17.154 | 17.278 | +0.124 |
| 0.1 | 16.698 | 16.304 | **−0.394** (anomaly) |
| 0.2 | 16.096 | 16.414 | +0.318 |
| 0.3 | 15.637 | 16.312 | +0.675 |
| 0.5 | 15.308 | 15.644 | +0.336 |

Single-seed, non-monotonic at σ=0.1 → diagnostic only, not paper figure.

---

## 7. Datasets — complete documentation

### 7.1 Ann Arbor (primary)

- **Source:** UMich Heat Resilience Hackathon proprietary dataset.
- **Total:** ~418 paired images + 202 hidden test images (used by the official hackathon leaderboard).
- **Our split (locked Week 7):** 336 train / 41 val / 42 test (from the ~418 with available scalar targets).
- **RGB:** 4000 × 3000 (12 MP), 4:3 aspect.
- **Thermal:** 640 × 512, 5:4 aspect. **Recovered scalar via 1-D LUT (`r2t_common.to_scalar`).** Not raw 16-bit thermal.
- **Edge mean (after robust norm):** 0.035 (highest of the three).
- **Path on Knox:** `/home/spant/UMich/umich-hackathon/data/Train_2/`
- **Why primary:** the target has the most high-frequency structure of the three datasets, making it the dataset where synthetic misalignment has the clearest measurable effect.

### 7.2 Kust4K (public)

- **Citation:** OuYang, Wang, Shen (2025). "Kust4K: An RGB-TIR dataset from UAV platform for robust urban traffic scenes semantic segmentation." Figshare DOI 10.6084/m9.figshare.29476610.v3.
- **Total:** 4024 paired images at 640 × 512, captured from UAV over urban traffic scenes.
- **Our split:** 1970 train / 283 val / 565 test, using the official `train.txt`/`val.txt`/`test.txt` split files and excluding stems listed in `broke_RGB.txt` and `broke_TIR.txt`.
- **Target:** raw grayscale TIR (8-bit). Not recovered scalar.
- **Edge mean (robust):** 0.029.
- **Knox path:** `/home/spant/UMich/umich-hackathon/data_cache/external/kust4k/`
- **Figshare URL:** https://figshare.com/articles/dataset/b_Kust4K_b_b_b_b_A_Large-scale_Multimodal_UAV_Dataset_for_Robust_Urban_Traffic_Scenes_Semantic_Segmentation_b/29476610

### 7.3 Caltech CART (public)

- **Citation:** Lee, Anderson, Raganathan, Zuo, Do, Gkioxari, Chung (2024). "CART: Caltech aerial RGB-thermal dataset in the wild." arXiv:2403.08997.
- **Total subset used:** the labeled RGB/thermal paired subset (`labeled_rgbt_pairs`). The broader dataset has additional imagery.
- **Our split:** 1822 train / 222 val / 238 test.
- **Target:** raw grayscale thermal. Not recovered scalar.
- **Edge mean (robust):** 0.026 (lowest, most smoothed).
- **Knox path:** `/home/spant/UMich/umich-hackathon/data_cache/external/caltech/`
- **CaltechDATA URL:** https://data.caltech.edu/records/cks6g-ps927
- **GitHub:** https://github.com/aerorobotics/caltech-aerial-rgbt-dataset

### 7.4 Quick stats summary

| Dataset | Train | Val | Test | Target type | Edge mean (robust) |
|---|---:|---:|---:|---|---:|
| Ann Arbor | 336 | 41 | 42 | Recovered scalar (1-D LUT) | 0.035 |
| Kust4K | 1970 | 283 | 565 | Raw grayscale TIR | 0.029 |
| CART | 1822 | 222 | 238 | Raw grayscale | 0.026 |

---

## 8. Architecture details

### 8.1 RGB-only affine registration head (in `week3_registration_v0.py`)

```python
class RGBInputAffineRegistrationTranslator(nn.Module):
    def __init__(self, encoder="convnext_tiny"):
        super().__init__()
        # Small convolutional encoder for the affine head
        self.reg_enc = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
        )
        # MLP head producing 6-DOF affine theta
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 6),
        )
        # Identity initialization: zero the final layer's weight,
        # set bias to identity affine [1, 0, 0, 0, 1, 0]
        nn.init.zeros_(self.theta[-1].weight)
        with torch.no_grad():
            self.theta[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        # The translator (ConvNeXt + UNetReg) and uncertainty head
        self.translator = UNetReg(encoder=encoder, in_ch=3, use_alpha=False)
        # ... uncertainty branch
```

### 8.2 UNetReg decoder (in `train_a1.py`)

ConvNeXt-tiny encoder produces 4 levels of features. UNetReg uses U-Net-style skip connections with `ConvBlock` decoder modules ending in a sigmoid head. Final output: scalar prediction in [0, 1] at the input resolution.

### 8.3 Forward pass

```python
def forward(self, rgb_raw, target=None):
    feat = self.reg_enc(rgb_raw)
    theta = self.theta(feat).view(-1, 2, 3)  # 6-DOF affine
    # apply_affine = grid_sample with affine grid
    warped_raw = apply_affine(rgb_raw, theta)
    pred = self.translator(warped_raw)["pred"]  # sigmoid-activated scalar
    uncertainty = self.unc_branch(warped_raw)  # diagnostic only
    return dict(
        pred=pred,
        theta=theta,
        warped_raw=warped_raw,
        uncertainty=uncertainty,
        flow=None,  # affine variant returns None
    )
```

### 8.4 Parameter counts

| Module | Approximate parameters |
|---|---:|
| Affine head (reg_enc + theta MLP) | ~250K |
| ConvNeXt-tiny encoder | ~28M (pretrained) |
| UNetReg decoder | ~5M |
| Uncertainty branch | ~500K |
| **Total trainable** | **~33M** |

For comparison, the Week 6 Swin-T+UNet baseline has ~30M parameters (similar order).

---

## 9. Loss equations in full

### 9.1 Primary method loss (locked Week 7)

```
L_total = L1(y_hat, y)                                          # unweighted reconstruction
        + 0.25 · (1 − SSIM(y_hat, y))                          # structural similarity
        + 0.10 · L_edge(y_hat, y)                              # edge magnitude on warped_raw
        + 0.02 · ||theta − I||²                                # affine identity prior
        + 0.5  · L1(x_warp, x_aligned)                         # warp-recovery (load-bearing)
        + 0    · mean(u)                                       # uncertainty mean (DISABLED)
        + 0    · TV(u)                                         # uncertainty TV (DISABLED)
```

Where:
- `y` = thermal target (normalized via robust per-sample rescale)
- `y_hat` = model prediction
- `theta` = predicted 2×3 affine parameters
- `I` = identity affine [[1,0,0],[0,1,0]]
- `x_warp` = `out["warped_raw"]` = the result of applying theta to the misaligned RGB input
- `x_aligned` = the *original* unmisaligned RGB (teacher-forced supervision signal)
- `u` = per-pixel uncertainty scalar in [0, 1]

### 9.2 What "uncertainty-decoupled" means concretely

The model still has an uncertainty head and still produces `u`. But:

1. The L1 reconstruction loss is **unweighted** (not weighted by `1/(1+u)` as in the earlier variant).
2. The two uncertainty regularizers (`mean(u)` and `TV(u)`) have their lambdas set to **0**.
3. The uncertainty map is still computed and saved for diagnostic visualization (you can see it in the hero figure).
4. CSV-level confirmation: `results/week7_ablation_summary.csv` columns `disable_uncertainty_weight=True`, `lambda_uncertainty=0.0`, `lambda_uncertainty_tv=0.0` for the primary deterministic rows. Final `u` mean ~0.70 (the branch is alive, just not pulled small by the regularizer).

### 9.3 Edge loss (in `week3_registration_v0.py`)

```
L_edge(y_hat, y) = L1( normalize_per_sample(edge_mag(x_warp)),
                       normalize_per_sample(edge_mag(y))      )
```

`edge_magnitude` is a Sobel-like gradient magnitude. The edge is computed on the *warped RGB input*, not on the prediction. This trains the warp to align gradient structure.

### 9.4 SSIM term

Standard SSIM from `r2t_common.ssim`. Returns a scalar in [-1, 1]; we use `1 - SSIM` as a loss.

### 9.5 Combined-loss recipe used by Swin-T baseline (for comparison)

```
L_combined = L1(y_hat, y)
           + 0.5 · (1 − SSIM(y_hat, y))
           + 0.5 · L_grad(y_hat, y)
```

Different weights than our registration recipe. This is the "swin_combined" loss recipe in `week3_registration_v0.py`. The cross-baseline loss-recipe confound is bounded at +0.201 dB (well under our +0.571 dB gain).

---

## 10. Reproducing key experiments — exact commands

All commands assume Knox (`/home/spant/UMich/umich-hackathon/rgb2thermal_wacv/`) with `weather_experiments/.venv` activated. They produce CSV/JSON outputs under `week*_runs/`.

### 10.1 Train the primary method (Ann Arbor, seed 42)

```bash
python week3_registration_v0.py \
  --name week7_convnext_affine_deterministic_lam0p5_ann_arbor_robust_sigma03_seed42_e50 \
  --dataset ann_arbor \
  --arch input_rgb_affine \
  --encoder convnext_tiny \
  --target-normalization robust \
  --train-sigma 0.3 --eval-sigma 0.3 \
  --max-translation-frac 0.20 --max-rotation-deg 20.0 --max-scale-frac 0.25 \
  --lambda-warp-rgb 0.5 \
  --lambda-ssim 0.25 --lambda-edge 0.10 --lambda-affine 0.02 \
  --lambda-uncertainty 0.0 --lambda-uncertainty-tv 0.0 \
  --disable-uncertainty-weight \
  --epochs 50 --bs 6 --lr 2e-4 \
  --seed 42 --res 256 \
  --out-dir week7_runs
```

### 10.2 The matched no-registration baseline

```bash
python week3_registration_v0.py \
  --name week7_convnext_no_registration_ann_arbor_robust_sigma03_seed42_e50 \
  --dataset ann_arbor \
  --arch no_registration \
  --encoder convnext_tiny \
  --target-normalization robust \
  --train-sigma 0.3 --eval-sigma 0.3 \
  --max-translation-frac 0.20 --max-rotation-deg 20.0 --max-scale-frac 0.25 \
  --epochs 50 --bs 6 --lr 2e-4 \
  --seed 42 --res 256 \
  --out-dir week7_runs
```

### 10.3 Swin-T no-registration baseline

```bash
python week6_swin_unet_baseline.py \
  --dataset ann_arbor \
  --arch no_registration \
  --encoder swin_tiny_patch4_window7_224.ms_in1k \
  --target-normalization robust \
  --train-sigma 0.3 --eval-sigma 0.3 \
  --max-translation-frac 0.20 --max-rotation-deg 20.0 --max-scale-frac 0.25 \
  --epochs 50 --batch-size 4 --lr 2e-4 \
  --seed 42 \
  --out-dir week7_runs
```

### 10.4 Cross-dataset transfer (Ann Arbor → Kust4K)

```bash
python week3_registration_v0.py \
  --name week5_transfer_affine_lam0p5_aa_to_kust4k_robust_seed42_e30 \
  --dataset ann_arbor --eval-dataset kust4k \
  --arch input_rgb_affine \
  --target-normalization robust \
  --lambda-warp-rgb 0.5 \
  --train-sigma 0.3 --eval-sigma 0.3 \
  --max-translation-frac 0.20 --max-rotation-deg 20.0 --max-scale-frac 0.25 \
  --epochs 30 --bs 6 --seed 42 \
  --out-dir week5_runs
```

### 10.5 Aggregate metrics into a summary CSV

```bash
# After all runs are done, regenerate the summary CSV
python collect_week7_ablation_metrics.py --base . --output results/week7_ablation_summary.csv
```

### 10.6 Build paper tables from CSVs

```bash
python build_week9_tables.py --stdout
# Should byte-match paper_draft/tables_and_captions.md
```

### 10.7 Regenerate figures

```bash
# Original Week 8 figures (requires checkpoints on Knox)
python week8_make_final_figures.py --output-dir figures/week8

# Polished previews (uses existing Week 8 PNGs; no checkpoint required)
python week11_polish_figures.py --input-dir figures/week8 --output-dir figures/week8
```

### 10.8 Generate this PDF

```bash
python md2pdf.py WEEKLY_HANDOFF_FOR_ADITHYA.md WEEKLY_HANDOFF_FOR_ADITHYA.pdf .
```

---

## 11. Complete repo file inventory

### 11.1 Paper draft (`paper_draft/`)

| File | Purpose | Size |
|---|---|---|
| `main_draft.md` | Full WACV draft, 8 sections + 2 appendix placeholders | ~300 lines |
| `tables_and_captions.md` | Tables 1-4 and figure captions | ~75 lines |
| `references.bib` | 42 BibTeX entries (40 cited; 2 dead) | ~330 lines |
| `method_diagram.svg` | Vector method diagram | ~82 lines |
| `no_overclaiming_checklist.md` | Wording rules, claim/evidence table, abstract sentence-by-sentence audit | ~38 lines |
| `citation_audit.md` | Citation coverage doc, camera-ready TODO | ~45 lines |
| `review_packet.md` | Internal review instructions + Prof. Siwo questions | ~49 lines |
| `ccai_short_draft.md` | 4-page workshop variant (climate-reframed) | ~137 lines |
| `ccai_cut_list.md` | WACV → CCAI keep/compress/cut map | ~55 lines |

### 11.2 Weekly memos (`/`)

| File | Topic |
|---|---|
| `PLAN.md` | The full 12-week plan with every checkbox + result + blocker |
| `WEEK2_PRELIMINARY.md` | Week 2 initial NO-GO writeup |
| `WEEK2_GO_NO_GO_MEMO.md` | Decision rationale |
| `WEEK2_5_DIAGNOSTICS.md` | Diagnostic protocol |
| `WEEK2_5_DIAGNOSTICS_RESULT.md` | The overturn |
| `WEEK3_REGISTRATION_V0_RESULT.md` | First registration variants |
| `WEEK4_REGISTRATION_V1_PROGRESS.md` | Registration v1 numbers |
| `WEEK4_TARGET_NORMALIZATION_AUDIT.md` | Edge-energy comparison |
| `WEEK5_PREFLIGHT.md` | 3-seed audit + lambda sweep + normalization choice |
| `WEEK5_TRANSFER_RESULT.md` | Cross-dataset transfer matrix |
| `WEEK5_FOLLOWUP_RESULT.md` | Week 5 follow-up controls |
| `WEEK6_BASELINES_RESULT.md` | Full baseline table + Swin-T flag |
| `WEEK7_SEED_AUDIT_RESULT.md` | 3-seed verification of Week 6 |
| `WEEK7_ABLATIONS_RESULT.md` | The headline pivot to uncertainty-decoupled |
| `WEEK8_QUALITATIVE_RESULT.md` | Figure generation notes |
| `WEEK9_DRAFT_SCAFFOLD.md` | Pre-draft framing map |
| `WEEK10_CCAI_WORKSHOP_PLAN.md` | CCAI workshop planning |
| `WEEK11_POLISH_RESULT.md` | Bibliography + polish + review packet |

### 11.3 Scripts (`/`)

| Script | Purpose |
|---|---|
| `data_prep.py` | Palette inversion, crop refinement, target generation |
| `r2t_common.py` | Shared losses, metrics, SSIM, combined_loss |
| `unified_dataset.py` | Multi-dataset loader with robust/histmatch normalization |
| `train_a1.py` | Base UNetReg architecture |
| `week2_pix2pix_baseline.py` | pix2pix harness; also source of `_warp_rgb` synthetic misalignment |
| `week3_registration_v0.py` | **Main registration trainer (5 architectures, 2 loss recipes)** |
| `week4_target_normalization_audit.py` | Edge-energy diagnostic |
| `week6_swin_unet_baseline.py` | Swin-T U-Net baseline + Swin-T affine variant |
| `week6_cyclegan_baseline.py` | CycleGAN baseline |
| `week8_make_qualitative_figures.py` | Candidate-grid figures |
| `week8_make_final_figures.py` | Hero, recovery, gallery, failure-case figures |
| `week11_polish_figures.py` | TrueType-font polish wrapper for Week 8 PNGs |
| `build_target_normalization_stats.py` | Per-dataset quantile stats builder |
| `collect_week4_registration_metrics.py` | Week 4 metric aggregator |
| `collect_week6_baseline_metrics.py` | Week 6 metric aggregator |
| `collect_week7_ablation_metrics.py` | Week 7 metric aggregator |
| `build_week9_tables.py` | **Regenerates paper tables from CSVs** |
| `md2pdf.py` | Markdown → PDF (uses xhtml2pdf) |
| `make_synthetic_misalignment.py` | Standalone misalignment data generator |

### 11.4 Figures (`figures/`)

| Subdirectory / file | Use |
|---|---|
| `figures/best_vs_gt.png` etc. | From-scratch rebuild galleries (pre-Week 1) |
| `figures/data_sanity_gallery.png` | Palette inversion + registration sanity check |
| `figures/week8/hero_ann_arbor_seed42.png` | Hero figure (raw) |
| `figures/week8/hero_ann_arbor_polished_seed42.png` | **Hero with caveat embedded** |
| `figures/week8/cross_dataset_gallery_seed42.png` | Cross-dataset gallery (raw, has in-figure K4K footnote) |
| `figures/week8/cross_dataset_gallery_polished_seed42.png` | **Polished version with title banner** |
| `figures/week8/failure_cases_ann_arbor_seed42.png` | Failure cases (raw, fixed selection logic) |
| `figures/week8/failure_cases_ann_arbor_polished_seed42.png` | **Polished version** |
| `figures/week8/misalignment_recovery_multisigma_seed42.png` | Multi-sigma recovery |
| `figures/week8/ann_arbor_candidate_grid_seed42.png` | Broad candidate grid |
| `figures/week8/week8_final_figure_metrics.csv` | Per-figure metrics |
| `figures/week8/ann_arbor_candidate_metrics_seed42.csv` | Candidate-grid per-sample metrics |

---

## 12. Full CSV inventory with column descriptions

### 12.1 Key CSVs (in `results/`)

| CSV | What it contains | Used in paper |
|---|---|---|
| `week2_5_diagnostics_summary.csv` | Multi-seed amplified misalignment drops | §5.1 motivating diagnostics |
| `week4_target_normalization_audit.csv` | Edge mean / p99 / entropy per dataset × norm mode | §3 Datasets, §5.5 external |
| `week5_preflight_registration_summary.csv` | 3-seed K4K and CART within-dataset audit | §5.5 |
| `week5_transfer_matrix_summary.csv` | 4-cell cross-dataset transfer matrix | §5.5 / Table 4 |
| `week5_aa_to_kust4k_3seed_summary.csv` | 3-seed AA → K4K transfer | §5.5 / Table 4 |
| `week5_matched_compute_control_summary.csv` | 50-epoch AA control under robust | §5.4 |
| `week5_pretrain_finetune_summary.csv` | External pretrain + AA fine-tune | §5.5 |
| `week6_baseline_summary.csv` | All baseline rows at seed 42 | §5.3 / Table 2 |
| `week7_seed_audit_summary.csv` | 3-seed audit of Week 6 | §5.4 |
| `week7_ablation_summary.csv` | **Full Week 7 ablation including the +0.571 dB rows** | **§5.4 / Table 3** (load-bearing) |
| `week5_target_normalization_stats.json` | Quantile stats for histmatch normalizer | infrastructure |
| `week5_target_norm_audit_raw.csv` etc. | Per-mode edge stats per dataset | §3 |

### 12.2 Common columns

Most week*_summary.csv files share this column schema (subset):

| Column | Meaning |
|---|---|
| `run` | Run directory name (`week<n>_runs/...`) |
| `arch` | `no_registration` / `input_rgb_affine` / `input_rgb_flow` / etc. |
| `dataset`, `eval_dataset` | Train and eval dataset names |
| `seed` | Random seed |
| `train_sigma`, `eval_sigma` | Misalignment sigma at train / eval |
| `translation_frac`, `rotation_deg`, `scale_frac` | Misalignment range (amplified = 0.20/20/0.25) |
| `target_normalization` | `raw` / `robust` / `histmatch` |
| `lambda_*` | Loss weights (`lambda_warp_rgb`, `lambda_ssim`, `lambda_edge`, `lambda_affine`, `lambda_uncertainty`, `lambda_uncertainty_tv`) |
| `disable_uncertainty_weight` | True/False — primary method has True |
| `epochs`, `train_count`, `val_count` | Training budget and counts |
| `final_mae`, `final_rmse`, `final_psnr`, `final_ssim`, `final_corr` | Eval metrics at last epoch |
| `final_theta_l2` | Mean L2 distance of theta from identity (alignment strength) |
| `final_uncertainty` | Mean uncertainty (only meaningful for uncertainty variants) |
| `final_warp_rgb_mae` | L1 between warped RGB and aligned RGB at eval |
| `delta_vs_no_registration` | Paired delta with the matched no_registration row |
| `knox_path` or `path` | Run dir on Knox |

---

## 13. Bibliography — all 42 entries

Defined in `paper_draft/references.bib`. Listed alphabetically by key.

### Urban heat / thermal remote sensing (9)

1. `imhoff2010remote` — Imhoff et al. 2010, urban heat island across biomes, RSE.
2. `ipcc2022wgii` — IPCC AR6 WGII 2022 (note: should be `@techreport` — TODO).
3. `liu2023island` — Liu et al. 2023, ISLAND: LST interpolation using land cover, arXiv:2309.12416.
4. `martin2022iris` — Martin et al. 2022, Singapore urban heat IR observatory, arXiv:2210.11663.
5. `phelan2015urban` — Phelan et al. 2015, urban heat island mechanisms, Annu Rev Env Res.
6. `stewart2012local` — Stewart & Oke 2012, local climate zones, BAMS.
7. `voogt2003thermal` — Voogt & Oke 2003, thermal remote sensing of urban climates, RSE.
8. `weng2004estimation` — Weng et al. 2004, LST-vegetation abundance, RSE.
9. `zhan2025satellite` — Zhan et al. 2025, satellite LST mischaracterization, arXiv:2509.16568.

### Architecture, registration, uncertainty, metrics (16)

10. `balakrishnan2019voxelmorph` — VoxelMorph deformable medical registration, IEEE TMI 2019.
11. `dai2017deformable` — Deformable convolutional networks, ICCV 2017.
12. `detone2016homography` — Deep image homography estimation, arXiv:1606.03798. (Type: should be `@article` — TODO.)
13. `dosovitskiy2021vit` — Vision Transformer (ViT), ICLR 2021.
14. `gal2016dropout` — Dropout as Bayesian approximation, ICML 2016.
15. `he2016resnet` — ResNet, CVPR 2016.
16. `he2022mae` — Masked autoencoders, CVPR 2022.
17. `isola2017pix2pix` — pix2pix conditional GANs, CVPR 2017.
18. `jaderberg2015stn` — Spatial transformer networks, NeurIPS 2015.
19. `kendall2018multi` — Multi-task uncertainty weighting, CVPR 2018.
20. `liang2021swinir` — SwinIR image restoration, ICCV Workshops 2021.
21. `liu2021swin` — Swin Transformer, ICCV 2021.
22. `liu2022convnext` — ConvNeXt, CVPR 2022.
23. `ronneberger2015unet` — U-Net, MICCAI 2015.
24. `wang2004ssim` — SSIM, IEEE TIP 2004.
25. `zamir2022restormer` — Restormer, CVPR 2022.
26. `zhang2018lpips` — LPIPS, CVPR 2018. (**Dead — not cited; either cite in §5 or remove.**)
27. `zhu2017cyclegan` — CycleGAN, ICCV 2017.

### RGB-T / RGB-IR datasets, fusion, RGB-to-thermal translation (17)

28. `canitez2026frequency` — Frequency-guided fusion RGB-T segmentation, arXiv:2605.26273.
29. `hwang2015kaist` — KAIST multispectral pedestrian detection, CVPR 2015.
30. `jia2021llvip` — LLVIP visible-infrared low-light, arXiv:2108.10831.
31. `kemker2017multispectral` — Multispectral segmentation dataset, arXiv:1703.01918.
32. `lee2024cart` — CART aerial RGB-thermal dataset, arXiv:2403.08997.
33. `liu2016multispectral` — Multispectral pedestrian detection, arXiv:1611.02644.
34. `ouyang2025kust4k` — Kust4K UAV RGB-TIR dataset, Figshare 10.6084/m9.figshare.29476610.v3. (Type: should be `@misc` — TODO.)
35. `sherpa2026conditional` — Sherpa et al. 2026 (**Prof. Siwo is coauthor — verify directly**), arXiv:2605.17564.
36. `shivakumar2019pst900` — PST900 RGB-Thermal segmentation, arXiv:1909.10980.
37. `vertens2020heatnet` — HeatNet day-night segmentation. (**Dead — not cited; either cite in §2 or remove.**)
38. `wang2024coadain` — RGB-to-thermal automotive diversity, IEEE Sensors 2024.
39. `xia2025thermalgen` — ThermalGen flow-based RGB-to-thermal, arXiv:2509.24878.
40. `zhang2025sgfnet` — Spectral-aware global fusion RGB-T seg, arXiv:2505.15491.
41. `zhao2025shifnet` — SAM2 RGB-Thermal seg with language guidance, arXiv:2503.02581.
42. `zhou2023csrpnet` — Channel-spatial relation RGB-T seg, arXiv:2308.12534.

### Bibliography fix list (before WACV submission)

1. **Cite or remove `vertens2020heatnet`** — natural fit for §2 multimodal aerial datasets paragraph.
2. **Cite or remove `zhang2018lpips`** — LPIPS metric; cite in §5 if discussed, else remove.
3. **Fix BibTeX types:** `detone2016homography` → `@article`, `ipcc2022wgii` → `@techreport`, `ouyang2025kust4k` → `@misc`.
4. **Verify 2025/2026 arXiv IDs**: `zhan2025satellite` (2509.16568), `xia2025thermalgen` (2509.24878), `zhang2025sgfnet` (2505.15491), `zhao2025shifnet` (2503.02581), `canitez2026frequency` (2605.26273), `sherpa2026conditional` (2605.17564). Open each in browser.
5. **`sherpa2026conditional`** — **Prof. Siwo personally verifies** since he's the senior author.
6. **Add DOI fields** during camera-ready (deferred).

---

## 14. How to write each paper section

### 14.1 Abstract (150-180 words)

Already drafted; checklist guarded. Make sure every sentence in the abstract maps to a `Supported` row in `no_overclaiming_checklist.md`. The sentence-by-sentence audit is built in — use it.

**Key claims that MUST appear:**
- "+0.571 ± 0.157 dB three-seed gain over matched no-registration."
- "uncertainty-decoupled" (don't say "uncertainty-free" or "deterministic").
- "synthetic warp-recovery supervision" (don't say "learned registration" alone — too broad).
- Kust4K null result mentioned (don't only mention the AA gain).

### 14.2 Introduction (~0.6 pages)

- Climate motivation → urban thermal proxies are useful but flawed → alignment and target conventions matter.
- The narrowed contribution: source-dataset robustness method + empirical protocol study.
- Five-point contribution list:
  1. Reproducible multi-dataset harness (AA, Kust4K, CART).
  2. Synthetic misalignment diagnostic — AA is alignment-sensitive, externals are weaker and target-confounded.
  3. Lightweight input-space affine head + direct RGB warp-recovery supervision.
  4. Uncertainty-decoupled controlled ablation.
  5. Qualitative figures including method-specific failures.

### 14.3 Related Work (~0.7 pages)

Four buckets (all cited already):

- **Paired/unpaired RGB-to-thermal & I2I:** isola2017pix2pix, zhu2017cyclegan, wang2024coadain, xia2025thermalgen, sherpa2026conditional, ronneberger2015unet, he2016resnet.
- **Multimodal aerial datasets:** hwang2015kaist, liu2016multispectral, kemker2017multispectral, shivakumar2019pst900, jia2021llvip, zhou2023csrpnet, lee2024cart, ouyang2025kust4k, zhang2025sgfnet, zhao2025shifnet, canitez2026frequency.
- **Cross-modal registration:** jaderberg2015stn, detone2016homography, dai2017deformable, balakrishnan2019voxelmorph.
- **Uncertainty in image translation:** gal2016dropout, kendall2018multi.

The current draft handles all four buckets. Make sure `vertens2020heatnet` lands in the aerial/multimodal bucket if you decide to keep it.

### 14.4 Datasets (~0.8 pages)

- Use Table 1 (Datasets) directly.
- Mention the recovered scalar target for AA explicitly (Discovery 1 from §2.1 above).
- Mention Kust4K's broken-sample exclusion explicitly (auditability).
- Mention CART's labeled subset specifically (not the full archive).
- Cite all three datasets where they appear (`lee2024cart`, `ouyang2025kust4k`).
- Add the urban-heat tail of citations in §3: liu2023island, martin2022iris, zhan2025satellite.

### 14.5 Method (~0.7 pages)

- Use Figure 2 (method diagram) here.
- Loss equation (§9.1 above).
- Training protocol summary (§4.2).
- Explicitly call out "uncertainty-decoupled" and explain why.

### 14.6 Experiments (~1.5 pages)

Subsections:
- 5.1 Synthetic Misalignment Diagnostics (Week 2.5 numbers)
- 5.2 Registration Variants (Week 3-4 unsupervised failures)
- 5.3 Baselines (Table 2)
- 5.4 Main Ablation (Table 3, the headline)
- 5.5 External and Transfer Results (Table 4)
- 5.6 Qualitative Results (Figures 1, 3, 4)

### 14.7 Discussion (~0.5 pages)

- Synthetic warp supervision works on AA but not universally.
- Uncertainty weighting hurts (genuine finding).
- External datasets are limited by target representation, not just architecture.
- The narrowed framing (source-dataset robustness + empirical protocol study).

### 14.8 Limitations (~0.3 pages)

- Synthetic, not real-world geometry.
- Small dataset (AA 336 train).
- Single-backbone evidence (ConvNeXt; Swin-T stacking is null).
- Smooth predictions around high-frequency details.
- Uncertainty branch is decorative (not validated as calibrated).
- Robust normalization changes PSNR scale (legacy 19.28 dB not directly comparable).

### 14.9 Conclusion (~0.2 pages)

Tight: synthetic warp supervision provides a small but repeatable improvement on an alignment-sensitive source dataset. The best variant is uncertainty-decoupled. The gain is real, narrow, and reproducible.

---

## 15. Anti-overclaim phrasebook

The no-overclaiming checklist enforces these patterns. Use them; avoid the alternatives.

| ✅ Use | ❌ Avoid |
|---|---|
| "synthetic warp-recovery supervision" | "learned registration" (too broad) |
| "uncertainty-decoupled" | "deterministic" (wrong — branch still computes u) |
| "diagnostic uncertainty map" | "calibrated uncertainty" |
| "small but repeatable gain" | "significant improvement," "boost," "outperforms" |
| "qualitative cross-dataset context" | "cross-dataset generalization" |
| "source-dataset robustness" | "universal RGB-to-thermal method" |
| "paired three-seed delta of +X.XX ± X.XX dB" | "+X.XX dB improvement" |
| "Kust4K does not show a statistically meaningful within-dataset gain" | "K4K is harder" or just omitting |
| "CART gains are loss-balance-sensitive" | "CART benefits from registration" |
| "illustrative upper-tail example" (for hero) | "typical result" |
| "selected by most negative paired PSNR delta" (failure cases) | "lowest-PSNR" or "hardest" |
| "non-monotonic; diagnostic only" (severity curve) | presenting it as a paper figure |

---

## 16. Reviewer Q&A preparation

Anticipated reviewer questions and how to answer.

### Q1. Why isn't your method just learned spatial transformer (Jaderberg 2015)?

**A:** STN is the architectural ancestor. Our contribution is the **training signal**: direct synthetic warp-recovery supervision via `L1(x_warp, x_aligned)`, where `x_aligned` is teacher-forced. The unsupervised STN variants we tested (Week 4 `input_rgb_flow`, `shared_rgb`) failed to beat no-registration. The supervised warp loss is what makes the difference.

### Q2. Why use ConvNeXt-tiny instead of a transformer backbone?

**A:** Three reasons. (1) Under the locked uncertainty-decoupled protocol, our ConvNeXt method actually **beats** Swin-T no-reg by +0.215 ± 0.113 dB (paired 3-seed, p ≈ 0.04, marginally significant); the +0.096 ± 0.096 dB Swin-T advantage from the Week 6 audit was against the OLDER uncertainty-weighted variant, which we no longer treat as the primary method. (2) Swin-T affine stacking is null (−0.064 ± 0.214 dB), so the method is backbone-dependent and we don't claim universality. (3) ConvNeXt-tiny + UNet is a strong, simple, widely-deployable baseline; demonstrating gains on it is a useful contribution for downstream practitioners.

### Q3. Why is uncertainty weighting bad?

**A:** Honestly, we don't fully understand. Our finding is empirical: uncertainty-weighted L1 + the two regularizers (mean(u), TV(u)) consistently underperform unweighted L1 with zero regularizer weights by ~0.39 dB at seed 42. We hypothesize that on small datasets with already-stable training, the additional regularization terms are pulling capacity away from reconstruction. We present this as an ablation-driven design choice, not a theoretical claim about uncertainty methods.

### Q4. Your gains are small. Why publish?

**A:** Three reasons. (1) The gain is **statistically significant** (paired t=6.30, p=0.012, df=2). (2) The negative findings (uncertainty weighting hurts, registration doesn't stack on Swin-T, Kust4K is null) are **scientifically valuable** for practitioners. (3) The empirical contributions (synthetic misalignment diagnostic protocol; target normalization audit; reproducible 3-seed paired-delta methodology) outlive the specific architectural finding.

### Q5. Why not test real-world misalignment instead of synthetic?

**A:** We agree this is a limitation (explicitly disclosed). Real-world alignment ground truth would require either (a) calibrated paired UAV captures with known relative camera geometry, or (b) human-annotated correspondences. Neither was available within the project scope. Synthetic misalignment lets us run controlled, reproducible experiments. The validity of the conclusion (synthetic warp supervision helps when alignment is the bottleneck) generalizes within the synthetic-misalignment paradigm.

### Q6. Why no SwinIR / Restormer baseline?

**A:** Resource constraints. We compare against the Swin-T encoder + U-Net decoder (a transformer-backbone proxy). True restoration networks would strengthen the baseline panel. Future work. The current comparison is fair within the "translation network" family.

### Q7. How sensitive is the method to lambda_warp?

**A:** On CART (the most lambda-sensitive dataset), the gain ranges from +0.222 dB at λ=0.1 to +1.257 dB at λ=2.0. We selected λ=0.5 as a locked compromise (stays above the 0.3 dB threshold, avoids loss-dominance). On Ann Arbor, the +0.571 dB gain is at λ=0.5; we did not sweep on AA.

### Q8. What about the 19.28 dB hackathon ensemble result?

**A:** That number is on **raw target normalization** with ensemble + TTA. The Week 7 robust-normalization protocol changes the PSNR scale (per-sample 1st-99th percentile rescaling expands dynamic range). The two numbers are not directly comparable. We document the legacy 19.28 dB as historical context but do not include it in the same-y-axis paper tables.

### Q9. What's the failure mode in the failure-case figure?

**A:** The dominant failure mode is **smoothing of high-frequency thermal structure** around roofs, vehicles, and hard building edges. The uncertainty maps do not fully localize these errors (another reason we treat uncertainty as diagnostic-only, not calibrated).

### Q10. Why is the cross-dataset gallery in the appendix?

**A:** Because the Ann Arbor row uses the locked Week 7 robust protocol, while the Kust4K and CART rows use older within-dataset checkpoints from earlier weeks (and raw target normalization). The protocols are not unified across the gallery rows. The gallery is qualitative context only, with explicit in-figure disclaimers.

---

## 17. Submission checklist

### 17.1 Before Aug 28 2026 (WACV R2)

- [ ] Internal review (Adithya + Santosh) complete with paragraph marks.
- [ ] Prof. Siwo's 5-question framing review complete.
- [ ] Claim-tightening pass through `main_draft.md` based on review marks.
- [ ] LaTeX conversion (Pandoc + biblatex/natbib).
- [ ] 3 BibTeX type fixes + 2 dead-entry decisions.
- [ ] 5-6 recent arXiv IDs verified.
- [ ] DOI fields populated.
- [ ] Final figure pass — replace any remaining PIL bitmap labels with TrueType.
- [ ] Re-render figures at final resolution (300 dpi or PDF vector).
- [ ] No-overclaiming checklist pass on every paragraph.
- [ ] Word count / page count within WACV R2 limit.
- [ ] arXiv preprint (consider parallel posting).
- [ ] Submit to WACV R2 by Aug 28 2026 AoE.

### 17.2 Before Sep mid-2026 (CCAI fallback, if CFP appears)

- [ ] Verify CCAI 2026 CFP page limit, format, supplementary policy.
- [ ] Convert `ccai_short_draft.md` to LaTeX with confirmed format.
- [ ] Cut to confirmed page limit.
- [ ] Submit.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **Ann Arbor (AA)** | Our primary proprietary UMich drone dataset |
| **Kust4K (K4K)** | Public Korean UAV RGB-TIR dataset (4024 pairs) |
| **CART** | Caltech aerial RGB-thermal dataset (labeled subset 2282 pairs) |
| **Robust normalization** | Per-sample 1st-99th percentile rescale of target to [0, 1] |
| **Histmatch normalization** | Map dataset's target quantiles to AA reference quantiles |
| **Raw normalization** | Original target unchanged |
| **Amplified sigma** | max_translation_frac=0.20, max_rotation_deg=20°, max_scale_frac=0.25 |
| **sigma** | Scalar in [0,1] scaling the misalignment range |
| **theta** | 2×3 affine matrix (6 parameters: 4 linear + 2 translation) |
| **Identity init** | Final theta head's weights zeroed; bias set to [1,0,0,0,1,0] |
| **theta_l2** | Mean L2 distance of predicted theta from identity (indicates warp magnitude) |
| **Warp recovery** | Direct L1 supervision: `L1(x_warp, x_aligned)` |
| **Uncertainty-decoupled** | Disable both the L1 weighting by u and the two u regularizers |
| **Paired delta** | Per-seed (affine − no-reg), then mean and std across seeds |
| **Same-family delta** | Paired delta within ConvNeXt or within Swin-T family only |
| **Loss recipe** | `registration` (our default) or `swin_combined` (the Swin-T baseline recipe) |
| **Threshold (0.3 dB)** | Pre-Week 4 audit-defined heuristic: a gain below 0.3 dB is "weak" |

---

## 19. What I'm asking from you

The review packet is at `paper_draft/review_packet.md`. Quick version:

### 19.1 Files to read (in order)

1. **`WEEKLY_HANDOFF_FOR_ADITHYA.pdf`** (this document) for context.
2. **`paper_draft/main_draft.md`** — the full WACV draft.
3. **`paper_draft/tables_and_captions.md`** — Tables 1-4 + figure captions.
4. **`paper_draft/no_overclaiming_checklist.md`** — the wording rules.
5. **`paper_draft/method_diagram.svg`** — open in browser.
6. **`figures/week8/*_polished_seed42.png`** — preview figures.

### 19.2 How to review

Mark every paragraph in `main_draft.md` with one of:

- **Clear** — paragraph is good as-is.
- **Unclear** — wording confuses me on first read.
- **Wrong** — claim is factually wrong against the data.
- **Too strong** — claim is technically supported but feels overclaimed.
- **Needs citation** — a claim needs a reference and doesn't have one.

**Please don't line-edit yet.** The priority is claim shape, not phrasing.

### 19.3 What to specifically look for

- Anywhere I say "outperforms" or "beats" without saying paired and seeded.
- Anywhere I describe the K4K result without saying "null."
- Anywhere a figure caption could be misread as a typical result rather than a single-sample illustration.
- Anywhere I describe the method as "registration" without saying "supervised synthetic warp recovery."
- Anywhere a 2025/2026 citation is critical and you can recall whether the paper is real.
- The Swin-T discussion — is the same-family vs vs-ConvNeXt-no-reg distinction in Table 3 readable?

### 19.4 Big-picture questions for our shared judgment

1. Should the CCAI workshop branch happen at all? If we get WACV accepted, the CCAI version is busywork. If we get WACV rejected, CCAI is our safety net. Worth keeping the parallel work warm?
2. Is the hero figure the right hero figure? It's a single AA scene; the method's gain is +0.93 dB on this sample versus +0.571 mean. We could pick a more typical-PSNR sample instead.
3. The Swin-T comparison — Table 3 shows +0.215 dB above Swin-T no-reg with p ≈ 0.04. That's marginal. Are we comfortable putting it in the abstract, or move it to §5.3?
4. The uncertainty story — "uncertainty weighting hurt in this protocol" is a real finding but framed cautiously. Should we lean harder on it as a contribution, or keep it as an ablation note?
5. Anything missing from Related Work? 40 papers in 4 main buckets — anything obvious you'd add or cut?

### 19.5 Verification

Every number in this PDF traces back to a CSV in `results/`. To verify:

```bash
# Headline +0.571 dB
grep "convnext_affine_unc_decoupled" results/week7_ablation_summary.csv
# Headline +0.215 dB
python build_week9_tables.py --stdout | grep "Swin-T no-reg"
# Kust4K null +0.096 ± 0.067 dB
grep "kust4k" results/week5_preflight_registration_summary.csv
# AA → K4K transfer +0.474 ± 0.061 dB
cat results/week5_aa_to_kust4k_3seed_summary.csv
```

---

*Hit me up if anything is unclear. The honest version of the project story has been hammered into shape by 11 weeks of audits; please push back on anything you think is over- or under-stated.*

— Santosh
