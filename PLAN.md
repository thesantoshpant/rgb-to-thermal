# Summer 2026 plan — WACV 2027 (R2) + CCAI safety net

**Locked:** 2026-06-something. Open this file at the start of every work session.

## The paper, in one sentence
> **Synthetic warp augmentation plus direct warp-recovery supervision can make aerial RGB-to-thermal translation more robust under weak alignment, but cross-dataset gains depend strongly on target representation.**

## Targets and hard dates
| Milestone | Date | Why |
|---|---|---|
| Go/no-go experiment result | end of **Week 2** | Decides if WACV story has a backbone |
| WACV 2027 R2 paper **registration** | **Aug 21, 2026** | Required to submit R2 |
| WACV 2027 R2 paper **submission** | **Aug 28, 2026** | Primary target |
| CCAI @ NeurIPS submission (est.) | mid-September 2026 | Non-archival safety net |
| WACV 2027 decisions | **Oct 9, 2026** | |
| WACV 2027 camera-ready | **Nov 2, 2026** | |
| WACV 2027 conference | **Jan 5–9, 2027** (Disney Springs) | |

## Strategy in one paragraph
Submit primarily to **WACV 2027 R2 (Aug 28, 2026), Algorithms track** only if Week 5 turns the Week 4 synthetic-supervised signal into a robust cross-dataset story. The current mechanism is not unsupervised registration discovering a bottleneck; it is synthetic warp pretraining / auxiliary supervision. In parallel, submit a shorter version to **CCAI @ NeurIPS Dec 2026** (non-archival, does not block future publication) so we always have a workshop paper around alignment diagnostics, target normalization, and aerial urban heat as a floor.

## Team and split
- **Santosh:** core engineering (data loaders, registration module, training infrastructure, evaluation harness).
- **Adithya:** experiment driving (ablations, baselines, sweeps), figures, writing first drafts.
- **Joint:** paper outline, ablation interpretation, final paper polish.
- **Prof. Siwo:** scope feedback, application framing, venue advice, final paper review.

## What we are NOT doing (rejected last round, do not revisit)
- Physics-informed generator with calibrated temperature/emissivity (we don't have the data).
- Foundation-model pretraining at scale (we don't have the compute).
- Adding more priors (depth, AlphaEarth, solar at full res) — all tested, do not help.
- Bigger backbones (ConvNeXt-S etc.) — do not help on 336 images.
- Per-image affine registration as a preprocessing step — does not help beyond the global crop (parallax is non-rigid).

---

## Weekly plan

> Each week: open this file, mark [x] on done items, write one-line **Blocker** and **Result** notes.

### Week 0 — Lock the plan (this week)
- [ ] Read this file with Adithya end-to-end.
- [ ] Confirm the task split.
- [ ] Confirm we have access to GPUs (re-check `nvidia-smi -i 0 -i 1`).
- [ ] Email Prof. Siwo: confirm summer collaboration cadence (weekly 30-min check-in or async?).
- [ ] Make sure `rgb-to-thermal` repo is checked out on both machines and the cluster.
- **Result:**
- **Blocker:**

### Week 1 — Reproduce + pull external data
**Goal:** unified data loader for **Ann Arbor + Kust4K + CART**; reproduce our current 19.28 dB.
- [x] Pull Kust4K (~4k pairs). Verify license and citation.
- [x] Pull Caltech CART. Verify license and citation.
- [x] Write `unified_dataset.py`: a single `Dataset` class that returns `(rgb_tensor, thermal_tensor, scalar_target, dataset_tag, alignment_quality_score)` for all 3 datasets.
- [x] Re-run `eval_v2.py` on official 202: must reproduce **19.28 ± 0.05 dB PSNR**.
- [x] Write `make_synthetic_misalignment.py`: takes an aligned RGB-thermal pair and applies controlled translation/scale/rotation perturbation, with a severity parameter `σ ∈ [0, 1]`.
- [x] Commit and push Week 1 scaffold/fixes to GitHub.
- **Result:** Reproduced weighted TTA ensemble at 19.28 dB PSNR on Knox; downloaded/extracted Caltech CART and Kust4K; unified loader reads Ann Arbor + Caltech + usable Kust4K train split as 336 + 1822 + 1970 samples; all-split counts are Caltech 2282 and Kust4K 2818 after excluding author-flagged broken stems; synthetic misalignment smoke test passed.
- **Blocker:** None.

### Week 2 — **GO/NO-GO experiment**
**Goal:** prove the bottleneck hypothesis on Kust4K + CART before we commit.
- [x] Train a small pix2pix baseline on Kust4K (aligned). Should match published numbers within reason.
- [x] Re-train the same model on the same data with **synthetic misalignment** at σ = {0.1, 0.2, 0.3, 0.5} applied at training time, measured against aligned test pairs.
- [x] Plot: **PSNR vs misalignment severity** (one line per dataset).
- [x] Also: do the same on CART.
- [x] Evaluate **GO criterion:** PSNR drops by ≥ 2 dB at σ = 0.3 on at least one dataset.
- [x] Evaluate **NO-GO branch:** drop < 1 dB, so run Week 2.5 diagnostics before any full pivot.
- [x] Write a one-page memo with the go/no-go decision and commit it.
- **Result:** Full Week 2 sweep completed on Knox. At σ=0.3, final PSNR drops were only 0.347 dB on Kust4K and 0.185 dB on CART, so the WACV registration-bottleneck story is provisional no-go under this narrow protocol only. See `WEEK2_GO_NO_GO_MEMO.md`, `results/week2_sweep_summary.csv`, and `figures/week2_psnr_vs_sigma.svg`.
- **Blocker:** The no-go signal is not trusted yet because the perturbation is small, the external targets are low-frequency, only one seed was run, and Ann Arbor was not tested. Complete Week 2.5 before pivoting.

### Week 2.5 — Diagnostics before pivot
**Goal:** decide whether Week 2 was a real no-go or a weak harness.
- [x] Run Ann Arbor sweep with σ = {0, 0.1, 0.2, 0.3, 0.5}.
- [x] Run validation-time misalignment: train σ=0, evaluate at σ = {0.1, 0.2, 0.3, 0.5}.
- [x] Run shuffled-RGB control to estimate thermal-prior-only PSNR.
- [x] Re-sweep with amplified perturbation range: max translation fraction 0.20, max rotation 20 deg, max scale fraction 0.25.
- [x] Run L1-only regression baseline alongside pix2pix.
- [x] Run 3 seeds for final decision cells and report mean ± std.
- [x] Update go/no-go memo: WACV story alive if any diagnostic shows a clean ≥ 2 dB drop; otherwise pivot with confidence.
- **Result:** Week 2.5 overturns the final no-go. Ann Arbor shows the expected alignment sensitivity: original σ=0.3 drop is 1.98 dB; amplified σ=0.3 three-seed mean drop is 2.61 dB; L1-only amplified drop is 2.50 dB. Kust4K/CART remain weaker, so they should be external generalization tests, not the sole go/no-go evidence. See `WEEK2_5_DIAGNOSTICS_RESULT.md` and `results/week2_5_diagnostics_summary.csv`.
- **Blocker:** Audit/improve Kust4K and CART target normalization before final external claims; current diagnostics still use raw normalized thermal grayscale for those datasets.

### Week 3 — Learned registration module v0
**Goal:** end-to-end trainable system that aligns *and* translates.
- [x] Architecture sketch: a small registration head that consumes RGB and thermal features and predicts (a) a parametric warp (start with affine), (b) a per-pixel "alignment uncertainty" map.
- [x] Loss: (1) photometric warp loss (warped RGB-derived edges ↔ thermal edges), (2) smoothness on the warp, (3) translation reconstruction weighted by `1 / (1 + uncertainty)`.
- [x] Plug v0 into the existing ConvNeXt+U-Net translator.
- [x] Share encoder features instead of using a separate registration head.
- [x] First training pass on Ann Arbor only; sanity-check on val.
- **Result:** Week 3 complete as an engineering milestone. `week3_registration_v0.py` has target-conditioned oracle, shared-RGB-feature affine registration, and no-registration baseline modes. On Ann Arbor amplified σ=0.3, no-registration reached PSNR 15.89, shared RGB feature v0 reached 15.81, and target-conditioned oracle reached 15.45; therefore v0 trains stably but does not yet beat the baseline. See `WEEK3_REGISTRATION_V0_RESULT.md` and `results/week3_registration_v0_summary.csv`.
- **Blocker:** Week 4 must improve over the no-registration baseline before we can claim learned registration recovers misalignment damage. Open before Week 5: improve Kust4K/CART target normalization.

### Week 4 — Registration module v1 + multi-dataset training
**Goal:** the model trains and improves over the matched no-registration baseline on at least one dataset.
- [x] Run same-protocol no-registration Ann Arbor baseline for the headline delta.
- [x] Start Kust4K/CART target-normalization audit.
- [x] Implement a target-normalization fix or choose the conservative reporting fallback.
- [x] Decide whether the primary registration mechanism should be input-space warp, feature-space warp, TPS, or small dense flow.
- [x] Test input-space affine as the interpretable paper-facing variant.
- [x] Test a small input-space dense-flow candidate.
- [x] Add known synthetic-warp supervision or an oracle inverse-warp diagnostic before more TPS/flow scaling.
- [x] Repeat supervised input-space affine with more seeds and decide whether to use it as pretraining/auxiliary loss.
- [x] Train on Kust4K. Compare against the matched no-registration baseline.
- [x] Train on CART. Same comparison.
- [x] **Decision point:** if learned registration beats matched no-registration by ≥ 0.3 dB on at least two datasets → continue. Else: revisit architecture or downgrade scope.
- **Result:** Complete, but marginal. Unsupervised shared-feature affine, input-space affine, and dense flow all underperformed no-registration on Ann Arbor. Adding synthetic RGB warp supervision to input-space affine gives a 3-seed Ann Arbor gain of +0.309 +/- 0.130 dB over no-registration. External same-dataset runs are mixed: Kust4K +0.147 dB is below threshold, while CART +1.196 dB is strong but single-seed and may be loss-balance-driven. Week 5 may continue only as a conditional follow-up around synthetic warp supervision, not as a clean unsupervised-registration story. See `WEEK4_REGISTRATION_V1_PROGRESS.md` and `results/week4_registration_v1_summary.csv`.
- **Blocker:** Week 5 must confirm Kust4K/CART over 3 seeds, run CART `lambda_warp_rgb` sensitivity, and lock target normalization before any cross-dataset transfer claim.

### Week 5 — Cross-dataset generalization
**Goal:** show whether the method transfers across datasets after target normalization.
- [x] Preconditions before transfer:
  - [x] Kust4K 3-seed no-registration vs supervised-affine confirmation.
  - [x] CART 3-seed no-registration vs supervised-affine confirmation.
  - [x] CART `lambda_warp_rgb` sensitivity sweep over `{0.1, 0.5, 1.0, 2.0}`.
  - [x] Target-normalization fix or explicit fallback locked before cross-dataset claims.
- [ ] Run the four transfer combinations:
  - Train Ann Arbor → test Kust4K
  - Train Kust4K → test Ann Arbor
  - Train Kust4K → test CART
  - Train CART → test Kust4K
- [x] Add `--eval-dataset` transfer support to `week3_registration_v0.py` and
  smoke-test Kust4K→CART with `target_normalization=robust`.
- [ ] Produce a transfer matrix table (rows = train, cols = test, cells = PSNR + SSIM).
- [ ] Pre-train on Kust4K + CART, fine-tune on Ann Arbor — does it beat our 19.28?
- **Preflight result:** Kust4K fails the registration-help threshold over three
  seeds (`+0.096 +/- 0.067 dB`). CART passes over three seeds
  (`+0.782 +/- 0.368 dB`) but is loss-balance-sensitive: the CART seed-42 delta
  falls to `+0.222 dB` at `lambda_warp_rgb=0.1` and rises to `+1.257 dB` at
  `lambda_warp_rgb=2.0`. Use `robust` target normalization for transfer
  diagnostics. Do not frame Week 5 as proof of a general unsupervised
  registration bottleneck.
- **Result:**
- **Blocker:** Cross-dataset transfer still needs the actual full matrix;
  preflight and the transfer smoke test only cleared the conditions for running
  it.

### Week 6 — Baselines
**Goal:** all required baselines reproduced fairly.
- [ ] Implement + train:
  - **pix2pix** (vanilla),
  - **CycleGAN** *or* **CUT** (pick one for unpaired/weakly-aligned comparison),
  - **ConvNeXt+U-Net regression** (we have it),
  - **A pretrained translation backbone** (Restormer or SwinIR) fine-tuned,
  - **Our current ensemble + TTA** (we have it).
- [ ] Match training compute across all baselines for fairness.
- [ ] Lock the baseline table; this is the "results" comparison table in the paper.
- **Result:**
- **Blocker:**

### Week 7 — Ablations
**Goal:** every claim in the paper has an ablation.
- [ ] No registration vs fixed crop vs learned-deterministic vs learned + uncertainty (full table).
- [ ] Uncertainty calibration on/off.
- [ ] Synthetic misalignment severity sweep (continuous curve).
- [ ] Loss term ablation (photometric / smoothness / uncertainty-weighted recon).
- [ ] Train-on-one-test-on-others (already in week 5; format for paper).
- [ ] Persistence of gains without TTA / without ensemble (important).
- **Result:**
- **Blocker:**

### Week 8 — Qualitative figures + failure cases
**Goal:** the figures that win or lose the paper.
- [ ] Hero figure: a single scene with RGB, GT thermal, our prediction, plus the uncertainty map. Pick something with a hot roof + cool tree + a building edge.
- [ ] Cross-dataset gallery: three rows (Ann Arbor / Kust4K / CART) × four columns (RGB / GT / ours / best baseline).
- [ ] Misalignment recovery figure: same scene with synthetic misalignment at σ ∈ {0, 0.2, 0.5}, our prediction vs baseline.
- [ ] Failure cases: at least 2 honest failures (e.g., parallax-heavy buildings; specular reflections).
- [ ] All figures committed in `figures/` at vector-quality (PDF) or 300+ dpi.
- **Result:**
- **Blocker:**

### Week 9 — Paper writing draft 1
**Goal:** full draft submitted to ourselves and Prof. Siwo.
- [ ] Outline → section sketches → first full draft.
- [ ] Sections: Abstract, Intro, Related Work (RGB-T translation, cross-modal registration, uncertainty in translation), Method, Experiments, Discussion, Limitations, Conclusion.
- [ ] Internal review by Adithya + Santosh (each marks every paragraph: clear / unclear / wrong).
- [ ] Send to Prof. Siwo by end of week, ask for top-level feedback on framing, not line edits.
- **Result:**
- **Blocker:**

### Week 10 — CCAI workshop submission
**Goal:** lock the workshop paper. Don't let it eat the WACV polish.
- [ ] Shorten the WACV draft to **4 pages** + appendix, reframed for the **climate/urban-heat** angle.
- [ ] Use the *empirical* finding (palette + alignment + cross-dataset evaluation) as the contribution at workshop; keep the *methodological* learned-registration angle for WACV.
- [ ] Submit to CCAI by its deadline (verify CCAI 2026 deadline in early September).
- [ ] Cap at 1 week of work; do not let polishing eat into WACV.
- **Result:**
- **Blocker:**

### Week 11 — Paper draft 2 + figures polish
**Goal:** the paper is "almost done."
- [ ] Address Prof. Siwo's feedback.
- [ ] Tighten claims; remove anything we can't defend.
- [ ] Re-do the worst 2 figures.
- [ ] Cite recent work end-to-end (RGB-T, registration, uncertainty in translation, aerial). At least 40 citations.
- [ ] Run a "no overclaiming" pass: every sentence in the abstract must be defensible by a specific experiment.
- **Result:**
- **Blocker:**

### Week 12 — WACV submission week
**Goal:** submit WACV R2 by Aug 28.
- [ ] Final writing pass; check all tables and numbers against the latest experiments.
- [ ] Supplementary material: extra ablations, training details, qualitative gallery, dataset card.
- [ ] **WACV registration: Aug 21, 2026.** Do not miss this.
- [ ] **WACV submission: Aug 28, 2026.**
- [ ] Take 1 day off after submission.
- **Result:**
- **Blocker:**

---

## After submission
- **Sept 1–21:** finalize and submit CCAI workshop version.
- **Oct 9:** WACV decision. If accept → camera-ready by Nov 2.
- **If reject:** address reviewer comments, target WACV 2028 R1 (Jun 2027 deadline) or another venue with stronger story.

## Risk register (re-read at the start of every week)
1. **The single-flight dataset reads as a toy problem to a tough reviewer.** Mitigation: multi-dataset evaluation is the entire paper, not a sidebar.
2. **The learned registration could collapse to identity.** Mitigation: photometric warp loss + smoothness + an explicit penalty on degenerate warps; run sanity check at end of Week 4.
3. **Time loss to two papers.** Mitigation: workshop version is a *subset* of the WACV draft, not a different story. Cap at 1 week.
4. **GPU contention on Knox.** Mitigation: small experiments on GPU 1 if 0 is busy; we did this before. Save checkpoints frequently.
5. **Scope drift into physics/foundation-model territory.** Mitigation: this file says no. Re-read it at the start of every week.

## Files and where they live
- Code: this repo, https://github.com/Santoshpant23/rgb-to-thermal
- Cluster work area: `/home/spant/UMich/umich-hackathon/rgb2thermal/` and a new `rgb2thermal_wacv/`
- Data caches: `data_cache/` on cluster
- This plan: `PLAN.md` at the repo root — always the source of truth.

## How we work
- Weekly check-in: Sunday evening, 30 min. Each person fills in Result/Blocker for the past week and reads the upcoming week aloud.
- Mid-week sync (optional): Wednesday, 15 min, on what's stuck.
- Anything that's not in this plan needs an explicit decision before it gets done.
