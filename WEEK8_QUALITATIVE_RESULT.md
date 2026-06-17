# Week 8 Qualitative Figures Result

Week 8 produced the qualitative figure set for the current paper draft. The
figures are committed under `figures/week8/` and all PNGs were regenerated with
300 dpi metadata.

## Generated Figures

| Figure | File | Purpose |
|---|---|---|
| Ann Arbor candidate grid | `figures/week8/ann_arbor_candidate_grid_seed42.png` | Broad validation quantiles: RGB, learned warp, target, prediction, error, uncertainty. |
| Misalignment recovery candidates | `figures/week8/misalignment_recovery_seed42.png` | Before/after warp residuals for selected Ann Arbor rows. |
| Hero candidate | `figures/week8/hero_ann_arbor_seed42.png` | Single scene with road, trees, building edge, hot roof, prediction, uncertainty, and error. |
| Multi-sigma recovery | `figures/week8/misalignment_recovery_multisigma_seed42.png` | Same Ann Arbor scene at sigma 0.0, 0.2, and 0.5 with baseline vs ours. |
| Cross-dataset gallery | `figures/week8/cross_dataset_gallery_seed42.png` | Ann Arbor, Kust4K, and CART representative rows with target, ours, and baseline. |
| Failure cases | `figures/week8/failure_cases_ann_arbor_seed42.png` | Ann Arbor samples where ours underperforms the no-registration baseline most. |

Metrics for the final figures are in
`figures/week8/week8_final_figure_metrics.csv`.

## Key Observations

- The hero scene is a reasonable paper candidate: ours improves over
  ConvNeXt no-registration on that sample (`16.94 dB` vs `16.01 dB`) and shows
  both useful structure and clear smoothing failure.
- Paper-draft guardrail: the hero's `+0.93 dB` gain is an illustrative upper
  tail example, not the typical effect size. Captions must point readers to the
  three-seed mean gain (`+0.571 +/- 0.157 dB`) and avoid implying a `~1 dB`
  average improvement.
- The multi-sigma recovery figure is honest. On the selected scene, ours helps
  at sigma `0.0` and `0.2`, but not at sigma `0.5` (`13.84 dB` ours vs
  `13.96 dB` baseline). Do not use this as a monotonic success figure.
- The failure cases now target method-specific underperformance: samples where
  ours has the most negative paired PSNR delta relative to the no-registration
  baseline. This is distinct from the earlier "hard scenes for both models"
  selection.
- The cross-dataset gallery should be framed as qualitative context only.
  Ann Arbor uses the locked Week 7 robust uncertainty-decoupled protocol, while
  Kust4K and CART rows use older within-dataset external checkpoints. They are
  not a unified quantitative cross-dataset protocol.
- The Kust4K row has an in-figure warning because the three-seed Kust4K gain is
  not significant (`+0.096 +/- 0.067 dB`). Do not use the single-sample visual
  win as evidence that registration helps on Kust4K.

## Figure Scripts

- `week8_make_qualitative_figures.py`: candidate Ann Arbor grids.
- `week8_make_final_figures.py`: hero, multi-sigma recovery, cross-dataset
  gallery, failure cases, and final figure metrics.

## Remaining Paper-Time Work

- Select the final hero panel and crop/annotate it for the paper layout.
- Decide whether the cross-dataset gallery belongs in the main paper or the
  appendix, because the external rows are qualitative-only.
- If the multi-sigma recovery figure is used in the main paper, repeat it over
  more scenes or keep it explicitly as a representative example.
