# Week 10 CCAI Workshop Plan

Started: 2026-06-17

## Deadline Status

The CCAI 2026 NeurIPS workshop deadline is not officially posted yet.

- Official Climate Change AI events page checked on 2026-06-17:
  https://www.climatechange.ai/events
- That page lists 2026 summer/community events and past NeurIPS 2025, but no
  NeurIPS 2026 CCAI workshop page yet.
- NeurIPS 2026 lists workshops on Dec 11-12, 2026 and a suggested workshop
  contribution submission date of Aug 29, 2026 AoE:
  https://neurips.cc/Conferences/2026/Dates
- Prior CCAI NeurIPS workshop page checked for format context:
  https://www.climatechange.ai/events/neurips2025

Planning assumption: prepare the CCAI version as a four-page workshop paper,
but do not treat the CCAI deadline or format as final until the 2026 CCAI CFP
appears. Use Aug 29, 2026 AoE as the conservative planning date for workshop
contributions.

## Goal

Build a CCAI-ready short version from the Week 9 WACV draft without opening new
experiments. The workshop version should emphasize the climate/urban-heat
motivation and the empirical protocol lessons:

1. Aerial RGB-to-thermal translation is useful for urban heat and climate
   adaptation workflows when thermal capture is expensive or sparse.
2. Alignment, target representation, and dataset conventions materially affect
   what RGB-to-thermal models appear to learn.
3. Synthetic warp supervision gives a modest, repeatable source-dataset
   robustness gain on Ann Arbor.
4. External datasets are useful diagnostics, but they do not support broad
   cross-dataset registration claims in the current protocol.

## Locked Claim

Synthetic warp supervision improves Ann Arbor robustness under controlled
synthetic misalignment by `+0.571 +/- 0.157 dB` over a matched ConvNeXt
no-registration baseline across three seeds. The result is useful but narrow:
it is not an unsupervised registration claim and not a universal
RGB-to-thermal generalization claim.

## Four-Page Shape

Target structure:

1. Abstract: 150-180 words.
2. Introduction and climate motivation: 0.6 pages.
3. Data/protocol: 0.8 pages.
4. Method: 0.7 pages.
5. Experiments/results: 1.2 pages.
6. Limitations and climate relevance: 0.5 pages.
7. References: outside the page budget if allowed by the CFP; otherwise
   compress aggressively.

Primary figures/tables for the short version:

- One compact method + qualitative figure if space allows.
- One result table combining baselines and ablations.
- Dataset/protocol details may become prose instead of a full table.

## Must Keep

- The three-seed Ann Arbor main result.
- The uncertainty-decoupled design choice and the warning that uncertainty
  weighting hurt in this protocol.
- The Kust4K null result and CART loss-balance caveat.
- The CCAI climate framing: urban heat, thermal scarcity, and reliable
  multimodal remote-sensing evaluation.

## Must Cut Or Move To Appendix

- Long Week 2 diagnostic narrative.
- Full cross-dataset transfer matrix.
- Single-seed severity curve.
- Detailed Swin-T stacking discussion.
- Most qualitative candidates; keep only one figure panel if needed.

## Parallel Review

Internal review of the Week 9 WACV draft should proceed while this short draft
is prepared. Do not wait for the CCAI version before sending the Week 9 draft
for top-level framing review.

## Open Items

- Monitor the official CCAI events page for a NeurIPS 2026 CFP.
- Replace the planning deadline with the official CCAI 2026 deadline when it is
  posted.
- Confirm the final page limit, appendix policy, and review format from the
  2026 CFP before converting this scaffold to LaTeX.
