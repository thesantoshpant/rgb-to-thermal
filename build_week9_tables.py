#!/usr/bin/env python3
"""Build paper-draft tables for Week 9 from committed result CSVs."""
from __future__ import annotations

import csv
import statistics as stats
from pathlib import Path
import sys


RESULTS = Path("results")
OUT = Path("paper_draft")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row: dict[str, str], key: str) -> float:
    val = row.get(key, "")
    return float(val) if val not in ("", None) else float("nan")


def mean_std(vals: list[float]) -> tuple[float, float]:
    return stats.mean(vals), stats.stdev(vals) if len(vals) > 1 else 0.0


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def dataset_table() -> str:
    norm = read_csv(RESULTS / "week5_target_norm_audit_robust.csv")
    edge = {r["dataset"]: fnum(r, "edge_mean") for r in norm}
    rows = [
        ["Ann Arbor", "336 / 41 / 42", "Recovered scalar thermal", "robust", fmt(edge.get("ann_arbor", 0.0), 3), "Primary source-dataset protocol"],
        ["Kust4K", "1970 / 283 / 565", "Raw grayscale TIR", "robust for transfer; raw in legacy qualitative row", fmt(edge.get("kust4k", 0.0), 3), "Official splits minus author-flagged broken stems"],
        ["CART", "1822 / 222 / 238", "Raw grayscale thermal", "robust for transfer; raw in legacy qualitative row", fmt(edge.get("caltech_cart", 0.0), 3), "Full labeled_rgbt_pairs archive"],
    ]
    return (
        table(["Dataset", "Train / Val / Test", "Target", "Normalization", "Edge mean", "Paper use"], rows)
        + "\n\n"
        + "Note: Kust4K and CART test counts were spot-checked with `UnifiedR2TDataset.from_roots(..., split=\"test\")`."
    )


def baseline_table() -> str:
    week6 = read_csv(RESULTS / "week6_baseline_summary.csv")
    by_label = {r["label"]: r for r in week6}
    week7 = read_csv(RESULTS / "week7_ablation_summary.csv")
    primary = [r for r in week7 if r["family"] == "convnext_affine_unc_decoupled"]
    primary_psnr, primary_std = mean_std([fnum(r, "final_psnr") for r in primary])
    rows = []
    for label in ["CycleGAN", "pix2pix", "Small U-Net L1", "ConvNeXt+U-Net", "Swin-T+U-Net"]:
        r = by_label[label]
        ssim = fmt(fnum(r, "final_ssim"), 3) if r.get("final_ssim") else "-"
        corr = fmt(fnum(r, "final_corr"), 3) if r.get("final_corr") else "-"
        note = r["note"]
        if label == "CycleGAN":
            note += "; single seed, unstable on this small supervised setting"
        rows.append([label, r["family"], r["seed"], fmt(fnum(r, "final_psnr"), 3), ssim, corr, note])
    rows.append([
        "Ours: ConvNeXt affine, uncertainty-decoupled",
        "paired_regression + affine",
        "42/7/123",
        f"{fmt(primary_psnr, 3)} +/- {fmt(primary_std, 3)}",
        "-",
        "-",
        "Primary 3-seed method; not directly single-seed comparable to rows above",
    ])
    return table(["Method", "Family", "Seed(s)", "PSNR", "SSIM", "Pearson r", "Note"], rows)


def main_ablation_table() -> str:
    rows = read_csv(RESULTS / "week7_ablation_summary.csv")
    families = {
        "convnext_no_reg": "ConvNeXt no-registration",
        "convnext_affine_unc": "ConvNeXt affine + uncertainty weighting",
        "convnext_affine_unc_decoupled": "ConvNeXt affine, uncertainty-decoupled",
        "swin_no_reg": "Swin-T no-registration",
        "swin_affine": "Swin-T affine",
    }
    psnrs = {k: [fnum(r, "final_psnr") for r in rows if r["family"] == k] for k in families}
    convnext_baseline = psnrs["convnext_no_reg"]
    family_baseline = {
        "convnext_no_reg": psnrs["convnext_no_reg"],
        "convnext_affine_unc": psnrs["convnext_no_reg"],
        "convnext_affine_unc_decoupled": psnrs["convnext_no_reg"],
        "swin_no_reg": psnrs["swin_no_reg"],
        "swin_affine": psnrs["swin_no_reg"],
    }
    out_rows = []
    for key, label in families.items():
        m, s = mean_std(psnrs[key])
        if key == "convnext_no_reg":
            delta_convnext = "-"
        else:
            paired = [a - b for a, b in zip(psnrs[key], convnext_baseline)]
            dm, ds = mean_std(paired)
            delta_convnext = f"{dm:+.3f} +/- {ds:.3f}"
        if key in ("convnext_no_reg", "swin_no_reg"):
            delta_family = "-"
        else:
            paired_family = [a - b for a, b in zip(psnrs[key], family_baseline[key])]
            fm, fs = mean_std(paired_family)
            delta_family = f"{fm:+.3f} +/- {fs:.3f}"
        out_rows.append([label, str(len(psnrs[key])), f"{m:.3f} +/- {s:.3f}", delta_convnext, delta_family])
    return (
        table(
            ["Variant", "Seeds", "PSNR mean +/- std", "Delta vs ConvNeXt no-reg", "Same-family registration delta"],
            out_rows,
        )
        + "\n\n"
        + "Note: Swin-T deltas versus ConvNeXt include the encoder/decoder-family change. "
        + "The direct Swin-T affine minus Swin-T no-registration delta is shown in the same-family column."
    )


def external_table() -> str:
    pre = read_csv(RESULTS / "week5_preflight_registration_summary.csv")
    transfer = read_csv(RESULTS / "week5_transfer_matrix_summary.csv")
    aa_k4k = read_csv(RESULTS / "week5_aa_to_kust4k_3seed_summary.csv")

    def family_delta(dataset: str) -> str:
        vals = [
            fnum(r, "delta_vs_no_registration")
            for r in pre
            if r["dataset"] == dataset
            and r["delta_vs_no_registration"]
            and "warprgb1" in r["run"]
        ]
        m, s = mean_std(vals)
        return f"{m:+.3f} +/- {s:.3f}"

    aa_vals = [fnum(r, "delta_vs_no_registration") for r in aa_k4k if r["delta_vs_no_registration"]]
    aa_m, aa_s = mean_std(aa_vals)
    rows = [
        ["Kust4K within-dataset", family_delta("kust4k"), "3", "No positive claim; CI overlaps zero"],
        ["CART within-dataset", family_delta("caltech_cart"), "3", "Passes mean threshold but loss-balance-sensitive"],
        ["Ann Arbor -> Kust4K transfer", f"{aa_m:+.3f} +/- {aa_s:.3f}", "3", "Only transfer cell that survives a 3-seed audit"],
    ]
    for label, train, eval_ds in [
        ("Kust4K -> Ann Arbor transfer", "kust4k", "ann_arbor"),
        ("Kust4K -> CART transfer", "kust4k", "caltech_cart"),
        ("CART -> Kust4K transfer", "caltech_cart", "kust4k"),
    ]:
        vals = [fnum(r, "delta_vs_no_registration") for r in transfer if r["dataset"] == train and r["eval_dataset"] == eval_ds and r["delta_vs_no_registration"]]
        rows.append([label, f"{vals[0]:+.3f}" if vals else "-", "1", "Single-seed diagnostic only"])
    return table(["External experiment", "Delta PSNR", "Seeds", "Interpretation"], rows)


def captions() -> str:
    return """## Draft Figure Captions

**Figure 1. Hero qualitative example.** Ann Arbor validation scene with visible
road, tree canopy, and hot roof/building edges. The method improves this sample
from `16.01 dB` to `16.94 dB`, but this is an illustrative upper-tail example;
the main three-seed gain is `+0.571 +/- 0.157 dB`.

**Figure 2. Method overview.** Synthetic RGB misalignment is applied while the
thermal target remains fixed. A lightweight RGB-only affine head predicts an
input-space warp, the warped RGB is translated by a ConvNeXt-tiny U-Net, and an
auxiliary warp-recovery loss supervises the predicted warp. The uncertainty map
is logged for diagnostics but is not used for reconstruction weighting in the
primary variant.

**Figure 3. Multi-sigma recovery.** Same Ann Arbor scene evaluated at synthetic
misalignment sigma `0.0`, `0.2`, and `0.5`. The method helps at sigma `0.2` on
this scene but loses slightly at sigma `0.5`, so this figure should be read as
representative behavior, not a monotonic severity claim.

**Figure 4. Failure cases.** Ann Arbor validation samples with the most negative
paired PSNR deltas for the method relative to the no-registration baseline. The
dominant failure is smoothing of high-frequency thermal structure around roofs,
vehicles, and building edges; uncertainty maps do not fully localize the errors.

**Appendix Figure A. Candidate grid.** PSNR-quantile validation examples showing
RGB input, learned warp, target, prediction, absolute error, and uncertainty.

**Appendix Figure B. Cross-dataset qualitative gallery.** Representative rows
for Ann Arbor, Kust4K, and CART. This is qualitative context only. The Kust4K
within-dataset gain is not statistically significant across seeds
(`+0.096 +/- 0.067 dB`), and Kust4K/CART rows use legacy within-dataset
checkpoints rather than the locked Ann Arbor Week 7 protocol.
"""


def main() -> None:
    sections = [
        "# Week 9 Paper Tables and Captions",
        "## Table 1. Datasets and Target Representation",
        dataset_table(),
        "## Table 2. Baseline Results",
        baseline_table(),
        "## Table 3. Main Registration Ablation",
        main_ablation_table(),
        "## Table 4. External and Transfer Results",
        external_table(),
        captions(),
    ]
    payload = "\n\n".join(section.rstrip("\n") for section in sections) + "\n"
    if "--stdout" in sys.argv:
        print(payload, end="")
        return
    OUT.mkdir(exist_ok=True)
    try:
        (OUT / "tables_and_captions.md").write_text(payload, encoding="utf-8")
    except PermissionError:
        print("warning: could not write paper_draft/tables_and_captions.md; printing to stdout", file=sys.stderr)
        print(payload, end="")
        return
    print(f"wrote {OUT / 'tables_and_captions.md'}")


if __name__ == "__main__":
    main()
