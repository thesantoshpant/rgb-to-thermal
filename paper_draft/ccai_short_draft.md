# Synthetic Warp Supervision for Aerial RGB-to-Thermal Translation in Urban Heat Workflows

## Abstract

Aerial thermal imagery can support urban heat analysis, infrastructure
inspection, and climate adaptation planning, but direct thermal capture is
often less available than RGB imagery. We study RGB-to-thermal translation under
the practical complications that appear in UAV data: imperfect RGB/TIR
alignment, target normalization choices, and dataset-specific thermal
conventions. Using an Ann Arbor UAV dataset and two external RGB-TIR datasets
(Kust4K and CART), we evaluate synthetic misalignment diagnostics and a
lightweight RGB-only affine registration head trained with synthetic
warp-recovery supervision. On Ann Arbor under amplified synthetic
misalignment, the final uncertainty-decoupled affine model improves over a
matched ConvNeXt no-registration baseline by `+0.571 +/- 0.157 dB` across three
seeds. External results are mixed: Kust4K shows no meaningful within-dataset
registration gain, while CART gains are sensitive to loss balance. These
results support a narrow but useful conclusion for climate remote sensing:
alignment and target conventions must be audited before RGB-to-thermal models
are used as thermal proxies.

## 1. Motivation

Urban heat studies benefit from surface-temperature maps, but thermal sensors
are less common than visible cameras on UAVs and repeated thermal collection
can be operationally expensive. RGB-to-thermal translation is therefore an
appealing proxy task: learn a model that predicts a thermal-like scalar map
from RGB imagery, then use abundant visible imagery to expand thermal coverage.

The proxy is only useful if the evaluation is honest. In paired UAV data, RGB
and thermal images can have different optics, field of view, capture timing,
and preprocessing. A model may appear to improve because the target is smooth,
because a normalization choice changes PSNR scale, or because the evaluation
ignores alignment sensitivity. We frame this workshop version around those
protocol issues and the modest method that survived our ablations.

## 2. Data And Protocol

We use three paired RGB/thermal sources. Ann Arbor is the primary source
dataset and contains 336 train, 41 validation, and 42 test examples in the
locked protocol. Its recovered scalar thermal targets contain enough structure
that synthetic misalignment produces measurable degradation. Kust4K is a public
UAV RGB-TIR dataset; we use official splits and exclude author-flagged broken
stems, leaving 1970 train, 283 validation, and 565 test examples. CART is used
through its labeled RGB/thermal paired subset, with 1822 train, 222 validation,
and 238 test examples.

Target representation is part of the experimental protocol. Ann Arbor uses a
recovered scalar target, while Kust4K and CART provide raw thermal grayscale
targets. We audit raw, robust percentile normalization, and histogram matching;
the locked cross-dataset protocol uses robust normalization. We report external
results conservatively because target edge statistics differ even after
normalization.

To test alignment sensitivity, we synthetically perturb the RGB input with
translation, rotation, and scale while keeping the thermal target fixed. The
amplified protocol uses maximum translation fraction 0.20, maximum rotation 20
degrees, and maximum scale change 0.25 at sigma 1.0. Most registration results
use train/eval sigma 0.3.

## 3. Method

The baseline is a ConvNeXt-tiny U-Net translator from RGB to thermal scalar
target. Our registration variant adds a small RGB-only affine head before the
translator. The affine head is initialized to identity and predicts an
input-space warp for the synthetically misaligned RGB input. The warped RGB is
then translated by the same ConvNeXt U-Net decoder.

Training uses thermal reconstruction losses and a direct RGB warp-recovery
loss:

```text
L = L1(y_hat, y)
  + lambda_ssim * (1 - SSIM(y_hat, y))
  + lambda_edge * L_edge(y_hat, y)
  + lambda_affine * ||theta - I||
  + lambda_warp * L1(x_warp, x_aligned)
```

The locked method uses `lambda_warp=0.5`. Earlier variants used an uncertainty
map to weight reconstruction. Ablations showed that this hurt performance, so
the final method is uncertainty-decoupled: the branch may still compute a
diagnostic uncertainty map, but uncertainty is not used to shape the primary
loss in the locked rows.

## 4. Results

Ann Arbor is alignment-sensitive. In diagnostic pix2pix runs, amplified sigma
0.3 caused a 2.61 dB three-seed mean drop, motivating the registration study.
Unsupervised affine and dense-flow variants did not reliably beat
no-registration. The useful signal came from direct synthetic warp-recovery
supervision.

The main controlled result is a ConvNeXt-family ablation. Under the locked
robust target protocol and amplified sigma 0.3, uncertainty-decoupled affine
registration improves over matched ConvNeXt no-registration by
`+0.571 +/- 0.157 dB` across three seeds. The uncertainty-weighted affine model
improves by only `+0.260 +/- 0.021 dB`, showing that uncertainty weighting was
a poor design choice in this setting. The final method is also
`+0.215 +/- 0.113 dB` above a Swin-T U-Net no-registration baseline, though
that margin is modest.

External results narrow the claim. Kust4K within-dataset registration gives
only `+0.096 +/- 0.067 dB`, consistent with no meaningful effect. CART gives a
larger mean gain, but that gain changes substantially with the weight on the
RGB warp-recovery loss. Ann Arbor-to-Kust4K transfer is positive in the
three-seed audit, but other transfer cells remain diagnostic rather than
headline results.

## 5. Climate Relevance And Limitations

For urban heat workflows, the key lesson is not that this model is ready to
replace thermal sensing. The lesson is that RGB-to-thermal proxy models need
alignment and target-representation audits before their outputs are used for
climate or adaptation analysis. A method that improves on one local UAV target
convention may not transfer cleanly to another thermal convention.

The main limitation is that alignment supervision is synthetic. The model is
trained to recover artificial perturbations, not measured camera geometry.
Second, the datasets do not share identical target semantics. Third, the source
dataset is small, and the visual predictions remain smooth around high-frequency
thermal structures such as roof edges, vehicles, and hard boundaries.

The workshop contribution is therefore empirical and methodological: a compact
evaluation showing how alignment, target normalization, and synthetic warp
supervision affect aerial RGB-to-thermal translation for urban heat-motivated
remote sensing.

## Planned Figures And Tables

- One compact method/qualitative figure, likely combining the method diagram
  with the Ann Arbor hero scene.
- One compressed result table with ConvNeXt no-reg, ConvNeXt affine
  uncertainty-decoupled, Swin-T no-reg, and the Kust4K/CART caveats.
- Appendix-only qualitative examples if the 2026 CFP permits supplementary
  material.
