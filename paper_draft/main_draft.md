# Synthetic Warp Supervision for Robust Aerial RGB-to-Thermal Translation

## Abstract

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

## 1. Introduction

Aerial thermal imagery is useful for urban heat analysis, infrastructure
inspection, and remote sensing workflows where surface temperature patterns
matter but direct thermal capture may be scarce or expensive. RGB imagery is
more abundant, easier to collect, and often higher resolution, motivating
paired RGB-to-thermal translation models that predict a thermal-like scalar map
from visible imagery. In UAV settings, however, paired RGB and thermal imagery
is rarely a perfectly controlled image-to-image problem. Slight camera
misalignment, different optics, target normalization choices, and dataset
preprocessing conventions can dominate the apparent behavior of a translation
model.

This paper studies whether learned registration can improve aerial RGB-to-
thermal translation under controlled synthetic misalignment. The motivating
hypothesis is simple: if RGB detail is useful for predicting thermal structure,
then perturbing the RGB input while keeping the thermal target fixed should hurt
translation, and a model with an explicit alignment mechanism should recover
some of the lost performance. Early experiments showed that this hypothesis is
easy to overstate. On external datasets such as Kust4K and CART, raw grayscale
thermal targets are smoother and less sensitive to small spatial perturbations
than the recovered scalar Ann Arbor targets. Single-seed experiments also
exaggerate small effects. We therefore treat registration as a controlled
source-dataset robustness intervention, not as a universal cross-dataset
solution.

Our final method adds a lightweight RGB-only affine head in front of a
ConvNeXt-tiny U-Net translator. During training, we synthetically misalign the
RGB input while keeping the thermal target and the original aligned RGB
available. The registration head predicts an input-space affine warp, and an
auxiliary RGB warp-recovery loss directly supervises the predicted correction.
The translated output is then produced from the warped RGB. The final design is
uncertainty-decoupled: the model still produces an uncertainty map for
diagnostics, but uncertainty is not used to weight reconstruction because that
choice consistently hurt performance in ablation.

The main result is modest but repeatable. On Ann Arbor under amplified
synthetic misalignment, uncertainty-decoupled affine registration improves over
a matched no-registration ConvNeXt baseline by `+0.571 +/- 0.157 dB` across
three seeds. The same method is also `+0.215 +/- 0.113 dB` above a Swin-T U-Net
baseline in the paired three-seed table. These gains are not large enough to
claim that registration solves aerial RGB-to-thermal translation, but they are
large enough to show that synthetic warp supervision can improve robustness
when the source dataset is alignment-sensitive.

Our contributions are:

1. A reproducible multi-dataset harness for aerial RGB-to-thermal translation
   using Ann Arbor, Kust4K, and CART.
2. A synthetic misalignment diagnostic showing that Ann Arbor is alignment
   sensitive while Kust4K/CART effects are weaker and target-representation
   confounded.
3. A lightweight input-space affine registration module trained with direct RGB
   warp-recovery supervision.
4. A controlled ablation showing that uncertainty-weighted reconstruction hurts
   in this protocol, while uncertainty-decoupled affine registration gives a
   repeatable gain.
5. Qualitative figures showing both useful behavior and failure modes,
   including examples where the method underperforms the no-registration
   baseline.

## 2. Related Work

RGB-to-thermal and RGB-TIR translation methods are commonly framed as paired
image-to-image translation, using regression losses, adversarial losses, or
hybrid perceptual objectives. Paired regression models such as U-Net-style
architectures are strong when aligned supervision is available, while unpaired
translation methods such as CycleGAN are attractive when paired data is scarce
but often underperform paired objectives on small supervised datasets. Our
experiments confirm that gap in this setting: CycleGAN and pix2pix-style
baselines are substantially below pretrained encoder-decoder regressors.

Multimodal aerial datasets introduce additional complications. UAV RGB and
thermal imagery can differ in field of view, lens distortion, capture timing,
and target scale. Public datasets such as Kust4K and CART provide useful
external context, but their targets and preprocessing differ from Ann Arbor.
Kust4K provides official splits and author-supplied broken-sample lists; CART's
labeled subset is smaller than the full imagery discussed in its broader
dataset material. These details matter because a numerical gain measured on one
target convention is not automatically comparable to a gain measured on another.

Cross-modal registration has a long history in remote sensing and multimodal
vision. Spatial transformer-style modules and affine or flow-based warpers can
be trained end-to-end, but the gradient signal from a downstream reconstruction
loss may be weak when the target is low-frequency or only loosely correlated
with visible edges. Our ablations reflect this: unsupervised affine and dense
flow variants did not reliably beat the no-registration baseline. The effective
variant required direct synthetic warp-recovery supervision.

Uncertainty maps are often used to down-weight ambiguous regions in image
translation and multimodal prediction. In our setting, however, uncertainty-
weighted reconstruction reduced performance. We therefore retain uncertainty
maps only as diagnostic outputs in the primary method. This is an empirical
design choice, not a claim that uncertainty is unhelpful in general.

## 3. Datasets and Target Representation

We evaluate on three paired RGB/thermal sources. Table 1 summarizes the dataset
counts and target conventions used in the draft.

Ann Arbor is the primary source dataset. It provides registered RGB inputs and
recovered scalar thermal targets. The split used throughout the locked Week 7
protocol has 336 training, 41 validation, and 42 test examples. Because the
target is a recovered scalar thermal map with visible high-frequency structure,
Ann Arbor is the dataset where synthetic misalignment has the clearest
measurable effect.

Kust4K is a public UAV RGB-TIR dataset. We use the official train/validation/
test split files and exclude sample stems listed in the dataset's broken RGB
and TIR lists. After those exclusions, the usable split contains 1970 training,
283 validation, and 565 test examples. Kust4K is important external context, but
it does not support a positive within-dataset registration claim in our results:
the three-seed gain is only `+0.096 +/- 0.067 dB`, consistent with no effect.

CART is used through its labeled RGB/thermal paired subset discussed above,
with 1822 training, 222 validation, and 238 test examples under our split. The
broader CART material includes additional imagery, but our supervised
experiments use only this labeled subset. CART shows a larger within-dataset
registration gain than Kust4K, but the gain is sensitive to the weight on the
RGB warp-recovery loss. We therefore treat CART as supportive but not
definitive evidence.

Target normalization is a central part of the protocol. Ann Arbor's recovered
scalar targets and the external datasets' raw grayscale thermal targets have
different edge statistics. We evaluated raw, robust percentile normalization,
and histogram matching. The locked Week 5-8 protocol uses robust normalization
for cross-dataset experiments, while some legacy qualitative rows use raw
targets because they are generated from earlier within-dataset checkpoints. The
paper must distinguish these settings explicitly.

## 4. Method

Figure 2 gives the method overview. Let `x` be an RGB input and `y` be the
aligned thermal scalar target. During training, we sample a synthetic affine
perturbation and apply it to the RGB input, producing `x_mis`. The thermal
target remains fixed. A small RGB-only registration head predicts affine
parameters `theta`, initialized at identity. The predicted warp produces
`x_warp = W(x_mis, theta)`, which is passed to a ConvNeXt-tiny U-Net translator
to predict `y_hat`.

The training objective combines thermal reconstruction terms and a direct RGB
warp-recovery term:

```text
L = L1(y_hat, y)
  + lambda_ssim * (1 - SSIM(y_hat, y))
  + lambda_edge * L_edge(y_hat, y)
  + lambda_affine * ||theta - I||
  + lambda_warp * L1(x_warp, x_aligned)
  + lambda_unc * mean(u)
  + lambda_unc_tv * TV(u)
```

The final method uses `lambda_warp=0.5` and decouples uncertainty from
reconstruction. Earlier variants used an uncertainty map to weight the L1 loss
and regularized the uncertainty mean and total variation. In the locked primary
rows, both uncertainty regularizer weights are set to zero (`lambda_unc=0`,
`lambda_unc_tv=0`); the branch still produces a diagnostic map but it does not
affect the loss. The uncertainty-weighted design was repeatable but weaker: it
improved over no-registration by `+0.260 +/- 0.021 dB`, while the
uncertainty-decoupled variant improves by `+0.571 +/- 0.157 dB`.

The synthetic perturbation uses translation, rotation, and scale. In the locked
amplified protocol, sigma `1.0` corresponds to a maximum translation fraction of
0.20, maximum rotation of 20 degrees, and maximum scale change of 0.25. Most
reported registration experiments use matched train/eval sigma `0.3`.

## 5. Experiments

### 5.1 Synthetic Misalignment Diagnostics

The initial Week 2 sweep suggested only small performance drops under
misalignment on Kust4K and CART. We did not accept that as a no-go signal
because the perturbation was small, the targets were low-frequency, and the
experiment used a single seed. Week 2.5 added Ann Arbor as a control, validation
time misalignment, shuffled-RGB controls, stronger perturbations, multiple
seeds, and L1-only baselines. These diagnostics showed that Ann Arbor is
alignment sensitive: amplified sigma `0.3` produced a three-seed mean drop of
2.61 dB in the diagnostic pix2pix setting, while external datasets remained
weaker and more target-confounded.

### 5.2 Registration Variants

Unsupervised learned registration did not solve the problem. Shared-feature
affine, input-space affine, and dense-flow variants without direct synthetic
warp supervision underperformed or matched the no-registration baseline. Adding
the RGB warp-recovery loss produced the first repeatable improvement, but the
effect was initially marginal. Week 7 ablations then identified the stronger
uncertainty-decoupled variant.

### 5.3 Baselines

Table 2 compares the main baseline families. CycleGAN, pix2pix, and a small
U-Net L1 baseline underperform pretrained encoder-decoder regressors. A Swin-T
U-Net is a stronger single-seed baseline than ConvNeXt no-registration, but the
three-seed audit weakens the claim that Swin-T dominates the registration
method. The final ConvNeXt affine uncertainty-decoupled model is the top row in
the controlled three-seed table, though its advantage over Swin-T is modest.

### 5.4 Main Ablation

Table 3 reports the registration ablation across both ConvNeXt and Swin-T
families. Within the ConvNeXt family, the uncertainty-weighted affine model
improves over no-registration by `+0.260 +/- 0.021 dB`. Decoupling uncertainty
from reconstruction raises the gain to `+0.571 +/- 0.157 dB`. The Swin-T rows
include a backbone/decoder-family change when compared against ConvNeXt
no-registration, so the same-family delta is the relevant registration test:
Swin-T affine does not reliably stack with registration (`-0.064 +/- 0.214 dB`
versus Swin-T no-registration). The paper should therefore frame the method as
a ConvNeXt registration ablation, not as a universal module that improves every
backbone.

### 5.5 External and Transfer Results

Table 4 summarizes the external evidence. Kust4K within-dataset registration
does not show a meaningful gain across seeds. CART has a larger mean gain, but
the effect depends on the warp-recovery loss weight, so it is not clean evidence
that CART benefits more from registration. Ann Arbor-to-Kust4K transfer is the
only transfer cell that survived a three-seed audit. Other transfer cells are
single-seed diagnostics and should not be used as headline claims.

### 5.6 Qualitative Results

Figure 1 shows a hero Ann Arbor scene with road, vegetation, building edges,
target thermal, our prediction, uncertainty, and error. The method improves that
sample by `+0.93 dB`, but this is an illustrative upper-tail example. The main
quantitative claim remains the three-seed mean `+0.571 +/- 0.157 dB`.

Figure 3 shows the same Ann Arbor scene across sigma `0.0`, `0.2`, and `0.5`.
The model helps at sigma `0.2` but loses slightly at sigma `0.5` on this scene,
so the figure is useful for explaining behavior but should not be presented as
a monotonic severity result.

Figure 4 shows method-specific failures: samples where our method has the most
negative paired PSNR delta relative to the no-registration baseline. The common
failure mode is smoothing of high-frequency thermal detail around roofs,
vehicles, and hard building edges. The uncertainty map does not fully localize
these errors.

Appendix Figure B includes a cross-dataset qualitative gallery. It is context
only. The Kust4K row includes an in-figure warning because the Kust4K three-seed
within-dataset gain is not significant.

## 6. Discussion

The central lesson is not that registration is a solved bottleneck. The stronger
lesson is that alignment sensitivity is dataset- and target-dependent, and that
registration heads need direct supervision when the downstream thermal signal is
too smooth or indirect to train the warp reliably. Synthetic warp-recovery
supervision gives the model a clear alignment target, and that improves Ann
Arbor robustness under the controlled perturbation.

The uncertainty result is also important. Our original uncertainty-weighted
design sounded plausible: down-weight ambiguous regions and let the model focus
on reliable pixels. In practice, that weighting reduced PSNR. The best variant
keeps the uncertainty branch as a diagnostic output but removes it from the
reconstruction weighting path. In the locked primary rows, the uncertainty
regularizer weights are also zero, so the branch is computed for diagnostics but
does not shape the objective. This finding should be presented as an
ablation-driven design choice rather than a theoretical claim about uncertainty.
We also bounded the loss-formulation confound directly: changing the ConvNeXt
no-registration baseline to the Swin-style combined loss improves the seed-42
PSNR by `+0.201 dB` in the explicit zero-uncertainty-terms control. The older
combined control is `0.037 dB` higher, a small run/control difference that does
not change the conclusion. This loss-recipe effect is smaller than the primary
three-seed affine gain (`+0.571 +/- 0.157 dB`).

The external results constrain the paper's scope. Kust4K does not support a
positive within-dataset registration claim across seeds. CART has stronger
numbers, but the loss-balance sensitivity means we cannot claim a clean
dataset-level effect. Cross-dataset transfer is limited. These results make the
paper more honest: it is a source-dataset robustness study with external
diagnostics, not a broad generalization benchmark.

## 7. Limitations

The main limitation is that the alignment supervision is synthetic. The model is
trained to recover known artificial perturbations, not measured real-world
camera geometry. This makes the experiment controlled and reproducible, but it
also narrows the claim.

The second limitation is target representation. Ann Arbor, Kust4K, and CART do
not share identical target semantics or edge statistics. Robust normalization
helps but does not make all PSNR values directly comparable.

The third limitation is scale. The source dataset is small, and even the
three-seed results are not a substitute for a larger multi-flight evaluation.
Qualitative figures are single-seed and should be used for explanation, not
statistical claims.

Finally, the visual outputs are still smooth. The model captures broad thermal
regions but misses fine thermal structures around roofs, cars, and sharp
boundaries. That failure mode is visible in the error maps and remains open.

## 8. Conclusion

Synthetic warp supervision provides a small but repeatable improvement for
aerial RGB-to-thermal translation on an alignment-sensitive source dataset. The
best variant uses an RGB-only input-space affine head with direct RGB
warp-recovery supervision and uncertainty-decoupled reconstruction. The gain is
real, but the claim should remain narrow: registration helps under the Ann Arbor
synthetic misalignment protocol, while external datasets expose target and
loss-balance limitations. This makes the method a defensible robustness module
and an empirical study of alignment sensitivity, not a universal RGB-to-thermal
solution.

## Appendix A. Additional Qualitative Examples

Appendix Figure A will include the PSNR-quantile Ann Arbor candidate grid from
`figures/week8/ann_arbor_candidate_grid_seed42.png`.

## Appendix B. Cross-Dataset Qualitative Context

Appendix Figure B will include `figures/week8/cross_dataset_gallery_seed42.png`.
This appendix panel is qualitative context only. The Kust4K row is explicitly
marked as non-significant across seeds.
