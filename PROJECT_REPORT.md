# RGB-to-Thermal Translation for Urban Heat Mapping
**Progress report — Team "Game of Drones" (Santosh Pant, Adithya, Galgallo, Nima)**
*University of Michigan "AI for Heat Resilience" Hackathon follow-up · May 2026*

## 1. Summary
We are predicting a building- and street-level heat map (a thermal image) from an ordinary color
(RGB) drone photo. Building on our hackathon entry, we found that two issues in how the data was
prepared were limiting accuracy more than the choice of model. After correcting them and training a
small set of models, the accuracy of the predicted heat maps improved substantially, and the
improvement holds on a separate set of images the models never saw during training. On the official
held-out test set, our best system reaches a PSNR of about **19 dB**, compared with roughly
**13–14 dB** before.

## 2. Why this matters
Cities are getting hotter, and the hottest surfaces (dark roofs, asphalt, sun-exposed walls) drive
local heat risk. Thermal cameras can reveal these surfaces but are expensive and not widely deployed.
If a model can infer the thermal pattern from inexpensive RGB imagery, heat-risk mapping becomes far
cheaper and more scalable. Making that translation as accurate as possible is the goal of this work.

## 3. The data
The dataset is paired drone imagery over Ann Arbor: a high-resolution color image and a corresponding
thermal image for each location, plus satellite-derived context and weather at the time of capture.
The imagery is proprietary (SmithGroup) and is not redistributed. We hold out a portion of the data
that the models never see during training, and report performance on it.

## 4. The two improvements that mattered most
### 4.1 Predicting the real heat values, not the image's color channel
The thermal images are stored as color-coded pictures (a "false-color" palette), not as raw
temperatures. The earlier approach trained the model to reproduce one color channel of these
pictures, but that channel does not increase steadily with temperature, so the model was being
optimized toward a distorted target. We reconstructed the underlying heat value for each pixel by
inverting the color palette (recovering the original scale to within about 2% error). Training on
this corrected target gives the model the right thing to aim for.

### 4.2 Aligning the color and thermal images
The color and thermal images come from two different cameras with different fields of view, so they
do not line up pixel-for-pixel. The earlier approach resized both to the same size, which left them
misaligned and effectively asked the model to match mismatched pairs. We found that the thermal
camera sees a fixed central region (about 65% of the width) of the color image; aligning to that
region restores the correspondence. **Effect of alignment alone** (same model and target, with vs
without alignment): **PSNR 18.4 vs 16.9 dB.**

## 5. The models
We trained three different kinds of models so we could compare them and combine their strengths:

- **A direct prediction network** — a standard image-to-image network with a pretrained vision
  backbone; accurate and simple.
- **A generative adversarial model (GAN)** — two networks, one drawing the heat map and one
  critiquing it; produces the sharpest, most realistic-looking results.
- **A physically-structured model** — it first identifies surface types (roof, road, vegetation,
  shadow) and assigns each a learned temperature, which makes its reasoning easy to inspect.

Finally, we **combine** the three by averaging their predictions (an "ensemble"), and average each
over mirrored versions of the input ("test-time augmentation"), which gives the most accurate result.

## 6. Results
All numbers are on the **official test set of 202 images the models never saw during training**.
Three standard measures are used:

- **PSNR** (decibels): closeness to the true heat map, pixel by pixel — higher is better.
- **SSIM** (0–1): how well the patterns and structures match — higher is better.
- **LPIPS** (0–1): how similar the images look perceptually — lower is better.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| **Combined ensemble (best)** | **19.3** | **0.71** | 0.36 |
| GAN (best single model, sharpest) | 18.9 | 0.69 | **0.32** |
| Direct prediction network | 18.7 | 0.70 | 0.37 |
| Physically-structured model | 18.4 | 0.69 | 0.39 |
| Simple baseline (constant average heat map) | 10.7 | 0.53 | 0.59 |

For reference, the hackathon system scored about 13–14 dB (measured against the earlier
color-channel target). Our gains come primarily from the two data corrections in Section 4.

![Figure 1](figures/best_vs_gt.png)

**Figure 1.** Color photo, real thermal, and best-model prediction. Top rows: training scenes;
bottom rows: unseen test scenes.

![Figure 2](figures/test_official_withGT_gallery.png)

**Figure 2.** Unseen test scenes: real thermal compared with each model.

**Generalization (no memorization).** Performance on the unseen test set is essentially the same as
on internal validation (about 19 dB in both), indicating the models learn the underlying pattern
rather than memorizing the training images.

## 7. What did not help (honest negative results)
We tested several additions that did not improve accuracy: extra inputs such as estimated depth and
satellite-context features, larger network backbones, and sun-position information at full resolution.
With only a few hundred training images, the simpler setup generalized best.

## 8. Limitations

- All images were captured within a single hour on one afternoon, so the model has not seen varied
  weather, seasons, or times of day, and may not transfer to very different conditions.
- A small residual misalignment remains because the two cameras have slightly different perspectives
  (parallax), which limits pixel-level accuracy.
- The recovered heat values are on a consistent relative scale, not absolute degrees.

## 9. Toward a publication and next steps
The most promising direction is more and more-varied data: pretraining on public aerial
color–thermal datasets and fine-tuning here, and ideally collecting imagery across different times
and conditions. The contributions worth writing up are: (i) the importance of recovering the true
thermal field and aligning the two sensors; (ii) a fair comparison of model families for this task;
and (iii) the interpretable, physically-structured model. We would welcome your guidance on framing
and scope.

## 10. Code and reproducibility
All code, documentation, and result figures are available at:
**https://github.com/Santoshpant23/rgb-to-thermal**
(The proprietary imagery is not included; the code expects the data to be supplied separately.)
