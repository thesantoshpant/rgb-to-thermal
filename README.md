# RGB → Thermal

Predicting urban thermal (heat) maps from ordinary RGB drone imagery. A rebuild of our
UMich "AI for Heat Resilience" hackathon project, focused on getting the most accurate
RGB→thermal translation possible (not just contest metrics).

The headline: two fixes to how the **data** was set up mattered more than any model choice,
and together with a small set of models they take accuracy from ~13–14 dB to **~19 dB PSNR on a
held-out test set the models never saw**.

📄 **Non-technical write-up with full results:** [`PROJECT_REPORT.md`](PROJECT_REPORT.md) (and `PROJECT_REPORT.pdf`).

![Best model vs ground truth](figures/best_vs_gt.png)
*RGB input · real thermal · best model. Top rows = train, bottom rows = unseen test.*

## Two findings that did most of the work
1. **Recover the real heat field, not the color channel.** The thermal images are a single
   color palette, not raw temperature. The original pipeline trained on the red channel, which
   isn't monotonic in temperature. Inverting the palette to a 1‑D scalar field (reconstruction
   residual ~5/255 vs ~42 for inferno) gives a clean, correct target. See `data_prep.py`.
2. **Register the RGB and thermal.** They come from different cameras with different fields of
   view, so they aren't pixel‑aligned. A fixed central ~0.65×width crop of the RGB lines them up
   (edge‑correlation 0.01 → ~0.18). Training on aligned pairs removes a large, silent error.
   Ablation: identical model/target, registered vs not → **18.4 vs 16.9 dB**.

## Approaches (all predict the scalar heat field, 512×640)
- **A1 — regression**: ImageNet‑pretrained ConvNeXt encoder + U‑Net decoder (`train_a1.py`).
- **A2 — conditional GAN**: same generator + PatchGAN, adversarial + L1 + LPIPS (`train_a2.py`).
- **A4 — physics‑structured**: predicts material masks + illumination, composes temperature from
  learned per‑material signatures (`train_a4.py`). Interpretable.
- **Ensemble + test‑time augmentation**: average the top models over flips (`eval_v2.py`).

## Results — official unseen test set (202 images)
| model | PSNR ↑ | SSIM ↑ | color‑LPIPS ↓ |
|---|---|---|---|
| **ensemble (A1+A2+A4) + TTA** | **19.28** | **0.712** | 0.36 |
| A2 (GAN) + TTA — best single | 18.91 | 0.694 | **0.32** (sharpest) |
| A1 (regression) + TTA | 18.73 | 0.703 | 0.37 |
| A4 (physics) + TTA | 18.42 | 0.696 | 0.39 |
| mean‑field baseline | 10.73 | 0.527 | 0.59 |

Validated on data never used for training or tuning; train ≈ val ≈ test, so it generalizes
rather than memorizes. What did **not** help: depth/satellite priors, bigger encoders, and
(at full resolution) sun‑geometry inputs. Full details in `REPORT.md` and `REPORT_v2.md`.

## Data (bring your own)
The drone imagery, satellite embeddings, and weather metadata are proprietary and **not** included.
The scripts expect this layout under the project root on the training machine:
```
data/Train_2/RGB/   data/Train_2/Thermal/   data/Test_2/RGB/
alphaearth-emb/     drone_and_weather_metadata.json     code/train_test_split.json
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision timm numpy pillow rasterio matplotlib pandas opencv-python-headless lpips scikit-image transformers
python data_prep.py        # palette inversion + registration + priors + split
python train_a1.py --name a1_rgb --epochs 80      # (also train_a2.py, train_a4.py)
python evaluate.py         # leaderboard + galleries
```

## Repo layout
- `data_prep.py`, `compute_solar.py` — data pipeline (palette inversion, registration, priors)
- `r2t_common.py` — dataset, losses, metrics, palette utils
- `train_a1.py` / `train_a2.py` / `train_a4.py` — the three model families
- `evaluate.py`, `eval_official.py`, `eval_v2.py` — evaluation, leaderboard, TTA + ensemble
- `make_galleries.py`, `best_gallery.py`, `sharp_compare.py` — qualitative figures
- `REPORT.md`, `REPORT_v2.md`, `APPROACH_v2.md`, `HOW_IT_WORKS_SIMPLE.md` — write‑ups
- `figures/`, `results/` — qualitative galleries and leaderboards

## Credit
Built on the UMich Center for Global Health Equity "AI for Heat Resilience" hackathon. Drone
imagery courtesy of SmithGroup. Baseline generator: ThermalGen (arplaboratory).
