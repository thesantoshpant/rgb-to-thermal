# External Dataset Notes

Week 1 source-of-truth notes for public RGB/Thermal data.

## Ann Arbor / SmithGroup

- Status: proprietary local dataset, already cached on Knox under
  `/home/spant/UMich/umich-hackathon/rgb2thermal/data_cache`.
- License: do not redistribute.
- Current reproduction anchor: `eval_v2.py` on the official 202-image test set gives
  `19.28 dB` PSNR for the weighted TTA ensemble.

## Caltech CART

- Status: verified and downloaded on Knox.
- Paper: *Caltech Aerial RGB-Thermal Dataset in the Wild*, ECCV 2024 / arXiv:2403.08997.
- Code: `https://github.com/aerorobotics/caltech-aerial-rgbt-dataset`.
- Data: `https://data.caltech.edu/records/cks6g-ps927`.
- Useful file for this project: `labeled_rgbt_pairs.zip` from CaltechDATA.
- Knox path:
  `/home/spant/UMich/umich-hackathon/rgb2thermal_wacv/data_cache/external/caltech/`.
- Extracted loader count: 2282 paired samples.
- Note: 2282 is the full `labeled_rgbt_pairs.zip` supervised subset, not a
  partial extraction. Larger CART image counts include unlabeled imagery outside
  this paired labeled archive.
- Dataset license on CaltechDATA: Creative Commons Attribution Non Commercial 4.0
  International.
- Repository license file: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International. Treat derivatives/code reuse conservatively and cite the dataset paper.

## Kust4K

- Status: verified and downloaded on Knox.
- Paper: *Kust4K: An RGB-TIR Dataset from UAV Platform for Robust Urban
  Traffic Scenes Semantic Segmentation*, Scientific Data, 2025.
- Data: `https://figshare.com/articles/dataset/_b_Kust4K_b_b_b_b_A_Large-scale_Multimodal_UAV_Dataset_for_Robust_Urban_Traffic_Scenes_Semantic_Segmentation_b_/29476610`.
- DOI: `10.6084/m9.figshare.29476610.v3`.
- License: Creative Commons Attribution 4.0 International.
- Useful files from Figshare: `RGB.zip`, `TIR.zip`, `train.txt`, `val.txt`,
  `test.txt`, `broke_RGB.txt`, `broke_TIR.txt`.
- Verified file checksums:
  - `RGB.zip`: `d0bc9895100f339b1e13f40f2efe532f`
  - `TIR.zip`: `90562a17a4160600b2a12359d2c48391`
- Knox path:
  `/home/spant/UMich/umich-hackathon/rgb2thermal_wacv/data_cache/external/kust4k/`.
- Raw extracted count: 4024 paired RGB/TIR samples.
- Usable loader count after excluding `broke_RGB.txt` and `broke_TIR.txt` stems:
  2818 total; train 1970, val 283, test 565.
- Direct numerical comparisons to published Kust4K training results must state
  this exclusion policy, because the official split files list more samples
  before removing author-flagged broken RGB/TIR stems.
