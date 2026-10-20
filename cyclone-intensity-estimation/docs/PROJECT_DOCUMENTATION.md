# Project documentation

**Status of evidence.** Statements about the eight papers come from the team's literature survey slides. Statements about
this implementation describe code. **No experimental result exists until training/evaluation has been run on real data;
the Results section below must be completed only from `artifacts/*_metrics.json`.**

## Abstract
Tropical cyclone intensity is conventionally estimated with the subjective Dvorak technique. This project builds a
leakage-safe deep-learning pipeline that estimates maximum sustained wind speed (knots) from single-channel infrared
satellite images, compares a from-scratch CNN, an ImageNet-pretrained ResNet-18 and simple baselines, reports error by
intensity category, and offers a Monte-Carlo-dropout uncertainty indicator, with a Streamlit interface.

## Problem statement
Given one infrared image of a tropical cyclone, estimate its maximum sustained wind speed, and measure honestly how
error changes with intensity and across storms/basins.

## Objectives
1. Validated data pipeline with storm-wise (and optional hold-out-basin) splits.
2. CNN vs pretrained ResNet-18 vs mean/ridge baselines.
3. MAE/RMSE in knots, per-category error, confusion matrix from binned predictions.
4. Rough uncertainty estimate; interactive demo.

## Literature survey summary and research gap
See `reports/literature_survey_summary.md`. Gaps from the survey: little use of storm history; single basin/satellite/
channel; reliance on best-track inputs; weak at extreme intensities; no uncertainty; small, imbalanced datasets.
**Addressed (as evaluation goals, not solved):** leakage-safe splits, per-category error, optional cross-basin test,
transfer-learning comparison, MC-dropout uncertainty, optional rare-class oversampling.
**Not addressed:** temporal modelling, size estimation, multichannel fusion, best-track independence beyond using only
the image.

## Methodology
Image -> resize (224) -> fixed 0-1 scaling -> [train only: random rotation, brightness jitter; no flips because flipping
reverses cyclone rotation sense] -> CNN/ResNet-18 -> standardised wind -> knots -> category (IMD or Saffir-Simpson).
Loss: SmoothL1 on standardised wind; AdamW; ReduceLROnPlateau; early stopping on validation loss; seeds fixed.

### Architecture diagram
```
satellite IR image --> preprocessing --> CNN / ResNet-18 --> wind (kt) --> category
                                              |--> MC-dropout passes --> uncertainty
```
### Data flow
```
TCIR.h5 --prepare_tcir--> PNG frames + manifest.csv --data_loader--> validated manifest --> train/val/test (by storm)
train --> checkpoint --> evaluate --> metrics JSON + figures --> Streamlit app
```
### Pseudocode
```
split storms into train/val/test (no storm in two sets)
for epoch: train on train set; compute validation loss; keep best checkpoint; stop if no improvement
load best checkpoint; predict test set; compute MAE, RMSE, bias, per-category error, baselines
```

## Evaluation methodology
Held-out test storms only. MAE, RMSE, bias in knots; error by true category; category accuracy/precision/recall/F1 and
confusion matrix computed by binning the regression output; baselines on the same test set. Do not compare numbers
directly with the papers: datasets, years, basins and units differ.

## Results
*(Fill in from `artifacts/*_metrics.json` after running evaluation. Leave empty if not run.)*

## Limitations
Dataset-dependent; noisy best-track labels; rare intense storms; one IR channel; no history; uploaded web images are not
comparable to training imagery; uncertainty is uncalibrated; PNG conversion quantises IR values to 8 bits.

## Future scope
ConvLSTM on frame sequences; multichannel input (IR+WV); size estimation with R34/RMW labels; INSAT-3D data; calibrated
uncertainty; per-basin fine-tuning.

## Conclusion
To be written from verified results only.

## Presentation script (about 3 minutes)
"Cyclone intensity is normally judged by experts from satellite pictures. We trained neural networks to do this
automatically. We surveyed eight papers and found common gaps: unfair data splits, weak accuracy on strong storms, and no
uncertainty. We built a pipeline that keeps whole storms in one split, compares a small CNN, a pretrained ResNet-18 and
simple baselines, reports error per intensity category, and gives a rough uncertainty. Here is the app: it only shows
predictions from a real trained model and warns otherwise. Our measured results are: [read from the metrics files]."

## Likely questions
- *Why storm-wise splits?* Frames of one storm are nearly identical; random splits leak and inflate accuracy.
- *Why ResNet-18 pretrained?* Small data; survey papers 1-2 point to transfer learning; we test whether it helps.
- *Why no size estimation?* Needs R34/RMW labels; not in our data scope.
- *Why compare with a mean predictor?* It shows whether the CNN learns anything beyond the average wind.
- *Is this better than the papers?* We do not claim that; settings differ.
- *Why no horizontal flip?* It reverses cyclone rotation direction (hemisphere-specific physics).
- *What is MC dropout?* Several stochastic passes; the spread is a rough uncertainty indicator, not calibrated.
- *What are the weaknesses?* See Limitations; intense storms are rare and usually under-estimated.
