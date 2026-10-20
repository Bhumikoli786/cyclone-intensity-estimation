# Literature Survey Summary (from the team's survey slides)

Source: `DL_Literature_Survey (5,6,7,8).pptx`, guide Dr. Joanne Gomes.
All figures below are **copied from the survey slides as reported by the original papers**.
They have not been re-checked against the papers and were NOT produced by this project's code.
Metrics use different units and test sets, so they are not directly comparable.

| # | Paper | Method | Data / scope | Reported result |
|---|---|---|---|---|
| 1 | Dharpure et al., 2023 | CNN, RNN, CNN-RNN vs SVM/RF | INSAT-3D IR, N. Indian Ocean, 1,184 images | MAE 4.89 kt (CNN-RNN); slides note 7.87 kt elsewhere in the paper |
| 2 | Jung et al., 2024 | CNN transfer learning COMS -> GK2A | W. North Pacific, 2011-2021 | Test RMSE 11.17 kt vs 13.99 kt without transfer; 6-h model 3.79 kt |
| 3 | Zhang et al., 2021 (TCICENet) | Classify intensity grade, then regress per grade | NW Pacific, 1,001 TCs, single IR band | RMSE 8.60 kt |
| 4 | Baek et al., 2022 (TC-SEM) | Multi-task CNN for size (RMW, R34) | W. North Pacific, COMS 4 IR bands | RMW MAE 2.05 nmi; R34 MAE 9.77 nmi |
| 5 | Ritchie et al., 2012 (DAV) | Classical deviation-angle variance + sigmoid (not deep learning) | 95 Atlantic TCs, GOES IR | RMSE 12.9 kt |
| 6 | Zhuo & Tan, 2021 (DeepTCNet) | Physics-augmented VGG-13, multitask intensity + size | N. Atlantic, 8-km IR | RMSE 8.7 kt |
| 7 | Manju et al., 2025 | Clustering + CNN detection + ConvLSTM on 4-frame sequences | 15 N. Indian Ocean cyclones, 1,200 LWIR images | 81.1% grading accuracy; wind RMSE 7.79 kt |
| 8 | Zhang et al., 2025 | Wavelet transform + IR/WV/PMW fusion, then regression | TCIR global, 10,271 samples | MAE 3.76, RMSE 4.62 **m/s** |

## Research gaps (slide 10)
- Little use of storm history / time evolution
- Single basin, satellite or IR channel; no cross-basin testing
- Dependence on best-track inputs (centre, intensity, fullness) unavailable in real time
- Weak performance at extreme intensities (Cat 4-5)
- No uncertainty estimates
- Small datasets, rare severe classes; random splits of correlated frames (papers 1, 4)

## What this project addresses (honest scope)
Addressed as evaluation goals, not claimed as solved:
1. Leakage-safe storm-wise splitting (gap noted for papers 1 and 4).
2. Error reported per intensity band, to expose weakness at extreme intensities.
3. Cross-basin test (train outside the Indian Ocean, test on it) - if TCIR is used.
4. Pretrained (transfer-learning) model vs from-scratch CNN vs trivial baselines.
5. Uncertainty via Monte-Carlo dropout - stretch goal.
Not addressed unless data allow: temporal modelling, size estimation, multichannel fusion.
