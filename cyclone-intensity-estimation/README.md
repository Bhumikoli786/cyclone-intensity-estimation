# Tropical Cyclone Intensity Estimation from Satellite Imagery (Deep Learning)

College project. Estimates **maximum sustained wind speed (knots)** from a single-channel infrared satellite image
with a CNN and a pretrained ResNet-18, and serves it through a Streamlit app.

> **Honesty note.** This repository contains no trained model and no results until you run training on real data.
> Cyclone *size* estimation and ConvLSTM temporal modelling are **not implemented** (future work).
> Design is inspired by the team's literature survey; it is not a reproduction of any paper.

## Folder structure
| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `config/config.yaml` | all non-secret settings (paths, image size, split, training) |
| `src/preprocessing.py` | load, resize, normalise, augment images |
| `src/data_loader.py` | manifest validation, storm-wise / hold-out-basin / chronological splits, dataset class |
| `src/prepare_tcir.py` | converts the TCIR `.h5` file to PNG frames + `manifest.csv` |
| `src/model.py` | `SimpleCNN`, `ResNetRegressor`, Monte-Carlo-dropout uncertainty |
| `src/train.py` | training loop, early stopping, checkpoints, loss plot |
| `src/evaluate.py` | test-set metrics, per-category error, baselines, figures |
| `src/metrics.py`, `src/baselines.py`, `src/plots.py` | metrics, mean / ridge baselines, figures |
| `src/predict.py` | load a checkpoint, validate an image, predict |
| `src/make_smoke_data.py` | ARTIFICIAL images to test the software only |
| `tests/` | unit tests |
| `notebooks/colab_train.ipynb` | step-by-step Colab notebook |
| `docs/PROJECT_DOCUMENTATION.md` | abstract, methodology, limitations, viva Q&A |

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt
python -m pytest
```

## 2. Dataset
Recommended: **TCIR** (Chen, Chen & Lin) - HDF5 with `matrix` (N x 201 x 201 x 4: IR1, WV, VIS, PMW) and `info`
(table). Check the licence/terms on the dataset page. Only the IR1 channel is used.
```bash
python -m src.prepare_tcir --h5 /path/TCIR.h5 --inspect          # READ the output: columns, wind units
python -m src.prepare_tcir --h5 /path/TCIR.h5 --out data/raw --stride 3
python -m src.data_loader --config config/config.yaml            # validate + split by storm
```
If the wind column is in m/s add `--wind-scale 1.94384`. If column detection fails use `--storm-col --time-col --wind-col`.
Own data (e.g. INSAT-3D): create `data/raw/manifest.csv` with columns
`image_path, storm_id, timestamp, wind_speed_kt` (+ optional `basin`) and 8-bit grayscale images.

Cross-basin experiment (needs a `basin` column): set `split.strategy: holdout_basin` and `split.holdout_basin: <code>`
in `config/config.yaml` (codes are printed by `--inspect`).

## 3. Train and evaluate (GPU recommended - use `notebooks/colab_train.ipynb`)
```bash
python -m src.train --config config/config.yaml --model cnn
python -m src.train --config config/config.yaml --model resnet18        # add --balance to oversample rare categories
python -m src.evaluate --checkpoint artifacts/best_resnet18.pt --mc-samples 20
```
Outputs in `artifacts/`: `best_*.pt`, `*_history.json`, loss curve, `*_metrics.json`, predictions CSV, figures.

## 4. Software smoke test (no real data - results are meaningless)
```bash
python -m src.make_smoke_data
python -m src.data_loader --config config/config_smoke.yaml
python -m src.train --config config/config_smoke.yaml --demo-run
python -m src.evaluate --checkpoint artifacts/demo_best_cnn.pt --config config/config_smoke.yaml
```
`demo_*` checkpoints are flagged in the app with a red warning.

## 5. Run the app
```bash
streamlit run app.py
```

## 6. GitHub
Create an empty repo named `cyclone-intensity-estimation` on github.com (no README), then:
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/cyclone-intensity-estimation.git
git push -u origin main
```
Never commit datasets or secrets (`.gitignore` already excludes them). To publish a trained checkpoint
(ResNet-18 is about 45 MB, under GitHub's 100 MB file limit): `git add -f artifacts/best_resnet18.pt`
plus its `*_metrics.json` and figures. Update later with `git add . && git commit -m "msg" && git push`.

## 7. Deploy on Streamlit Community Cloud
Sign in at share.streamlit.io with GitHub > New app > pick the repo, branch `main`, main file `app.py` > Deploy.
Dependencies come from `requirements.txt`. PyTorch is large: if the build runs out of memory or time, try adding
`--extra-index-url https://download.pytorch.org/whl/cpu` as the first line of `requirements.txt`. Without a committed
checkpoint the live app still runs but shows "prediction disabled".
