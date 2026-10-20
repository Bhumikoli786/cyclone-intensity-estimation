"""Evaluate a checkpoint on the held-out test split and write metrics + figures to artifacts/.

    python -m src.evaluate --checkpoint artifacts/best_resnet18.pt --config config/config.yaml --mc-samples 20
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics as M
from src.baselines import MeanBaseline, RidgeBaseline
from src.data_loader import load_config
from src.plots import plot_category_mae, plot_confusion, plot_scatter


def predict_dataframe(lm, df, batch_size=64, mc_samples=0, num_workers=0):
    import torch
    from torch.utils.data import DataLoader
    from src.data_loader import CycloneDataset
    from src.model import mc_dropout_predict
    c = lm.config
    ds = CycloneDataset(df, c["image_size"], c["normalization"], c["max_pixel_value"], augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds, stds = [], []
    for x, _ in loader:
        x = x.to(lm.device)
        if mc_samples >= 2:
            mean, sd = mc_dropout_predict(lm.model, x, mc_samples)
            stds.append((sd * lm.target_std).cpu().numpy())
        else:
            with torch.no_grad():
                lm.model.eval()
                mean = lm.model(x)
        preds.append((mean * lm.target_std + lm.target_mean).cpu().numpy())
    y_pred = np.concatenate(preds)
    return y_pred, (np.concatenate(stds) if stds else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on a held-out split.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mc-samples", type=int, default=0, help=">=2 enables Monte-Carlo dropout uncertainty")
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()

    from src.predict import load_checkpoint
    cfg = load_config(a.config)
    lm = load_checkpoint(a.checkpoint)
    scheme = lm.meta.get("category_scheme", cfg["category"]["scheme"])
    pdir = Path(cfg["data"]["processed_dir"])
    df = pd.read_csv(pdir / f"{a.split}.csv")
    if df.empty:
        raise SystemExit(f"ERROR: {a.split}.csv is empty")
    y_true = df["wind_speed_kt"].to_numpy(dtype=float)
    y_pred, y_std = predict_dataframe(lm, df, a.batch_size, a.mc_samples)

    name = Path(a.checkpoint).stem
    out = Path(cfg.get("artifacts_dir", "artifacts"))
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "checkpoint": str(a.checkpoint), "model_name": lm.meta["model_name"], "split": a.split,
        "demo_run": bool(lm.meta.get("demo_run", False)), "wind_unit": "knots", "scheme": scheme,
        "split_strategy": cfg["split"]["strategy"], "n_storms": int(df["storm_id"].nunique()),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": M.regression_metrics(y_true, y_pred),
        "per_category": M.per_category_metrics(y_true, y_pred, scheme),
        "classification_from_regression": M.classification_metrics(y_true, y_pred, scheme),
    }
    if "basin" in df.columns:
        res["per_basin"] = M.group_metrics(y_true, y_pred, df["basin"])
    if y_std is not None:
        res["uncertainty_mc_dropout"] = M.uncertainty_summary(y_true, y_pred, y_std)
    if not a.no_baselines:
        train_df = pd.read_csv(pdir / "train.csv")
        base = {"mean_predictor": M.regression_metrics(y_true, MeanBaseline().fit(train_df["wind_speed_kt"]).predict(len(df)))}
        p = cfg["preprocessing"]
        ridge = RidgeBaseline(16, 10.0, p["normalization"], p["max_pixel_value"]).fit(train_df, seed=cfg.get("seed", 42))
        base["ridge_16x16_pixels"] = M.regression_metrics(y_true, ridge.predict(df))
        res["baselines"] = base

    pred_df = df[[c for c in ("image_path", "storm_id", "timestamp", "basin") if c in df.columns]].copy()
    pred_df["wind_true_kt"], pred_df["wind_pred_kt"] = y_true, y_pred
    if y_std is not None:
        pred_df["wind_std_kt"] = y_std
    pred_df.to_csv(out / f"{name}_{a.split}_predictions.csv", index=False)
    plot_scatter(y_true, y_pred, out / f"{name}_{a.split}_scatter.png")
    plot_category_mae(res["per_category"], out / f"{name}_{a.split}_category_mae.png")
    cm = res["classification_from_regression"]
    plot_confusion(cm["confusion_matrix"], cm["labels"], out / f"{name}_{a.split}_confusion.png")
    (out / f"{name}_{a.split}_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    o = res["overall"]
    print(f"{a.split} set ({o['n']} images): MAE {o['mae_kt']:.2f} kt | RMSE {o['rmse_kt']:.2f} kt | bias {o['bias_kt']:+.2f} kt")
    if "baselines" in res:
        for k, v in res["baselines"].items():
            print(f"  baseline {k}: MAE {v['mae_kt']:.2f} kt | RMSE {v['rmse_kt']:.2f} kt")
    if res["demo_run"]:
        print("NOTE: this checkpoint is a DEMO RUN (software test). Do not report these numbers as results.")
    print(f"Saved metrics, predictions and figures to {out}/")


if __name__ == "__main__":
    main()
