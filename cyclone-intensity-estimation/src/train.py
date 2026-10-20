"""Train a wind-speed regression model.

    python -m src.train --config config/config.yaml --model resnet18
    python -m src.train --config config/config_smoke.yaml --demo-run     (software test only)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import CycloneDataset, intensity_category, load_config
from src.model import build_model, count_parameters
from src.plots import plot_loss_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(_worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def read_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    d = Path(cfg["data"]["processed_dir"])
    out = {}
    for name in ("train", "val", "test"):
        p = d / f"{name}.csv"
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found. Run: python -m src.data_loader --config <your config> first.")
        out[name] = pd.read_csv(p)
        if out[name].empty:
            raise ValueError(f"{p} is empty.")
    return out


def make_loader(df, cfg, augment, shuffle, batch_size, num_workers, sampler=None, seed=42):
    p, a = cfg["preprocessing"], cfg["augmentation"]
    ds = CycloneDataset(df, p["image_size"], p["normalization"], p["max_pixel_value"],
                        augment=augment and a.get("enabled", True), aug_cfg=a, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=_worker_init if num_workers > 0 else None)


def balanced_sampler(df: pd.DataFrame, scheme: str, seed: int) -> WeightedRandomSampler:
    """Give rare (usually intense) categories a higher chance of being drawn."""
    cats = pd.Series([intensity_category(float(w), scheme) for w in df["wind_speed_kt"]])
    w = 1.0 / cats.map(cats.value_counts()).to_numpy(dtype=float)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df), replacement=True,
                                 generator=g)


def run_epoch(model, loader, loss_fn, device, mean, std, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, abs_kt, n = 0.0, 0.0, 0
    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            pred = model(x)
            loss = loss_fn(pred, (y - mean) / std)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = x.size(0)
            total += loss.item() * b
            abs_kt += (pred.detach() * std + mean - y).abs().sum().item()
            n += b
    if n == 0:
        raise RuntimeError("Data loader returned no samples.")
    return total / n, abs_kt / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a cyclone wind-speed regression model.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model", choices=["cnn", "resnet18"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--balance", action="store_true", help="oversample rare intensity categories")
    ap.add_argument("--demo-run", action="store_true", help="mark checkpoint as a software test, not a result")
    a = ap.parse_args()

    cfg = load_config(a.config)
    t = cfg["training"]
    model_name = a.model or t["model"]
    epochs = a.epochs or t["epochs"]
    bs = a.batch_size or t["batch_size"]
    lr = a.lr or t["lr"]
    pretrained = t.get("pretrained", True) and not a.no_pretrained and model_name == "resnet18"
    seed = cfg.get("seed", 42)
    scheme = cfg["category"]["scheme"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {model_name} | pretrained: {pretrained} | demo_run: {a.demo_run}")

    splits = read_splits(cfg)
    for k, v in splits.items():
        print(f"  {k}: {len(v)} images, {v['storm_id'].nunique()} storms")
    mean = float(splits["train"]["wind_speed_kt"].mean())
    std = float(splits["train"]["wind_speed_kt"].std() or 1.0)
    sampler = balanced_sampler(splits["train"], scheme, seed) if a.balance else None
    nw = t["num_workers"]
    train_loader = make_loader(splits["train"], cfg, True, True, bs, nw, sampler, seed)
    val_loader = make_loader(splits["val"], cfg, False, False, bs, nw, None, seed)

    model = build_model(model_name, cfg["preprocessing"]["in_channels"], pretrained, t["dropout"]).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    out_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "demo_" if a.demo_run else ""
    ckpt_path = out_dir / f"{prefix}best_{model_name}.pt"
    history = {"train_loss": [], "val_loss": [], "train_mae_kt": [], "val_mae_kt": []}
    best, bad = float("inf"), 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, tm = run_epoch(model, train_loader, loss_fn, device, mean, std, opt)
        vl, vm = run_epoch(model, val_loader, loss_fn, device, mean, std)
        sched.step(vl)
        for k, v in zip(history, (tl, vl, tm, vm)):
            history[k].append(v)
        print(f"epoch {epoch:03d}/{epochs} | train loss {tl:.4f} MAE {tm:.2f} kt | val loss {vl:.4f} "
              f"MAE {vm:.2f} kt | {time.time() - t0:.0f}s")
        if vl < best:
            best, bad = vl, 0
            torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                        "config": {"image_size": cfg["preprocessing"]["image_size"],
                                   "normalization": cfg["preprocessing"]["normalization"],
                                   "max_pixel_value": cfg["preprocessing"]["max_pixel_value"],
                                   "in_channels": cfg["preprocessing"]["in_channels"], "dropout": t["dropout"]},
                        "target_mean": mean, "target_std": std, "wind_unit": "knots", "category_scheme": scheme,
                        "demo_run": bool(a.demo_run), "epoch": epoch, "val_loss": vl, "val_mae_kt": vm,
                        "seed": seed, "pretrained_start": bool(pretrained), "balanced_sampling": bool(a.balance),
                        "split_strategy": cfg["split"]["strategy"], "trained_on_config": str(a.config)}, ckpt_path)
        else:
            bad += 1
            if bad >= t["patience"]:
                print(f"Early stopping (no validation improvement for {t['patience']} epochs).")
                break
    (out_dir / f"{prefix}{model_name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_loss_curves(history, out_dir / f"{prefix}{model_name}_loss_curve.png")
    print(f"Best checkpoint: {ckpt_path} (val loss {best:.4f}). Next: python -m src.evaluate --checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
