"""Create a SYNTHETIC smoke-test dataset (data/smoke) to check that the software pipeline runs.

!! These images are artificial blobs, NOT satellite data, and the labels are made up. Anything trained
!! on them is a software test only and must never be reported as a cyclone-intensity result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def make_image(wind_kt: float, rng: np.random.Generator, size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    core = 4 + 0.15 * wind_kt                      # artificial rule: stronger -> bigger bright core
    img = 200 * np.exp(-(r / (size * 0.35)) ** 2) + 55 * np.exp(-(r / core) ** 2) + rng.normal(0, 6, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_smoke_dataset(out_dir="data/smoke", n_storms=12, frames=8, seed=0, base_config="config/config.yaml"):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "ir").mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(n_storms):
        peak = rng.uniform(40, 130)
        for f in range(frames):
            wind = float(np.clip(peak * np.sin(np.pi * (f + 1) / (frames + 1)) + 20, 20, 140))
            name = f"ir/SMOKE{s:02d}_{f:02d}.png"
            Image.fromarray(make_image(wind, rng)).save(out / name)
            rows.append({"image_path": name, "storm_id": f"SMOKE{s:02d}",
                         "timestamp": f"2000-01-{s + 1:02d} {f * 6 % 24:02d}:00", "wind_speed_kt": round(wind, 1)})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    cfg = yaml.safe_load(open(base_config, encoding="utf-8"))
    cfg["data"].update(manifest_csv=str(out / "manifest.csv"), image_root=str(out),
                       processed_dir=str(out / "processed"), report_path="reports/smoke_validation.json")
    cfg["preprocessing"]["image_size"] = 96
    cfg["training"].update(epochs=3, batch_size=16, num_workers=0, pretrained=False, model="cnn")
    cfg_path = Path("config/config_smoke.yaml")
    cfg_path.write_text("# SYNTHETIC smoke-test config. Not real data.\n" + yaml.safe_dump(cfg, sort_keys=False),
                        encoding="utf-8")
    return out / "manifest.csv", cfg_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/smoke")
    a = ap.parse_args()
    m, c = make_smoke_dataset(a.out)
    print(f"Created SYNTHETIC smoke data: {m} and config {c}. NOT real cyclone data.")
