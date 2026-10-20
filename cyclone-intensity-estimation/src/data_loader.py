"""Manifest validation, leakage-safe splitting and a PyTorch-compatible dataset.

Expected manifest CSV columns (one row per satellite image):
    image_path     path to the image, relative to data.image_root
    storm_id       identifier of the cyclone, e.g. "2019_FANI"
    timestamp      UTC time of the image, e.g. "2019-05-01 03:00"
    wind_speed_kt  maximum sustained wind speed in KNOTS (the regression target)
Optional columns: category, r34_km, rmw_km (size labels; only used if present).

Run a validation from the command line:
    python -m src.data_loader --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.preprocessing import ImageLoadError, augment_image, load_image, preprocess_array

REQUIRED_COLUMNS = ["image_path", "storm_id", "timestamp", "wind_speed_kt"]

# Lower bounds in knots. Always state which scheme you used in your report.
CATEGORY_SCHEMES = {
    "imd": [  # India Meteorological Department (North Indian Ocean)
        (0, "Below depression"), (17, "Depression"), (28, "Deep Depression"),
        (34, "Cyclonic Storm"), (48, "Severe Cyclonic Storm"),
        (64, "Very Severe Cyclonic Storm"), (90, "Extremely Severe Cyclonic Storm"),
        (120, "Super Cyclonic Storm"),
    ],
    "saffir_simpson": [  # Atlantic / East Pacific hurricane scale
        (0, "Tropical Depression"), (34, "Tropical Storm"), (64, "Category 1"),
        (83, "Category 2"), (96, "Category 3"), (113, "Category 4"), (137, "Category 5"),
    ],
}


class ManifestError(Exception):
    """Raised when the manifest is missing, malformed or has no usable rows."""


def intensity_category(wind_kt: float, scheme: str = "imd") -> str:
    """Map a wind speed in knots to a category name for the chosen scheme."""
    if scheme not in CATEGORY_SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}; choose from {list(CATEGORY_SCHEMES)}")
    if wind_kt is None or not np.isfinite(wind_kt) or wind_kt < 0:
        raise ValueError(f"Invalid wind speed: {wind_kt!r}")
    name = CATEGORY_SCHEMES[scheme][0][1]
    for lower, label in CATEGORY_SCHEMES[scheme]:
        if wind_kt >= lower:
            name = label
    return name


def validate_manifest(df: pd.DataFrame, image_root: str | Path = ".", check_images: bool = True,
                      wind_range=(0.0, 200.0)):
    """Remove unusable rows and return (clean_df, report). Nothing is silently invented."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"Manifest is missing required column(s): {missing}. "
                            f"Required columns are {REQUIRED_COLUMNS}.")
    state = {"df": df.copy()}
    dropped: dict[str, int] = {}

    def drop(mask: pd.Series, reason: str) -> None:
        n = int(mask.sum())
        if n:
            dropped[reason] = n
            state["df"] = state["df"][~mask]

    d = state["df"]
    d["wind_speed_kt"] = pd.to_numeric(d["wind_speed_kt"], errors="coerce")
    drop(d["wind_speed_kt"].isna(), "missing_or_non_numeric_wind")
    d = state["df"]
    drop(~d["wind_speed_kt"].between(wind_range[0], wind_range[1]), "wind_out_of_range")
    d = state["df"]
    drop(d["storm_id"].isna() | (d["storm_id"].astype(str).str.strip() == ""), "missing_storm_id")
    d = state["df"]
    state["df"] = d.assign(timestamp=pd.to_datetime(d["timestamp"], errors="coerce"))
    drop(state["df"]["timestamp"].isna(), "invalid_timestamp")
    d = state["df"]
    drop(d["image_path"].isna() | d["image_path"].duplicated(keep="first"), "missing_or_duplicate_image_path")

    root = Path(image_root)
    d = state["df"].copy()
    d["abs_path"] = [str(root / str(p)) for p in d["image_path"]]
    state["df"] = d
    drop(~d["abs_path"].map(lambda p: Path(p).is_file()), "missing_file")

    if check_images:
        def readable(p: str) -> bool:
            try:
                with Image.open(p) as im:
                    im.verify()
                return True
            except Exception:
                return False
        d = state["df"]
        drop(~d["abs_path"].map(readable), "unreadable_image")

    clean = state["df"].reset_index(drop=True)
    report = {
        "rows_in": int(len(df)),
        "rows_out": int(len(clean)),
        "dropped": dropped,
        "n_storms": int(clean["storm_id"].nunique()) if len(clean) else 0,
        "wind_speed_kt": ({"min": float(clean["wind_speed_kt"].min()),
                           "max": float(clean["wind_speed_kt"].max()),
                           "mean": float(clean["wind_speed_kt"].mean())} if len(clean) else None),
    }
    return clean, report


def load_manifest(csv_path: str | Path, image_root: str | Path = ".", check_images: bool = True,
                  wind_range=(0.0, 200.0)):
    """Read a manifest CSV, validate it, and raise a clear error if nothing is usable."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ManifestError(
            f"Manifest not found: {csv_path}. Create it as described in README.md "
            "(columns: image_path, storm_id, timestamp, wind_speed_kt).")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ManifestError(f"Could not read manifest {csv_path}: {exc}") from exc
    clean, report = validate_manifest(df, image_root, check_images, wind_range)
    if clean.empty:
        raise ManifestError(f"No valid rows left after validation. Details: {report}")
    return clean, report


def split_by_storm(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42):
    """Whole storms go to exactly one split, so neighbouring frames cannot leak."""
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Need train_frac > 0, val_frac > 0 and train_frac + val_frac < 1")
    storms = np.array(sorted(df["storm_id"].astype(str).unique()))
    if len(storms) < 3:
        raise ManifestError(f"Storm-wise splitting needs at least 3 storms; found {len(storms)}.")
    storms = np.random.default_rng(seed).permutation(storms)
    n = len(storms)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * (1 - train_frac - val_frac))))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ManifestError("Too few storms for the requested split fractions.")
    groups = {"train": storms[:n_train], "val": storms[n_train:n_train + n_val], "test": storms[n_train + n_val:]}
    ids = df["storm_id"].astype(str)
    splits = {k: df[ids.isin(v)].reset_index(drop=True) for k, v in groups.items()}
    assert_no_storm_leakage(splits)
    return splits


def split_chronological(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Oldest images -> train, newest -> test. A storm may straddle a boundary; report this."""
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Need train_frac > 0, val_frac > 0 and train_frac + val_frac < 1")
    d = df.sort_values("timestamp").reset_index(drop=True)
    n = len(d)
    if n < 3:
        raise ManifestError("Need at least 3 images for a chronological split.")
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return {"train": d.iloc[:a].reset_index(drop=True), "val": d.iloc[a:b].reset_index(drop=True),
            "test": d.iloc[b:].reset_index(drop=True)}


def split_holdout_group(df: pd.DataFrame, group_col: str, holdout_value: str, val_frac: float = 0.15,
                        seed: int = 42):
    """Cross-region test: every storm of one group (e.g. basin == 'IO') is the test set.

    The remaining storms are split storm-wise into train and validation.
    """
    if group_col not in df.columns:
        raise ManifestError(f"Column {group_col!r} not found in the manifest; cannot hold out a group.")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    mask = df[group_col].astype(str) == str(holdout_value)
    if not mask.any():
        raise ManifestError(f"No rows with {group_col} == {holdout_value!r}. "
                            f"Available values: {sorted(df[group_col].astype(str).unique())}")
    test = df[mask].reset_index(drop=True)
    rest = df[~mask]
    storms = np.array(sorted(rest["storm_id"].astype(str).unique()))
    if len(storms) < 2:
        raise ManifestError("Need at least 2 storms outside the held-out group for train/val.")
    storms = np.random.default_rng(seed).permutation(storms)
    n_val = max(1, int(round(len(storms) * val_frac)))
    ids = rest["storm_id"].astype(str)
    splits = {"train": rest[ids.isin(storms[n_val:])].reset_index(drop=True),
              "val": rest[ids.isin(storms[:n_val])].reset_index(drop=True), "test": test}
    assert_no_storm_leakage(splits)
    return splits


def assert_no_storm_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any storm appears in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]["storm_id"].astype(str)) & set(splits[b]["storm_id"].astype(str))
            if shared:
                raise AssertionError(f"Storm leakage between {a} and {b}: {sorted(shared)[:5]}")


class CycloneDataset:
    """Map-style dataset (works with torch.utils.data.DataLoader; needs no torch itself).

    __getitem__ returns (image, wind_kt): image float32 (1, H, W) in [0, 1], wind float32 scalar.
    """

    def __init__(self, df: pd.DataFrame, image_size: int = 224, normalization: str = "fixed",
                 max_pixel_value: float = 255.0, augment: bool = False, aug_cfg: dict | None = None,
                 seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.image_size, self.normalization, self.max_pixel_value = image_size, normalization, max_pixel_value
        self.augment, self.aug_cfg = augment, aug_cfg or {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            arr = load_image(row["abs_path"])
        except ImageLoadError as exc:
            raise RuntimeError(f"Failed to load training image: {exc}") from exc
        x = preprocess_array(arr, self.image_size, self.normalization, self.max_pixel_value)
        if self.augment:
            x = augment_image(x[0], self.rng,
                              self.aug_cfg.get("max_rotation_deg", 180.0),
                              self.aug_cfg.get("brightness_jitter", 0.05),
                              self.aug_cfg.get("horizontal_flip", False))[np.newaxis, ...]
        return x, np.float32(row["wind_speed_kt"])


def load_config(path: str | Path = "config/config.yaml") -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_splits(cfg: dict):
    """Validate the manifest, split it, write CSVs + a JSON report. Returns (splits, report)."""
    d = cfg["data"]
    clean, report = load_manifest(d["manifest_csv"], d["image_root"], d.get("check_images_open", True),
                                  tuple(d.get("wind_range_kt", [0, 200])))
    s = cfg["split"]
    if s["strategy"] == "storm":
        splits = split_by_storm(clean, s["train_fraction"], s["val_fraction"], cfg.get("seed", 42))
    elif s["strategy"] == "holdout_basin":
        splits = split_holdout_group(clean, "basin", s.get("holdout_basin"), s["val_fraction"],
                                     cfg.get("seed", 42))
    elif s["strategy"] == "chronological":
        splits = split_chronological(clean, s["train_fraction"], s["val_fraction"])
    else:
        raise ValueError(f"Unknown split strategy: {s['strategy']!r}")
    out = Path(d["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out / f"{name}.csv", index=False)
    report["split_strategy"] = s["strategy"]
    report["split_sizes"] = {k: int(len(v)) for k, v in splits.items()}
    report["split_storms"] = {k: int(v["storm_id"].nunique()) for k, v in splits.items()}
    if "basin" in clean.columns:
        report["basin_counts"] = {str(k): int(v) for k, v in clean["basin"].value_counts().items()}
    report["wind_unit"] = d.get("wind_unit", "knots")
    rp = Path(d["report_path"])
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return splits, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the dataset manifest and create splits.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    try:
        _, rep = prepare_splits(load_config(args.config))
    except (ManifestError, FileNotFoundError, ValueError) as err:
        raise SystemExit(f"ERROR: {err}")
    print(json.dumps(rep, indent=2))
