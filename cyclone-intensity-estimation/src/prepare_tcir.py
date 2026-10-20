"""Convert the TCIR HDF5 dataset into 8-bit infrared PNG frames + a manifest CSV.

TCIR (Chen, Chen, Lin) stores `matrix` (N x 201 x 201 x 4 channels: IR1, WV, VIS, PMW) and `info`
(a pandas table). The exact info column names are NOT assumed: run --inspect first, and pass
--storm-col / --time-col / --wind-col / --basin-col if the automatic detection is wrong.

Examples:
    python -m src.prepare_tcir --h5 TCIR.h5 --inspect
    python -m src.prepare_tcir --h5 TCIR.h5 --out data/raw --stride 3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CANDIDATES = {
    "storm_id": ["id", "storm_id", "sid", "tc_id"],
    "time": ["time", "timestamp", "datetime", "date"],
    "wind": ["vmax", "intensity", "wind", "msw", "max_wind"],
    "basin": ["region", "basin", "data_set", "dataset"],
}
KNOWN_BASINS_NOTE = "Check the unique values printed by --inspect to choose basin names."


class ConversionError(Exception):
    pass


def detect_columns(info: pd.DataFrame, overrides: dict | None = None) -> dict:
    """Find storm/time/wind/basin columns (case-insensitive); overrides win. Basin is optional."""
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    lower = {c.lower(): c for c in info.columns}
    found = {}
    for key, names in CANDIDATES.items():
        if key in overrides:
            if overrides[key] not in info.columns:
                raise ConversionError(f"Column {overrides[key]!r} not in info table: {list(info.columns)}")
            found[key] = overrides[key]
            continue
        for n in names:
            if n in lower:
                found[key] = lower[n]
                break
    missing = [k for k in ("storm_id", "time", "wind") if k not in found]
    if missing:
        raise ConversionError(f"Could not detect column(s) {missing} in {list(info.columns)}. "
                              "Pass --storm-col / --time-col / --wind-col explicitly.")
    return found


def parse_time(series: pd.Series) -> pd.Series:
    """Parse timestamps; handles YYYYMMDDHH / YYYYMMDDHHMM digit strings and normal date strings."""
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for n, fmt in ((10, "%Y%m%d%H"), (12, "%Y%m%d%H%M"), (8, "%Y%m%d")):
        m = s.str.fullmatch(r"\d{%d}" % n)
        if m.any():
            out[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
    rest = out.isna()
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], errors="coerce")
    return out


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")[:40] or "x"


def estimate_bt_range(frames: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([f[np.isfinite(f)].ravel() for f in frames]) if frames else np.array([])
    if vals.size == 0:
        raise ConversionError("No finite pixel values found in the sampled IR frames.")
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if hi - lo < 1e-6:
        raise ConversionError("IR frames are (almost) constant; check --ir-index.")
    return float(lo), float(hi)


def frame_to_uint8(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    f = np.where(np.isfinite(frame), frame, lo).astype(np.float32)
    return (np.clip((f - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def convert_arrays(info: pd.DataFrame, matrix, out_dir, ir_index: int = 0, stride: int = 1,
                   basins: list[str] | None = None, max_frames: int | None = None, wind_scale: float = 1.0,
                   columns: dict | None = None, seed: int = 42):
    """Write IR frames as PNG and return (manifest_df, stats). `matrix` may be a NumPy array or h5py dataset."""
    if len(info) != matrix.shape[0]:
        raise ConversionError(f"info has {len(info)} rows but matrix has {matrix.shape[0]} frames.")
    if matrix.ndim != 4 or not (0 <= ir_index < matrix.shape[3]):
        raise ConversionError(f"matrix shape {matrix.shape} / ir_index {ir_index} not valid (need N,H,W,C).")
    cols = detect_columns(info, columns)
    d = pd.DataFrame({"row": np.arange(len(info)),
                      "storm_id": info[cols["storm_id"]].astype(str).str.strip().values,
                      "timestamp": parse_time(info[cols["time"]]).values,
                      "wind": pd.to_numeric(info[cols["wind"]], errors="coerce").values * wind_scale})
    if "basin" in cols:
        d["basin"] = info[cols["basin"]].astype(str).values
    dropped = {"missing_wind": int(d["wind"].isna().sum()), "invalid_time": int(d["timestamp"].isna().sum())}
    d = d.dropna(subset=["wind", "timestamp"])
    d = d[d["storm_id"] != ""]
    if basins:
        if "basin" not in d.columns:
            raise ConversionError("--basins given but no basin column was detected; use --basin-col.")
        d = d[d["basin"].isin(basins)]
    d = d.sort_values(["storm_id", "timestamp"])
    if stride > 1:
        d = d[d.groupby("storm_id").cumcount() % stride == 0]
    if max_frames and len(d) > max_frames:
        d = d.sample(n=max_frames, random_state=seed).sort_values(["storm_id", "timestamp"])
    if d.empty:
        raise ConversionError("No frames left after filtering.")

    rng = np.random.default_rng(seed)
    sample_rows = rng.choice(d["row"].to_numpy(), size=min(300, len(d)), replace=False)
    lo, hi = estimate_bt_range([np.asarray(matrix[int(i), :, :, ir_index], dtype=np.float32)
                                for i in sorted(sample_rows)])

    out_dir = Path(out_dir)
    (out_dir / "ir").mkdir(parents=True, exist_ok=True)
    rows, bad_frames = [], 0
    for r in d.itertuples(index=False):
        frame = np.asarray(matrix[int(r.row), :, :, ir_index], dtype=np.float32)
        if np.isfinite(frame).mean() < 0.5:
            bad_frames += 1
            continue
        rel = f"ir/{_safe(r.storm_id)}_{pd.Timestamp(r.timestamp).strftime('%Y%m%d%H%M')}_{r.row}.png"
        Image.fromarray(frame_to_uint8(frame, lo, hi)).save(out_dir / rel)
        row = {"image_path": rel, "storm_id": r.storm_id,
               "timestamp": pd.Timestamp(r.timestamp).strftime("%Y-%m-%d %H:%M"), "wind_speed_kt": float(r.wind)}
        if "basin" in d.columns:
            row["basin"] = r.basin
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    stats = {"ir_index": ir_index, "scaling_range_raw_units": [lo, hi], "n_frames_written": len(rows),
             "dropped_before_write": dropped, "frames_skipped_mostly_nan": bad_frames,
             "columns_used": cols, "wind_scale_applied": wind_scale, "stride": stride,
             "note": "8-bit scaling: 0 = low raw value, 255 = high raw value (0.5-99.5 percentile range)."}
    (out_dir / "conversion_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return manifest, stats


def inspect_h5(h5_path: str) -> None:
    import h5py
    info = pd.read_hdf(h5_path, key="info", mode="r")
    with h5py.File(h5_path, "r") as hf:
        print("HDF5 keys:", list(hf.keys()))
        m = hf["matrix"]
        print("matrix shape:", m.shape, "dtype:", m.dtype, "(expected N x 201 x 201 x 4)")
        k = min(50, m.shape[0])
        for c in range(m.shape[3]):
            f = np.asarray(m[:k, :, :, c], dtype=np.float32)
            print(f"  channel {c}: min={np.nanmin(f):.2f} max={np.nanmax(f):.2f} nan_fraction={np.isnan(f).mean():.3f}")
    print("\ninfo columns:", list(info.columns))
    print(info.head())
    try:
        cols = detect_columns(info)
        print("\nauto-detected:", cols)
        w = pd.to_numeric(info[cols["wind"]], errors="coerce")
        print(f"wind column range: {w.min()} .. {w.max()}  (≈170 or below suggests knots; ≈85 or below suggests m/s)")
        if "basin" in cols:
            print("basin values:", info[cols["basin"]].astype(str).value_counts().to_dict(), KNOWN_BASINS_NOTE)
    except ConversionError as e:
        print("\nAuto-detection failed:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert TCIR HDF5 to PNG frames + manifest.csv")
    ap.add_argument("--h5", required=True, help="path to the TCIR .h5 file")
    ap.add_argument("--inspect", action="store_true", help="print file structure and exit")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--ir-index", type=int, default=0, help="channel index of IR1 (default 0)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame of each storm")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--basins", nargs="*", default=None)
    ap.add_argument("--wind-scale", type=float, default=1.0, help="multiply wind by this to get knots")
    ap.add_argument("--storm-col"); ap.add_argument("--time-col"); ap.add_argument("--wind-col")
    ap.add_argument("--basin-col")
    a = ap.parse_args()
    try:
        if a.inspect:
            inspect_h5(a.h5)
        else:
            import h5py
            info = pd.read_hdf(a.h5, key="info", mode="r")
            with h5py.File(a.h5, "r") as hf:
                man, st = convert_arrays(info, hf["matrix"], a.out, a.ir_index, a.stride, a.basins, a.max_frames,
                                         a.wind_scale, {"storm_id": a.storm_col, "time": a.time_col,
                                                        "wind": a.wind_col, "basin": a.basin_col})
            print(json.dumps(st, indent=2))
            print(f"Wrote {len(man)} frames and {Path(a.out) / 'manifest.csv'}")
    except (ConversionError, FileNotFoundError, KeyError, ImportError) as err:
        raise SystemExit(f"ERROR: {err}")
