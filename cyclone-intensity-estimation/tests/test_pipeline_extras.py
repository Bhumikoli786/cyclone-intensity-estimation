"""Tests for the TCIR converter, smoke data, hold-out split, baselines and upload validation."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.baselines import MeanBaseline, RidgeBaseline
from src.data_loader import (ManifestError, assert_no_storm_leakage, load_manifest, split_holdout_group)
from src.make_smoke_data import make_smoke_dataset
from src.predict import find_checkpoints, load_uploaded_image, validate_array
from src.prepare_tcir import ConversionError, convert_arrays, detect_columns, parse_time
from src.preprocessing import ImageLoadError


def _fake_tcir(n=40):
    """Synthetic arrays shaped like TCIR (test fixture only, not real data)."""
    rng = np.random.default_rng(0)
    matrix = rng.uniform(190, 300, (n, 32, 32, 4)).astype(np.float32)
    matrix[3, :, :, 0] = np.nan                                    # fully missing frame
    info = pd.DataFrame({"ID": [f"S{i // 8}" for i in range(n)],
                         "time": [2017010100 + 3 * (i % 8) for i in range(n)],
                         "Vmax": rng.uniform(20, 120, n), "region": ["WP" if i < 24 else "IO" for i in range(n)]})
    return info, matrix


def test_parse_time_digit_strings_and_text():
    out = parse_time(pd.Series([2017010103, "2017-01-01 06:00", "garbage"]))
    assert out.iloc[0] == pd.Timestamp("2017-01-01 03:00") and out.iloc[1] == pd.Timestamp("2017-01-01 06:00")
    assert pd.isna(out.iloc[2])


def test_detect_columns_and_override_error():
    info, _ = _fake_tcir()
    assert detect_columns(info)["wind"] == "Vmax"
    try:
        detect_columns(info, {"wind": "nope"})
    except ConversionError:
        return
    raise AssertionError("expected ConversionError")


def test_convert_arrays_writes_valid_manifest():
    info, matrix = _fake_tcir()
    with tempfile.TemporaryDirectory() as d:
        man, stats = convert_arrays(info, matrix, d)
        assert len(man) == 39 and stats["frames_skipped_mostly_nan"] == 1      # the all-NaN frame is skipped
        assert set(man["basin"]) == {"WP", "IO"}
        clean, rep = load_manifest(Path(d) / "manifest.csv", d)
        assert len(clean) == len(man) and rep["dropped"] == {}
        assert np.asarray(Image.open(Path(d) / man.loc[0, "image_path"])).dtype == np.uint8


def test_convert_arrays_stride_basin_and_errors():
    info, matrix = _fake_tcir()
    with tempfile.TemporaryDirectory() as d:
        man, _ = convert_arrays(info, matrix, d, stride=2, basins=["IO"])
        assert set(man["basin"]) == {"IO"} and len(man) < 16
        try:
            convert_arrays(info.iloc[:5], matrix, d)
        except ConversionError:
            return
    raise AssertionError("row-count mismatch should raise")


def test_holdout_basin_split_is_cross_basin_and_leak_free():
    info, matrix = _fake_tcir()
    with tempfile.TemporaryDirectory() as d:
        man, _ = convert_arrays(info, matrix, d)
        sp = split_holdout_group(man, "basin", "IO", val_frac=0.5, seed=1)
        assert set(sp["test"]["basin"]) == {"IO"} and "IO" not in set(sp["train"]["basin"])
        assert_no_storm_leakage(sp)
        try:
            split_holdout_group(man, "basin", "XX")
        except ManifestError as e:
            assert "Available values" in str(e)
            return
    raise AssertionError("unknown basin should raise")


def test_smoke_dataset_is_loadable():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(Path(__file__).resolve().parents[1])
        try:
            m, cfg = make_smoke_dataset(Path(d) / "smoke", n_storms=4, frames=3)
            clean, rep = load_manifest(m, Path(d) / "smoke")
            assert len(clean) == 12 and clean["wind_speed_kt"].between(20, 140).all()
        finally:
            os.chdir(old)
            Path("config/config_smoke.yaml").unlink(missing_ok=True)


def test_baselines():
    assert MeanBaseline().fit([10, 20]).predict(3).tolist() == [15.0, 15.0, 15.0]
    with tempfile.TemporaryDirectory() as d:
        rows = []
        for i in range(12):
            Image.fromarray(np.full((20, 20), 10 * i, np.uint8)).save(Path(d) / f"{i}.png")
            rows.append({"abs_path": str(Path(d) / f"{i}.png"), "wind_speed_kt": 20.0 + 5 * i})
        df = pd.DataFrame(rows)
        r = RidgeBaseline(size=8).fit(df)
        assert r.predict(df).shape == (12,)


def test_upload_validation():
    import io
    def png(arr, mode=None):
        b = io.BytesIO(); Image.fromarray(arr).save(b, format="PNG"); b.seek(0); return b
    arr, warns = load_uploaded_image(png(np.random.default_rng(0).integers(0, 255, (80, 80), dtype=np.uint8)))
    assert arr.shape == (80, 80) and warns == []
    _, warns = load_uploaded_image(png(np.random.default_rng(0).integers(0, 255, (80, 80, 3), dtype=np.uint8)))
    assert any("colour" in w for w in warns)
    for bad in (png(np.zeros((80, 80), np.uint8)), png(np.random.default_rng(0).integers(0, 255, (20, 20), dtype=np.uint8))):
        try:
            load_uploaded_image(bad)
        except ImageLoadError:
            continue
        raise AssertionError("expected ImageLoadError")
    try:
        load_uploaded_image(io.BytesIO(b"not an image"))
    except ImageLoadError:
        return
    raise AssertionError("expected ImageLoadError")


def test_find_checkpoints_missing_dir():
    assert find_checkpoints("definitely_missing_dir") == []
