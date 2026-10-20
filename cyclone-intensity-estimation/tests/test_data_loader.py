"""Tests for src/data_loader.py. Synthetic images/labels exist ONLY inside these tests."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data_loader import (CycloneDataset, ManifestError, assert_no_storm_leakage, intensity_category,
                             load_manifest, split_by_storm, split_chronological, validate_manifest)


def _make_dataset(root: Path, n_storms=10, frames=4):
    rows = []
    for s in range(n_storms):
        for f in range(frames):
            name = f"s{s}_f{f}.png"
            Image.fromarray(np.full((20, 20), 10 * f, np.uint8)).save(root / name)
            rows.append({"image_path": name, "storm_id": f"STORM{s}",
                         "timestamp": f"2020-01-{s + 1:02d} {f * 3:02d}:00", "wind_speed_kt": 30 + 5 * f + s})
    return pd.DataFrame(rows)


def test_validate_clean_manifest_keeps_everything():
    with tempfile.TemporaryDirectory() as d:
        df = _make_dataset(Path(d))
        clean, rep = validate_manifest(df, d)
        assert len(clean) == len(df) and rep["dropped"] == {} and rep["n_storms"] == 10


def test_validate_drops_bad_rows_with_reasons():
    with tempfile.TemporaryDirectory() as d:
        df = _make_dataset(Path(d))
        df.loc[0, "wind_speed_kt"] = np.nan            # missing label
        df.loc[1, "wind_speed_kt"] = 999               # out of range
        df.loc[2, "image_path"] = "no_such_file.png"   # missing file
        df.loc[3, "timestamp"] = "not a date"          # bad timestamp
        (Path(d) / df.loc[4, "image_path"]).write_bytes(b"junk")  # corrupt image
        clean, rep = validate_manifest(df, d)
        assert len(clean) == len(df) - 5
        for key in ["missing_or_non_numeric_wind", "wind_out_of_range", "missing_file",
                    "invalid_timestamp", "unreadable_image"]:
            assert rep["dropped"].get(key) == 1, (key, rep["dropped"])


def test_missing_required_column_raises():
    try:
        validate_manifest(pd.DataFrame({"image_path": ["a.png"]}))
    except ManifestError:
        return
    raise AssertionError("expected ManifestError")


def test_load_manifest_missing_csv_raises_helpful_error():
    try:
        load_manifest("nope/manifest.csv")
    except ManifestError as e:
        assert "README" in str(e)
        return
    raise AssertionError("expected ManifestError")


def test_storm_split_has_no_leakage_and_covers_all_rows():
    with tempfile.TemporaryDirectory() as d:
        clean, _ = validate_manifest(_make_dataset(Path(d)), d)
        sp = split_by_storm(clean, 0.6, 0.2, seed=1)
        assert_no_storm_leakage(sp)
        assert sum(len(v) for v in sp.values()) == len(clean)
        assert all(len(v) > 0 for v in sp.values())


def test_storm_split_is_reproducible():
    with tempfile.TemporaryDirectory() as d:
        clean, _ = validate_manifest(_make_dataset(Path(d)), d)
        a, b = split_by_storm(clean, seed=3), split_by_storm(clean, seed=3)
        assert set(a["test"]["storm_id"]) == set(b["test"]["storm_id"])


def test_storm_split_needs_three_storms():
    with tempfile.TemporaryDirectory() as d:
        clean, _ = validate_manifest(_make_dataset(Path(d), n_storms=2), d)
        try:
            split_by_storm(clean)
        except ManifestError:
            return
        raise AssertionError("expected ManifestError")


def test_leakage_check_detects_overlap():
    df = pd.DataFrame({"storm_id": ["A", "B"]})
    try:
        assert_no_storm_leakage({"train": df, "test": df})
    except AssertionError:
        return
    raise AssertionError("leakage should have been detected")


def test_chronological_split_order():
    with tempfile.TemporaryDirectory() as d:
        clean, _ = validate_manifest(_make_dataset(Path(d)), d)
        sp = split_chronological(clean)
        assert sp["train"]["timestamp"].max() <= sp["val"]["timestamp"].min()
        assert sp["val"]["timestamp"].max() <= sp["test"]["timestamp"].min()


def test_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as d:
        clean, _ = validate_manifest(_make_dataset(Path(d)), d)
        ds = CycloneDataset(clean, image_size=32, augment=True)
        x, y = ds[0]
        assert len(ds) == len(clean) and x.shape == (1, 32, 32) and x.dtype == np.float32
        assert np.isscalar(y) or y.shape == ()


def test_intensity_categories():
    assert intensity_category(20, "imd") == "Depression"
    assert intensity_category(34, "imd") == "Cyclonic Storm"
    assert intensity_category(125, "imd") == "Super Cyclonic Storm"
    assert intensity_category(100, "saffir_simpson") == "Category 3"
    try:
        intensity_category(-5)
    except ValueError:
        return
    raise AssertionError("negative wind should raise")
