"""Tests for src/preprocessing.py. Images here are tiny synthetic TEST FIXTURES only."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import (ImageLoadError, augment_image, load_image, normalize_image,
                               preprocess_array, preprocess_image, resize_image)


def _write_png(path, size=(40, 30), value=None):
    rng = np.random.default_rng(0)
    arr = (np.full(size[::-1], value, dtype=np.uint8) if value is not None
           else rng.integers(0, 256, size[::-1], dtype=np.uint8))
    Image.fromarray(arr).save(path)


def test_load_image_returns_2d_float32():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.png"
        _write_png(p)
        arr = load_image(p)
        assert arr.shape == (30, 40) and arr.dtype == np.float32


def test_load_image_missing_file_raises():
    try:
        load_image("does_not_exist.png")
    except ImageLoadError:
        return
    raise AssertionError("expected ImageLoadError")


def test_load_image_corrupt_file_raises():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.png"
        p.write_bytes(b"not an image")
        try:
            load_image(p)
        except ImageLoadError:
            return
        raise AssertionError("expected ImageLoadError")


def test_resize_shape():
    assert resize_image(np.zeros((30, 40), np.float32), 64).shape == (64, 64)


def test_normalize_fixed_range():
    out = normalize_image(np.array([[0, 127.5, 255]], np.float32), "fixed", 255.0)
    assert out.min() >= 0 and out.max() <= 1 and abs(out[0, 1] - 0.5) < 1e-6


def test_normalize_per_image_and_constant():
    out = normalize_image(np.array([[10, 20, 30]], np.float32), "per_image")
    assert out.min() == 0 and out.max() == 1
    assert normalize_image(np.full((3, 3), 7, np.float32), "per_image").sum() == 0


def test_normalize_bad_mode():
    try:
        normalize_image(np.zeros((2, 2)), "nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_preprocess_shape_and_dtype():
    out = preprocess_array(np.random.default_rng(1).random((50, 70)).astype(np.float32) * 255, 32)
    assert out.shape == (1, 32, 32) and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1


def test_preprocess_image_from_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.png"
        _write_png(p)
        assert preprocess_image(p, image_size=16).shape == (1, 16, 16)


def test_augment_keeps_shape_and_range_and_is_seeded():
    img = np.random.default_rng(2).random((32, 32)).astype(np.float32)
    a = augment_image(img, np.random.default_rng(5))
    b = augment_image(img, np.random.default_rng(5))
    assert a.shape == img.shape and a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, b)
