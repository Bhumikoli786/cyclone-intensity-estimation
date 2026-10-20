"""Image preprocessing: load, resize, normalise, augment.

Uses only NumPy and Pillow so it can be tested without PyTorch.
Every function works on 2-D float32 arrays (height x width) unless stated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ImageLoadError(Exception):
    """Raised when an image file is missing, corrupt or unsupported."""


def pil_to_array(im: Image.Image) -> np.ndarray:
    """Convert an open PIL image to a 2-D float32 array (single channel)."""
    # keep 16-bit / float images as they are, convert everything else to 8-bit grey
    if im.mode not in ("I;16", "I", "F"):
        im = im.convert("L")
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ImageLoadError(f"Unexpected image shape {arr.shape}")
    return arr


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as a 2-D float32 array (single channel, raw pixel values)."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Image file not found: {path}")
    try:
        with Image.open(path) as im:
            return pil_to_array(im)
    except ImageLoadError:
        raise
    except Exception as exc:  # Pillow raises several different error types
        raise ImageLoadError(f"Could not read image {path}: {exc}") from exc


def load_image_from_file(file_obj) -> tuple[np.ndarray, str]:
    """Load from a file-like object (e.g. a Streamlit upload). Returns (array, original_mode)."""
    try:
        with Image.open(file_obj) as im:
            mode = im.mode
            return pil_to_array(im), mode
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"Could not read the uploaded file as an image: {exc}") from exc


def resize_image(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D array to (size, size) with bilinear interpolation."""
    if size <= 0:
        raise ValueError("size must be a positive integer")
    im = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32))
    im = im.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def normalize_image(arr: np.ndarray, mode: str = "fixed", max_pixel_value: float = 255.0) -> np.ndarray:
    """Scale pixel values to roughly [0, 1].

    mode="fixed":     divide by max_pixel_value (same scaling for every image)
    mode="per_image": min-max scale each image on its own
    """
    arr = arr.astype(np.float32)
    if mode == "fixed":
        return np.clip(arr / float(max_pixel_value), 0.0, 1.0)
    if mode == "per_image":
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-8:  # constant image: avoid divide-by-zero
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)
    raise ValueError(f"Unknown normalization mode: {mode!r} (use 'fixed' or 'per_image')")


def augment_image(
    arr01: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 180.0,
    brightness_jitter: float = 0.05,
    horizontal_flip: bool = False,
) -> np.ndarray:
    """Training-time augmentation on an already-normalised [0, 1] image."""
    out = arr01.astype(np.float32)
    if max_rotation_deg > 0:
        angle = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
        im = Image.fromarray(np.ascontiguousarray(out))
        out = np.asarray(im.rotate(angle, resample=Image.BILINEAR, fillcolor=0.0), dtype=np.float32)
    if horizontal_flip and rng.random() < 0.5:
        out = out[:, ::-1].copy()
    if brightness_jitter > 0:
        out = out * float(rng.uniform(1 - brightness_jitter, 1 + brightness_jitter))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preprocess_array(
    arr: np.ndarray,
    image_size: int = 224,
    normalization: str = "fixed",
    max_pixel_value: float = 255.0,
) -> np.ndarray:
    """Raw 2-D array -> model input of shape (1, image_size, image_size), float32."""
    arr = resize_image(arr, image_size)
    arr = normalize_image(arr, normalization, max_pixel_value)
    return arr[np.newaxis, ...].astype(np.float32)


def preprocess_image(path: str | Path, **kwargs) -> np.ndarray:
    """Load an image file and preprocess it (see preprocess_array for options)."""
    return preprocess_array(load_image(path), **kwargs)
