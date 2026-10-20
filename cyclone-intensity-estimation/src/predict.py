"""Loading a trained checkpoint and predicting wind speed for one image.

Everything that needs PyTorch is imported lazily so the validation helpers work without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_loader import intensity_category
from src.preprocessing import ImageLoadError, load_image_from_file, preprocess_array

MIN_SIDE = 64
KT_TO_MS = 0.514444


def find_checkpoints(artifacts_dir: str | Path = "artifacts") -> list[Path]:
    d = Path(artifacts_dir)
    return sorted(d.glob("*.pt")) if d.is_dir() else []


def validate_array(arr: np.ndarray, original_mode: str = "L") -> list[str]:
    """Raise ImageLoadError for unusable images; return a list of non-fatal warnings."""
    warnings: list[str] = []
    if min(arr.shape) < MIN_SIDE:
        raise ImageLoadError(f"Image is too small ({arr.shape[1]}x{arr.shape[0]} px); need at least "
                             f"{MIN_SIDE}x{MIN_SIDE}.")
    if float(arr.std()) < 1e-6:
        raise ImageLoadError("Image is a single constant colour - there is no cloud structure to analyse.")
    if original_mode in ("RGB", "RGBA", "P", "CMYK"):
        warnings.append("This is a colour image. It was converted to grey, but the model expects a single-channel "
                        "infrared brightness image scaled like the training data; results are unreliable.")
    if float(arr.max()) <= 1.0:
        warnings.append("Pixel values are all <= 1. The model expects 8-bit values (0-255); this image may already "
                        "be normalised and will look almost black.")
    h, w = arr.shape
    if max(h, w) / min(h, w) > 1.25:
        warnings.append("The image is not square; it will be stretched to a square, which distorts the storm.")
    return warnings


def load_uploaded_image(file_obj) -> tuple[np.ndarray, list[str]]:
    arr, mode = load_image_from_file(file_obj)
    return arr, validate_array(arr, mode)


@dataclass
class LoadedModel:
    model: object
    device: object
    config: dict
    target_mean: float
    target_std: float
    meta: dict


def load_checkpoint(path: str | Path, device: str | None = None) -> LoadedModel:
    import torch
    from src.model import build_model
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    cfg = ckpt["config"]
    model = build_model(ckpt["model_name"], cfg["in_channels"], pretrained=False, dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "config")}
    return LoadedModel(model, dev, cfg, float(ckpt["target_mean"]), float(ckpt["target_std"]), meta)


def predict_array(lm: LoadedModel, arr: np.ndarray, mc_samples: int = 0) -> dict:
    """Predict wind speed (knots) for one raw 2-D image array."""
    import torch
    from src.model import mc_dropout_predict
    c = lm.config
    x = preprocess_array(arr, c["image_size"], c["normalization"], c["max_pixel_value"])
    xt = torch.from_numpy(x).unsqueeze(0).to(lm.device)
    if mc_samples and mc_samples >= 2:
        mean, std = mc_dropout_predict(lm.model, xt, mc_samples)
        std_kt = float(std.item() * lm.target_std)
    else:
        with torch.no_grad():
            lm.model.eval()
            mean = lm.model(xt)
        std_kt = None
    wind = max(float(mean.item() * lm.target_std + lm.target_mean), 0.0)
    scheme = lm.meta.get("category_scheme", "imd")
    return {"wind_kt": wind, "wind_ms": wind * KT_TO_MS, "std_kt": std_kt,
            "category": intensity_category(wind, scheme), "scheme": scheme}


def predict_file(checkpoint: str | Path, image_path: str | Path, mc_samples: int = 0) -> dict:
    lm = load_checkpoint(checkpoint)
    return predict_array(lm, load_image(image_path), mc_samples)


from src.preprocessing import load_image  # noqa: E402  (kept at the bottom to keep the top light)

if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Predict wind speed (kt) for one image with a trained checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mc-samples", type=int, default=0)
    a = ap.parse_args()
    try:
        print(json.dumps(predict_file(a.checkpoint, a.image, a.mc_samples), indent=2))
    except (FileNotFoundError, ImageLoadError) as err:
        raise SystemExit(f"ERROR: {err}")
