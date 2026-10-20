"""Needs PyTorch (+ torchvision for ResNet). Skipped automatically if torch is not installed."""
import pytest

torch = pytest.importorskip("torch")

from src.model import build_model, count_parameters, mc_dropout_predict  # noqa: E402


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_output_shape(name):
    model = build_model(name, in_channels=1, pretrained=False)
    out = model(torch.rand(3, 1, 96, 96))
    assert out.shape == (3,)
    assert count_parameters(model) > 0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("nope")


def test_mc_dropout_shapes_and_variance():
    model = build_model("cnn", 1, False, dropout=0.5)
    mean, std = mc_dropout_predict(model, torch.rand(4, 1, 64, 64), n_samples=8)
    assert mean.shape == (4,) and std.shape == (4,) and float(std.min()) >= 0
    assert float(std.max()) > 0            # dropout is active, so passes differ


def test_checkpoint_roundtrip(tmp_path):
    from src.predict import load_checkpoint, predict_array
    import numpy as np
    model = build_model("cnn", 1, False, 0.3)
    p = tmp_path / "demo_best_cnn.pt"
    torch.save({"model_name": "cnn", "state_dict": model.state_dict(),
                "config": {"image_size": 64, "normalization": "fixed", "max_pixel_value": 255.0,
                           "in_channels": 1, "dropout": 0.3},
                "target_mean": 60.0, "target_std": 25.0, "category_scheme": "imd", "demo_run": True}, p)
    lm = load_checkpoint(p, "cpu")
    res = predict_array(lm, np.random.default_rng(0).uniform(0, 255, (80, 80)).astype(np.float32), mc_samples=4)
    assert res["wind_kt"] >= 0 and res["std_kt"] is not None and "category" in res
