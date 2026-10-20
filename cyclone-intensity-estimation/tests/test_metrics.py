import numpy as np

from src.metrics import (category_labels, classification_metrics, group_metrics, per_category_metrics,
                         regression_metrics, uncertainty_summary)


def test_regression_metrics_known_values():
    m = regression_metrics([10, 20, 30], [12, 18, 36])
    assert abs(m["mae_kt"] - (2 + 2 + 6) / 3) < 1e-9
    assert abs(m["rmse_kt"] - np.sqrt((4 + 4 + 36) / 3)) < 1e-9
    assert abs(m["bias_kt"] - (2 - 2 + 6) / 3) < 1e-9


def test_regression_metrics_validation():
    for bad in (([], []), ([1, 2], [1])):
        try:
            regression_metrics(*bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_negative_prediction_is_clipped_for_category():
    assert category_labels([-10], "imd") == ["Below depression"]


def test_per_category_only_lists_present_categories():
    out = per_category_metrics([20, 22, 100], [25, 20, 90], "imd")
    assert set(out) == {"Depression", "Extremely Severe Cyclonic Storm"}
    assert out["Depression"]["n"] == 2


def test_classification_metrics_perfect_prediction():
    c = classification_metrics([20, 40, 100], [21, 41, 101], "imd")
    assert c["accuracy"] == 1.0
    cm = np.array(c["confusion_matrix"])
    assert cm.shape == (8, 8) and cm.trace() == 3


def test_group_metrics():
    g = group_metrics([10, 10, 50], [12, 12, 40], ["A", "A", "B"])
    assert g["A"]["n"] == 2 and abs(g["B"]["bias_kt"] + 10) < 1e-9


def test_uncertainty_summary_keys():
    rng = np.random.default_rng(0)
    y = rng.uniform(20, 100, 50)
    s = uncertainty_summary(y, y + rng.normal(0, 5, 50), np.full(50, 5.0))
    assert 0 <= s["coverage_within_2std"] <= 1 and "spearman_std_vs_abs_error" in s
