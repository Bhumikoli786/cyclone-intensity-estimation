"""Evaluation metrics (NumPy / scikit-learn only). All wind speeds are in KNOTS."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data_loader import CATEGORY_SCHEMES, intensity_category


def _as_arrays(y_true, y_pred):
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D arrays of equal length, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("Cannot compute metrics on an empty array")
    return a, b


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and mean error (bias) in knots. Positive bias = model over-estimates."""
    a, b = _as_arrays(y_true, y_pred)
    err = b - a
    return {"n": int(a.size), "mae_kt": float(np.abs(err).mean()),
            "rmse_kt": float(np.sqrt((err ** 2).mean())), "bias_kt": float(err.mean())}


def category_labels(winds, scheme: str = "imd") -> list[str]:
    """Winds (kt) -> category names. Negative predictions are clipped to 0."""
    return [intensity_category(max(float(w), 0.0), scheme) for w in winds]


def per_category_metrics(y_true, y_pred, scheme: str = "imd") -> dict:
    """Regression error grouped by the TRUE category - shows weakness at extreme intensities."""
    a, b = _as_arrays(y_true, y_pred)
    cats = np.array(category_labels(a, scheme))
    out = {}
    for _, name in CATEGORY_SCHEMES[scheme]:
        m = cats == name
        if m.any():
            out[name] = regression_metrics(a[m], b[m])
    return out


def classification_metrics(y_true, y_pred, scheme: str = "imd") -> dict:
    """Category metrics obtained by binning the regression output (no separate classifier)."""
    a, b = _as_arrays(y_true, y_pred)
    labels = [name for _, name in CATEGORY_SCHEMES[scheme]]
    t, p = category_labels(a, scheme), category_labels(b, scheme)
    prec, rec, f1, sup = precision_recall_fscore_support(t, p, labels=labels, zero_division=0)
    present = sup > 0
    return {
        "scheme": scheme, "labels": labels,
        "accuracy": float(np.mean(np.array(t) == np.array(p))),
        "macro_f1_present_classes": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {l: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i, l in enumerate(labels)},
        "confusion_matrix": confusion_matrix(t, p, labels=labels).tolist(),
    }


def group_metrics(y_true, y_pred, groups) -> dict:
    """Regression metrics per group value (e.g. per basin)."""
    a, b = _as_arrays(y_true, y_pred)
    g = np.asarray(groups).astype(str)
    return {k: regression_metrics(a[g == k], b[g == k]) for k in sorted(set(g))}


def uncertainty_summary(y_true, y_pred, y_std) -> dict:
    """Does the predicted spread track the real error? (Spearman rank correlation)"""
    from scipy.stats import spearmanr
    a, b = _as_arrays(y_true, y_pred)
    s = np.asarray(y_std, dtype=float)
    if s.shape != a.shape:
        raise ValueError("y_std must have the same length as y_true")
    err = np.abs(b - a)
    rho = float(spearmanr(s, err).statistic) if s.std() > 0 and err.std() > 0 else float("nan")
    return {"mean_predicted_std_kt": float(s.mean()), "spearman_std_vs_abs_error": rho,
            "coverage_within_2std": float(np.mean(err <= 2 * s))}
