"""Streamlit interface. Run with:  streamlit run app.py

Honesty rules built into this app:
  * A wind-speed prediction is shown ONLY when a trained checkpoint exists in artifacts/.
  * Checkpoints marked demo_run (software tests) get a red warning and are never called results.
  * Metrics are read from files written by src/evaluate.py; nothing is typed in by hand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_config
from src.preprocessing import ImageLoadError, load_image, preprocess_array
from src.predict import find_checkpoints, load_uploaded_image, validate_array

ART = Path("artifacts")
st.set_page_config(page_title="Cyclone Intensity Estimation", page_icon="🌀", layout="wide")

st.markdown("""<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.metric-note {color:#666; font-size:0.85rem;}
</style>""", unsafe_allow_html=True)

st.title("🌀 Tropical Cyclone Intensity Estimation from Satellite Imagery")
st.caption("Deep-learning college project · wind-speed regression (knots) from single-channel infrared images")

@st.cache_resource
def _load_model(path: str):
    from src.predict import load_checkpoint  # imports PyTorch only when a prediction is requested
    return load_checkpoint(path)


checkpoints = find_checkpoints(ART)
real_ckpts = [c for c in checkpoints if not c.name.startswith("demo_")]
if not checkpoints:
    st.warning("**No trained model found in `artifacts/`.** Predictions are disabled until a model is trained on a real "
               "dataset (see README). You can still explore the project and preview image preprocessing.")
elif not real_ckpts:
    st.error("**DEMO MODE:** only software-test checkpoints (`demo_*.pt`) exist. They were trained on artificial data. "
             "Their outputs are NOT cyclone estimates.")

tab_over, tab_method, tab_pred, tab_res, tab_lim = st.tabs(
    ["Overview", "Methodology", "Predict", "Results", "Limitations"])

with tab_over:
    st.header("What this project does")
    st.write("Tropical cyclone intensity (maximum sustained wind speed) is traditionally estimated by experts using "
             "the Dvorak technique, which is partly subjective. This project trains convolutional neural networks to "
             "estimate wind speed in **knots** directly from a geostationary infrared satellite image.")
    st.subheader("Objectives")
    st.markdown("""
1. Build a validated, leakage-safe data pipeline (storm-wise train / validation / test splits).
2. Train a CNN from scratch and a pretrained ResNet-18, and compare them with simple baselines.
3. Report MAE and RMSE in knots, error by intensity category, and a rough uncertainty estimate.
4. Provide this interface to preview and (with a trained model) run predictions.
""")
    st.info("**Scope:** intensity only. Cyclone *size* (RMW / R34) is **not** implemented: it needs labelled size data "
            "that this project does not use. Temporal (ConvLSTM) modelling is listed as future work.")

with tab_method:
    st.header("Literature-based methodology")
    st.write("The design is *inspired by* the team's literature survey. It is **not** a reproduction of any paper.")
    st.table(pd.DataFrame({
        "Paper (from survey)": ["Dharpure et al. 2023", "Jung et al. 2024", "Zhang et al. 2021 (TCICENet)",
                                "Baek et al. 2022 (TC-SEM)", "Ritchie et al. 2012 (DAV)", "Zhuo & Tan 2021 (DeepTCNet)",
                                "Manju et al. 2025", "Zhang et al. 2025"],
        "Idea": ["CNN on INSAT-3D IR", "Transfer learning", "IR-only CNN cascade", "Multi-task size CNN",
                 "Classical symmetry measure", "Physics-augmented CNN", "ConvLSTM on sequences",
                 "Wavelet + channel fusion"],
        "Used here?": ["CNN baseline (idea)", "Pretrained ResNet-18 (idea)", "IR-only input (idea)",
                       "No - size not implemented", "Reference only (not deep learning)", "No",
                       "Future work", "No"],
    }))
    st.markdown("""
**Pipeline:** image → resize → normalise → (training only: random rotation, brightness jitter) → CNN → standardised
wind → convert to knots → map to an intensity category.

**Addressing survey gaps (as evaluation goals, not claims of solving them):** leakage-safe storm-wise splits,
error reported per intensity category, optional cross-basin test, pretrained vs scratch comparison, and Monte-Carlo
dropout as a rough uncertainty indicator.
""")

with tab_pred:
    st.header("Predict")
    cfg_default = {"image_size": 224, "normalization": "fixed", "max_pixel_value": 255.0}
    source = st.radio("Image source", ["Upload an image", "Pick a real test-set image"], horizontal=True)
    arr, warns, truth = None, [], None
    try:
        if source == "Upload an image":
            up = st.file_uploader("Single-channel infrared satellite image (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if up is not None:
                arr, warns = load_uploaded_image(up)
        else:
            try:
                tdf = pd.read_csv(load_config()["data"]["processed_dir"] + "/test.csv")
            except Exception:
                tdf = None
            if tdf is None or tdf.empty:
                st.info("No test split found. Prepare a real dataset first (README, section 'Dataset').")
            else:
                idx = st.selectbox("Test image", range(len(tdf)),
                                   format_func=lambda i: f"{tdf.loc[i, 'storm_id']} | {tdf.loc[i, 'timestamp']}")
                arr = load_image(tdf.loc[idx, "abs_path"])
                warns = validate_array(arr)
                truth = float(tdf.loc[idx, "wind_speed_kt"])
    except ImageLoadError as err:
        st.error(f"Cannot use this image: {err}")
        arr = None

    if arr is not None:
        for w in warns:
            st.warning(w)
        pre = preprocess_array(arr, cfg_default["image_size"], cfg_default["normalization"],
                               cfg_default["max_pixel_value"])[0]
        c1, c2 = st.columns(2)
        c1.image(np.clip(arr / max(float(arr.max()), 1.0), 0, 1), caption=f"Input ({arr.shape[1]}×{arr.shape[0]} px)",
                 use_container_width=True)
        c2.image(pre, caption="After resizing and normalisation (what the model sees)", use_container_width=True)

        if not checkpoints:
            st.info("**Prediction disabled:** no trained checkpoint in `artifacts/`. Train a model on real data "
                    "(README, 'Train') and copy `best_*.pt` into `artifacts/`.")
        else:
            choice = st.selectbox("Model checkpoint", [c.name for c in checkpoints])
            mc = st.slider("Monte-Carlo dropout samples (0 = off)", 0, 50, 0)
            if st.button("Estimate intensity", type="primary"):
                try:
                    from src.predict import predict_array
                    lm = _load_model(str(ART / choice))
                    res = predict_array(lm, arr, mc)
                    if lm.meta.get("demo_run"):
                        st.error("DEMO CHECKPOINT - trained on artificial data. This number is NOT a cyclone estimate.")
                    st.metric("Estimated maximum sustained wind", f"{res['wind_kt']:.0f} kt",
                              help="1 kt ≈ 0.514 m/s")
                    st.write(f"≈ {res['wind_ms']:.0f} m/s · category ({res['scheme'].upper()}): **{res['category']}**")
                    if res["std_kt"] is not None:
                        st.write(f"Uncertainty indicator (MC-dropout std): ± {res['std_kt']:.1f} kt "
                                 "(rough guide, not a calibrated interval)")
                    if truth is not None:
                        st.write(f"Best-track wind for this test image: **{truth:.0f} kt** "
                                 f"(error {res['wind_kt'] - truth:+.0f} kt)")
                    st.caption("Expect larger errors for very intense cyclones and for images unlike the training data.")
                except ImportError:
                    st.error("PyTorch is not installed in this environment (pip install -r requirements.txt).")
                except Exception as err:  # show a readable message instead of a stack trace
                    st.error(f"Prediction failed: {err}")

with tab_res:
    st.header("Evaluation results")
    files = sorted(ART.glob("*_metrics.json")) if ART.is_dir() else []
    if not files:
        st.info("No evaluation files yet. Run `python -m src.evaluate --checkpoint artifacts/best_resnet18.pt` "
                "after training. Numbers shown here always come from those files.")
    for f in files:
        m = json.loads(f.read_text(encoding="utf-8"))
        st.subheader(f"{m['model_name']} · {m['split']} split · {m['overall']['n']} images, {m['n_storms']} storms")
        if m.get("demo_run"):
            st.error("DEMO RUN: software test on artificial data. Not a scientific result.")
        o = m["overall"]
        a, b, c = st.columns(3)
        a.metric("MAE", f"{o['mae_kt']:.2f} kt"); b.metric("RMSE", f"{o['rmse_kt']:.2f} kt")
        c.metric("Bias (pred − obs)", f"{o['bias_kt']:+.2f} kt")
        if "baselines" in m:
            st.write("Baselines on the same split:")
            st.dataframe(pd.DataFrame(m["baselines"]).T[["mae_kt", "rmse_kt"]])
        st.write("Error by true intensity category:")
        st.dataframe(pd.DataFrame(m["per_category"]).T[["n", "mae_kt", "rmse_kt", "bias_kt"]])
        cl = m["classification_from_regression"]
        st.write(f"Category accuracy (from regression output, {cl['scheme'].upper()}): {cl['accuracy']:.3f}")
        stem = f.name.replace("_metrics.json", "")
        imgs = [ART / f"{stem}_scatter.png", ART / f"{stem}_category_mae.png", ART / f"{stem}_confusion.png"]
        cols = st.columns(3)
        for col, p in zip(cols, imgs):
            if p.exists():
                col.image(str(p), use_container_width=True)
        if "uncertainty_mc_dropout" in m:
            st.json(m["uncertainty_mc_dropout"])
    for lc in sorted(ART.glob("*_loss_curve.png")) if ART.is_dir() else []:
        st.image(str(lc), caption=lc.name, width=450)

with tab_lim:
    st.header("Limitations")
    st.markdown("""
- Accuracy is only as good as the dataset: **no result is valid until trained and evaluated on real data**.
- Labels (best-track wind) are themselves estimates with noise.
- Very intense cyclones are rare, so models usually under-estimate them - see the per-category table.
- Uploaded web images are generally *not* comparable to the training data (different channel, scaling, resolution).
- One infrared channel, no storm history, no environmental inputs, no cyclone size estimation.
- The uncertainty indicator is rough and not calibrated.
""")
