import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Pushover Predictor (XGB)", page_icon="🧱", layout="wide")
ART_DIR = Path(__file__).resolve().parent

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    meta = joblib.load(ART_DIR / "meta.joblib")
    Xsc  = joblib.load(ART_DIR / "Xsc.pkl")
    Ysc  = joblib.load(ART_DIR / "Ysc.pkl")
    models = joblib.load(ART_DIR / "xgb_models.joblib")  # list of per-target models
    return meta, Xsc, Ysc, models

meta, Xsc, Ysc, models = load_artifacts()

# ----------------------------
# Resolve schema (supports both key styles)
# ----------------------------
def pick_first_key(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

CAND_X = ["X_columns", "FEATURES", "features", "feature_names", "X_cols", "input_columns"]
CAND_Y = ["Y_columns", "YVARS", "targets", "target_names", "Y_cols", "output_columns"]

FEATURES = pick_first_key(meta, CAND_X)
YVARS    = pick_first_key(meta, CAND_Y)

if FEATURES is None or YVARS is None:
    st.error("meta.joblib is missing FEATURES/YVARS (or X_columns/Y_columns).")
    st.write("Meta keys found:", list(meta.keys()))
    st.stop()

# Training flags (from your notebook cfg)
log_X = bool(meta.get("log_transform_X", True) or meta.get("cfg", {}).get("log_transform_X", True))
log_Y = bool(meta.get("log_transform_Y", True) or meta.get("cfg", {}).get("log_transform_Y", True))

# ----------------------------
# Preprocessing (MATCH TRAINING)
# ----------------------------
def fwd_X(df_or_np):
    X = df_or_np.values if hasattr(df_or_np, "values") else np.asarray(df_or_np)
    X = X.astype(np.float32, copy=False)
    if log_X:
        X = np.log1p(np.clip(X, a_min=0.0, a_max=None)).astype(np.float32)
    return Xsc.transform(X)

def inv_Y(Yz):
    Y = Ysc.inverse_transform(Yz)
    if log_Y:
        Y = np.expm1(Y).astype(np.float32)
    return Y

def predict_multioutput_xgb(models, X):
    outs = [m.predict(X).reshape(-1, 1) for m in models]
    return np.hstack(outs)

# ----------------------------
# UI labels (inputs)
# ----------------------------
FEATURE_UI = {
    "NS": {"label": "Number of stories", "unit": "stories"},
    "BW": {"label": "Bay width", "unit": "mm"},
    "BN": {"label": "Number of bays", "unit": "count"},
    "FM": {"label": "Infill strength", "unit": "MPa"},
    "TM": {"label": "Infill thickness", "unit": "mm"},
    "IP": {"label": "Infill percentage", "unit": "%"},
    "IP_GS": {"label": "Infill % at ground storey", "unit": "%"},
    "FCK": {"label": "Concrete strength (fck)", "unit": "MPa"},
    "AC": {"label": "Area of column", "unit": "mm^2"},
    "AB": {"label": "Area of beam", "unit": "mm^2"},
    "rhoC": {"label": "Longitudinal reinforcement ratio (column)", "unit": "-"},
    "rhoB": {"label": "Longitudinal reinforcement ratio (beam)", "unit": "-"},
}

def nice_label(key: str) -> str:
    ui = FEATURE_UI.get(key, {})
    label = ui.get("label", key)
    unit = (ui.get("unit", "") or "").strip()
    return f"{label} ({key}) [{unit}]" if unit else f"{label} ({key})"

# ----------------------------
# Output units (your given units)
# ----------------------------
OUTPUT_UI = {
    "F1":   {"unit": "kN"},
    "K1":   {"unit": "kN/m"},
    "F2":   {"unit": "kN"},
    "D2":   {"unit": "m"},
    "K23":  {"unit": "kN/m"},
    "Fres": {"unit": "kN"},
}
def nice_out_label(key: str) -> str:
    unit = (OUTPUT_UI.get(key, {}).get("unit", "") or "").strip()
    return f"{key} [{unit}]" if unit else key

# ----------------------------
# App Layout
# ----------------------------
st.title("Pushover Curve Predictor (XGB)")
st.caption("Enter inputs in the shown units. Preprocessing is handled automatically.")

st.sidebar.header("Model info")
st.sidebar.write(f"Inputs: {len(FEATURES)}")
st.sidebar.write(f"Outputs: {len(YVARS)}")
st.sidebar.write("Preprocessing: handled automatically")
st.sidebar.write(f"Model type: XGBoost (per-output models: {len(models)})")

st.subheader("Pushover curve prediction")
cols = st.columns(3)
inputs = []

for i, name in enumerate(FEATURES):
    with cols[i % 3]:
        label = nice_label(name)
        if name in ["NS", "BN"]:
            val = st.number_input(label, min_value=1, value=1, step=1)
        else:
            val = st.number_input(label, min_value=0.0, value=0.0, format="%.6f")
        inputs.append(val)

if st.button("Predict pushover curve parameters"):
    try:
        X_in = np.array(inputs, dtype=np.float32).reshape(1, -1)
        Xz = fwd_X(X_in)

        # sanity check (optional)
        zmax = float(np.max(np.abs(Xz)))
        if zmax > 6:
            st.warning(f"Inputs appear far from the training distribution (max |z| ≈ {zmax:.2f}). Check units/ranges.", icon="⚠️")

        Yz = predict_multioutput_xgb(models, Xz)
        Yo = inv_Y(Yz)

        df_raw = pd.DataFrame(Yo, columns=YVARS)
        df_show = df_raw.rename(columns={c: nice_out_label(c) for c in df_raw.columns})

        st.success("Prediction")
        st.dataframe(df_show.style.format("{:.6f}"))
        st.download_button("Download CSV", df_show.to_csv(index=False), "xgb_pred_single.csv", "text/csv")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

st.subheader("Batch prediction (CSV)")
up = st.file_uploader("Upload CSV with EXACT columns (same names & order as training features).", type=["csv"])
if up is not None:
    try:
        df_in = pd.read_csv(up)
        if list(df_in.columns) != list(FEATURES):
            st.error("CSV columns must match training feature names AND order exactly.")
            st.write("Expected:", FEATURES)
            st.write("Found:", list(df_in.columns))
        else:
            Xz = fwd_X(df_in)
            Yz = predict_multioutput_xgb(models, Xz)
            Yo = inv_Y(Yz)

            out_raw = pd.DataFrame(Yo, columns=YVARS)
            out_show = out_raw.rename(columns={c: nice_out_label(c) for c in out_raw.columns})

            st.success(f"Predicted {len(out_show)} rows.")
            st.dataframe(out_show.head().style.format("{:.6f}"))
            st.download_button("Download predictions", out_show.to_csv(index=False), "xgb_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Failed to score file: {e}")