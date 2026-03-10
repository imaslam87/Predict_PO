import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import BytesIO

st.set_page_config(page_title="Pushover Predictor (XGB)", page_icon="🧱", layout="wide")
ART_DIR = Path(__file__).resolve().parent


# ----------------------------
# Helpers
# ----------------------------
def pick_first_key(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    meta = joblib.load(ART_DIR / "meta.joblib")
    Xsc = joblib.load(ART_DIR / "Xsc.pkl")
    Ysc = joblib.load(ART_DIR / "Ysc.pkl")
    models = joblib.load(ART_DIR / "xgb_models.joblib")  # list of per-output models
    return meta, Xsc, Ysc, models


try:
    meta, Xsc, Ysc, models = load_artifacts()
except Exception as e:
    st.error(f"Artifact load failed: {e}")
    st.stop()


# ----------------------------
# Resolve schema
# ----------------------------
CAND_X = ["X_columns", "FEATURES", "features", "feature_names", "X_cols", "input_columns"]
CAND_Y = ["Y_columns", "YVARS", "targets", "target_names", "Y_cols", "output_columns"]

FEATURES = pick_first_key(meta, CAND_X)
YVARS = pick_first_key(meta, CAND_Y)

if FEATURES is None or YVARS is None:
    st.error("meta.joblib does not contain feature/target names in a recognized format.")
    st.write("Meta keys found:", list(meta.keys()))
    st.stop()

# Training flags (internal)
cfg = meta.get("cfg", {}) if isinstance(meta, dict) else {}
log_X = bool(meta.get("log_transform_X", cfg.get("log_transform_X", True)))
log_Y = bool(meta.get("log_transform_Y", cfg.get("log_transform_Y", True)))


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
# UI labels + units (inputs)
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
# UI labels + units (outputs)
# ----------------------------
OUTPUT_UI = {
    "F1": {"unit": "kN"},
    "K1": {"unit": "kN/m"},
    "F2": {"unit": "kN"},
    "D2": {"unit": "m"},
    "K23": {"unit": "kN/m"},
    "Fres": {"unit": "kN"},
}


def nice_out_label(key: str) -> str:
    unit = (OUTPUT_UI.get(key, {}).get("unit", "") or "").strip()
    return f"{key} [{unit}]" if unit else key


# ----------------------------
# App layout
# ----------------------------
st.title("Pushover Curve Predictor (XGB)")
st.caption("Enter inputs in the shown units. Preprocessing is handled automatically.")

st.sidebar.header("Model info")
st.sidebar.write(f"Inputs: {len(FEATURES)}")
st.sidebar.write(f"Outputs: {len(YVARS)}")
st.sidebar.write("Preprocessing: handled automatically")
st.sidebar.write(f"Model type: XGBoost (per-output models: {len(models)})")


# ----------------------------
# Single case prediction
# ----------------------------
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

        zmax = float(np.max(np.abs(Xz)))
        if zmax > 6:
            st.warning(
                f"Inputs appear far from the training distribution (max |z| ≈ {zmax:.2f}). Check units/ranges.",
                icon="⚠️",
            )

        Yz = predict_multioutput_xgb(models, Xz)
        Yo = inv_Y(Yz)

        df_raw = pd.DataFrame(Yo, columns=YVARS)
        df_show = df_raw.rename(columns={c: nice_out_label(c) for c in df_raw.columns})

        st.success("Prediction")
        st.dataframe(df_show.style.format("{:.6f}"))

        # ---- Force–Displacement plot (compact + rounded ticks + end ticks) ----
        try:
            r = df_raw.iloc[0]
            F1 = float(r["F1"])
            K1 = float(r["K1"])
            F2 = float(r["F2"])
            D2 = float(r["D2"])  # metres
            K23 = float(r["K23"])
            Fres = float(r["Fres"])

            if abs(K1) < 1e-12 or abs(K23) < 1e-12:
                st.warning("Cannot plot F–D curve because K1 or K23 is too close to zero.", icon="⚠️")
            else:
                D1 = F1 / K1
                D3 = D2 + (F2 - Fres) / K23

                x_m = [0.0, D1, D2, D3]
                y_kN = [0.0, F1, F2, Fres]

                plt.rcParams["font.family"] = "Times New Roman"

                # Plot size (edit these if you want)
                fig, ax = plt.subplots(figsize=(3.2, 2), dpi=160)
                ax.plot(x_m, y_kN, marker="o", linewidth=2.0, color="blue")

                ax.set_xlabel("Displacement (m)", fontname="Times New Roman")
                ax.set_ylabel("Base Shear (kN)", fontname="Times New Roman")
                ax.grid(True, alpha=0.3)

                # Axis limits with padding
                x_max = max(x_m) if len(x_m) else 1.0
                y_max = max(y_kN) if len(y_kN) else 1.0
                raw_x_lim = x_max * 1.05 if x_max > 0 else 1.0
                raw_y_lim = y_max * 1.05 if y_max > 0 else 1.0

                # Rounded (integer-like) ticks + ensure last tick reaches axis end
                ax.set_xlim(0.0, raw_x_lim)
                ax.set_ylim(0.0, raw_y_lim)

                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))

                # Force tick computation, then lock the end of axis to the last tick
                fig.canvas.draw()
                xt = ax.get_xticks()
                yt = ax.get_yticks()
                ax.set_xlim(0.0, float(xt[-1]))
                ax.set_ylim(0.0, float(yt[-1]))

                # No decimals (rounded labels)
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

                fig.tight_layout(pad=0.6)

                st.subheader("Pushover curve (predicted F–D)")
                try:
                    st.pyplot(fig, clear_figure=True, use_container_width=False)
                except TypeError:
                    st.pyplot(fig, clear_figure=True)

                # Download plot
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    "Download F–D plot (PNG)",
                    data=buf,
                    file_name="pushover_curve_fd.png",
                    mime="image/png",
                )
        except Exception as e:
            st.warning(f"Could not generate F–D plot: {e}")

        st.download_button(
            "Download CSV",
            df_show.to_csv(index=False),
            "xgb_pred_single.csv",
            "text/csv",
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.markdown("---")


# ----------------------------
# Batch prediction
# ----------------------------
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

            zmax = float(np.max(np.abs(Xz)))
            if zmax > 6:
                st.warning(
                    f"Some rows appear far from the training distribution (max |z| ≈ {zmax:.2f}). Check units/ranges.",
                    icon="⚠️",
                )

            Yz = predict_multioutput_xgb(models, Xz)
            Yo = inv_Y(Yz)

            out_raw = pd.DataFrame(Yo, columns=YVARS)
            out_show = out_raw.rename(columns={c: nice_out_label(c) for c in out_raw.columns})

            st.success(f"Predicted {len(out_show)} rows.")
            st.dataframe(out_show.head().style.format("{:.6f}"))

            st.download_button(
                "Download predictions",
                out_show.to_csv(index=False),
                file_name="xgb_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Failed to score file: {e}")