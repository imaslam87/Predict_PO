import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import BytesIO

# ============================================================
# GLOBAL FONT SETTINGS (EDIT ONLY HERE)
# ============================================================
APP_FONT_FAMILY = "Times New Roman"

UI_FONT_SIZE_PX = 16    # Streamlit UI: input/output labels, values, tables, headings
PLOT_FONT_SIZE_PT = 8  # Matplotlib plot: labels, ticks, legend (points)

# Apply to Streamlit UI (inputs, outputs, tables, text)
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: "{APP_FONT_FAMILY}", Times, serif !important;
        font-size: {UI_FONT_SIZE_PX}px !important;
    }}
    div[data-testid="stDataFrame"] * {{
        font-family: "{APP_FONT_FAMILY}", Times, serif !important;
        font-size: {UI_FONT_SIZE_PX}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Apply to Matplotlib plots
plt.rcParams.update(
    {
        "font.family": APP_FONT_FAMILY,
        "font.size": PLOT_FONT_SIZE_PT,
        "axes.labelsize": PLOT_FONT_SIZE_PT,
        "xtick.labelsize": PLOT_FONT_SIZE_PT,
        "ytick.labelsize": PLOT_FONT_SIZE_PT,
        "legend.fontsize": PLOT_FONT_SIZE_PT,
    }
)

# ============================================================
# STREAMLIT CONFIG
# ============================================================
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
        base_row = {FEATURES[i]: float(inputs[i]) for i in range(len(FEATURES))}

        def scenario_row(kind: str):
            r = dict(base_row)
            infill_keys = ["FM", "TM", "IP", "IP_GS"]

            if kind == "Input":
                return r

            if kind == "Bare frame":
                for k in infill_keys:
                    if k in r:
                        r[k] = 0.0
                return r

            if kind == "Fully Infilled":
                if "IP" in r:
                    r["IP"] = 100.0
                if "IP_GS" in r:
                    r["IP_GS"] = 100.0
                return r

            if kind == "Soft story":
                if "IP_GS" in r:
                    r["IP_GS"] = 0.0
                return r

            return r

        def predict_one(row_dict):
            X1 = np.array([row_dict[c] for c in FEATURES], dtype=np.float32).reshape(1, -1)
            Xz1 = fwd_X(X1)
            Yz1 = predict_multioutput_xgb(models, Xz1)
            Yo1 = inv_Y(Yz1)
            return pd.DataFrame(Yo1, columns=YVARS).iloc[0]

        def fd_points(pred_row):
            F1 = float(pred_row["F1"])
            K1 = float(pred_row["K1"])
            F2 = float(pred_row["F2"])
            D2 = float(pred_row["D2"])  # m
            K23 = float(pred_row["K23"])
            Fres = float(pred_row["Fres"])

            if abs(K1) < 1e-12 or abs(K23) < 1e-12:
                return None, "K1 or K23 too close to zero."

            D1 = F1 / K1
            D3 = D2 + (F2 - Fres) / K23

            x_m = [0.0, D1, D2, D3]
            y_kN = [0.0, F1, F2, Fres]
            pts = sorted(zip(x_m, y_kN), key=lambda t: t[0])
            x_m = [p[0] for p in pts]
            y_kN = [p[1] for p in pts]
            return (x_m, y_kN), None

        # Scenario styles: (legend_name, color, linestyle)
        scenarios = [
            ("Input", "blue", "-"),             # solid
            ("Bare frame", "black", "--"),      # dashed
            ("Fully Infilled", "green", ":"),   # dotted
            ("Soft story", "red", "-."),        # dash-dot
        ]

        curves = []
        pred_table = []
        err_msgs = []

        for name, color, ls in scenarios:
            row_dict = scenario_row(name)
            pred = predict_one(row_dict)

            pred_table.append({**pred.to_dict(), "Scenario": name})

            pts, err = fd_points(pred)
            if err:
                err_msgs.append(f"{name}: {err}")
                continue

            curves.append((name, color, ls, pts[0], pts[1]))

        if err_msgs:
            st.warning("Some scenarios could not be plotted:\n- " + "\n- ".join(err_msgs), icon="⚠️")

        # Scenario comparison table
        df_cmp = pd.DataFrame(pred_table)
        cols_cmp = ["Scenario"] + [c for c in YVARS if c in df_cmp.columns]
        df_cmp = df_cmp[cols_cmp]
        df_cmp_show = df_cmp.rename(columns={c: nice_out_label(c) for c in YVARS if c in df_cmp.columns})

        st.subheader("Scenario comparison (predicted parameters)")
        num_cols = [c for c in df_cmp_show.columns if c != "Scenario"]
        st.dataframe(df_cmp_show.style.format("{:.6f}", subset=num_cols))

        # Plot all curves together
        st.subheader("Pushover curve (predicted F–D) — scenarios")

        fig, ax = plt.subplots(figsize=(3.2, 2.5), dpi=170)

        for name, color, ls, x_m, y_kN in curves:
            xs = np.linspace(min(x_m), max(x_m), 250)
            ys = np.interp(xs, x_m, y_kN)
            ax.plot(xs, ys, linewidth=0.9, color=color, linestyle=ls, label=name)

        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Base Shear (kN)")
        ax.grid(True, alpha=0.3)

        if curves:
            all_x = np.concatenate([np.array(c[3], dtype=float) for c in curves])
            all_y = np.concatenate([np.array(c[4], dtype=float) for c in curves])
            raw_x_lim = float(np.max(all_x)) * 1.05 if float(np.max(all_x)) > 0 else 1.0
            raw_y_lim = float(np.max(all_y)) * 1.05 if float(np.max(all_y)) > 0 else 1.0
        else:
            raw_x_lim, raw_y_lim = 1.0, 1.0

        ax.set_xlim(0.0, raw_x_lim)
        ax.set_ylim(0.0, raw_y_lim)

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        fig.canvas.draw()
        xt = ax.get_xticks()
        yt = ax.get_yticks()
        ax.set_xlim(0.0, float(xt[-1]))
        ax.set_ylim(0.0, float(yt[-1]))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        # Legend: upper-right + shorter line samples
        ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=1.4,
            handletextpad=0.6,
            borderaxespad=0.4,
        )

        fig.tight_layout(pad=0.6)

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
            file_name="pushover_curve_fd_scenarios.png",
            mime="image/png",
        )

        # Export the scenario comparison table
        st.download_button(
            "Download scenario predictions (CSV)",
            df_cmp_show.to_csv(index=False),
            "xgb_pred_scenarios.csv",
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