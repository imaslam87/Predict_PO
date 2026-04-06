import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import BytesIO

# ============================================================
# GLOBAL FONT SETTINGS
# ============================================================
APP_FONT_FAMILY = "Times New Roman"
UI_FONT_SIZE_PX = 15
PLOT_FONT_SIZE_PT = 11
PLOT_TICK_FONT = 8
PLOT_LEGEND_FONT = 8

# ============================================================
# COMPACT CSS (inputs)
# ============================================================
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: "{APP_FONT_FAMILY}", Times, serif !important;
        font-size: {UI_FONT_SIZE_PX}px !important;
    }}
    div[data-baseweb="select"] > div {{
        min-height: 30px !important;
        padding: 0px 6px !important;
        border-radius: 6px !important;
        max-width: 190px !important;
    }}
    div[data-testid="stTextInput"] input {{
        min-height: 30px !important;
        padding: 0px 6px !important;
        border-radius: 6px !important;
        max-width: 190px !important;
    }}
    div[data-testid="stSelectbox"], div[data-testid="stTextInput"] {{
        max-width: 195px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "font.family": APP_FONT_FAMILY,
        "font.size": PLOT_FONT_SIZE_PT,
        "axes.labelsize": PLOT_FONT_SIZE_PT,
        "xtick.labelsize": PLOT_TICK_FONT,
        "ytick.labelsize": PLOT_TICK_FONT,
        "legend.fontsize": PLOT_LEGEND_FONT,
    }
)

st.set_page_config(page_title="Pushover Predictor (XGB)", page_icon="🧱", layout="wide")
ART_DIR = Path(__file__).resolve().parent

# ============================================================
# IMPORTANT MODEL/UNIT ASSUMPTIONS (per your retraining)
# ============================================================
# Your updated model predicts D2 in METRES (because you trained with D2/1000 -> metres)
D2_IS_METRES = True

# You asked to correct units of K1 and K23 to kN/mm WITHOUT changing values.
# So for curve construction we treat K1,K23 numerically as kN/mm.
K_UNITS_ARE_KN_PER_MM = True

# ============================================================
# Helpers (THIS FIXES YOUR NameError)
# ============================================================
def pick_first_key(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def safe_float_from_text(s: str, default: float = 0.0) -> float:
    try:
        if s is None:
            return default
        s = str(s).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def artifact_signature(files):
    """
    Creates a signature based on file modified times + file sizes.
    Any file replacement forces Streamlit to reload cached artifacts.
    """
    sig_parts = []
    for f in files:
        p = ART_DIR / f
        try:
            sig_parts.append(
                f"{f}:mtime={int(os.path.getmtime(p))}:size={os.path.getsize(p)}"
            )
        except Exception:
            sig_parts.append(f"{f}:missing")
    return "|".join(sig_parts)

# ============================================================
# Load artifacts with auto-reload signature
# ============================================================
ARTIFACT_FILES = ["meta.joblib", "Xsc.pkl", "Ysc.pkl", "xgb_models.joblib"]
ART_SIG = artifact_signature(ARTIFACT_FILES)

@st.cache_resource
def load_artifacts(_sig: str):
    meta = joblib.load(ART_DIR / "meta.joblib")
    Xsc  = joblib.load(ART_DIR / "Xsc.pkl")
    Ysc  = joblib.load(ART_DIR / "Ysc.pkl")
    models = joblib.load(ART_DIR / "xgb_models.joblib")
    return meta, Xsc, Ysc, models

try:
    meta, Xsc, Ysc, models = load_artifacts(ART_SIG)
except Exception as e:
    st.error(f"Artifact load failed: {e}")
    st.stop()

# ============================================================
# Schema + training flags
# ============================================================
FEATURES = meta.get("FEATURES", None)
YVARS    = meta.get("YVARS", None)
cfg      = meta.get("cfg", {}) if isinstance(meta, dict) else {}

if FEATURES is None or YVARS is None:
    st.error("meta.joblib does not contain FEATURES and YVARS.")
    st.write("Meta keys:", list(meta.keys()) if isinstance(meta, dict) else type(meta))
    st.stop()

log_X = bool(cfg.get("log_transform_X", True))
log_Y = bool(cfg.get("log_transform_Y", True))

# ============================================================
# Preprocessing
# ============================================================
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

def predict_one(row_dict):
    X1 = np.array([row_dict[c] for c in FEATURES], dtype=np.float32).reshape(1, -1)
    Xz1 = fwd_X(X1)
    Yz1 = predict_multioutput_xgb(models, Xz1)
    Yo1 = inv_Y(Yz1)[0]
    return {YVARS[i]: float(Yo1[i]) for i in range(len(YVARS))}

# ============================================================
# UI labels
# ============================================================
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
    "rhoC": {"label": "Reinf. ratio (column)", "unit": "-"},
    "rhoB": {"label": "Reinf. ratio (beam)", "unit": "-"},
}

def nice_label(key: str) -> str:
    ui = FEATURE_UI.get(key, {})
    label = ui.get("label", key)
    unit = (ui.get("unit", "") or "").strip()
    return f"{label} ({key}) [{unit}]" if unit else f"{label} ({key})"

# Dropdown constraints
NS_OPTIONS = list(range(1, 13))
BW_OPTIONS = [4000, 6000]
BN_OPTIONS = list(range(1, 9))
FM_OPTIONS = [3, 6, 9, 15]
TM_OPTIONS = [115, 230]
IP_OPTIONS = list(range(1, 101))
IPGS_OPTIONS = list(range(1, 101))
FCK_OPTIONS = [30, 40]

def axis_end_mm_from_NS(ns: float) -> float:
    ns_i = int(round(float(ns)))
    fixed = {2: 150.0, 4: 300.0, 8: 600.0, 12: 900.0}
    return fixed.get(ns_i, 75.0 * float(ns_i))

def scenario_row(base_row, kind: str):
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
        r["IP"] = 100.0
        r["IP_GS"] = 100.0
        return r
    if kind == "Soft story":
        r["IP_GS"] = 0.0
        return r
    return r

# ============================================================
# Curve construction
# D2 from model is metres -> convert to mm for plotting
# K1, K23 treated as kN/mm (labels corrected, values unchanged)
# ============================================================
def curve_key_points_mm(pred):
    F1 = float(pred["F1"])
    K1 = float(pred["K1"])
    F2 = float(pred["F2"])
    K23 = float(pred["K23"])
    Fres = float(pred["Fres"])

    D2_m = float(pred["D2"]) if D2_IS_METRES else float(pred["D2"]) / 1000.0
    D2_mm = 1000.0 * D2_m

    if abs(K1) < 1e-12 or abs(K23) < 1e-12:
        return None, "K1 or K23 too close to zero."

    # Treat K as kN/mm => displacements are in mm directly
    D1_mm = F1 / K1
    D3_mm = D2_mm + (F2 - Fres) / K23

    x = [0.0, D1_mm, D2_mm, D3_mm]
    y = [0.0, F1, F2, Fres]
    pts = sorted(zip(x, y), key=lambda t: t[0])
    x_mm = [p[0] for p in pts]
    y_kN = [p[1] for p in pts]
    return (x_mm, y_kN), None

def extend_to_axis_end(x_mm, y_kN, axis_end_mm):
    last_y = float(y_kN[-1])
    pts = [(xx, yy) for xx, yy in zip(x_mm, y_kN) if xx <= axis_end_mm]
    if not pts:
        pts = [(0.0, 0.0)]
    x2 = [p[0] for p in pts]
    y2 = [p[1] for p in pts]
    if x2[-1] < axis_end_mm:
        x2.append(axis_end_mm)
        y2.append(last_y)
    return x2, y2

# ============================================================
# PAGE
# ============================================================
st.title("Pushover Curve Predictor (XGB)")
st.caption("App updated. Fixes NameError and ensures new artifacts reload automatically.")

left, right = st.columns([0.38, 0.62], gap="small")

if "inputs_dict" not in st.session_state:
    st.session_state["inputs_dict"] = {}

with left:
    st.subheader("Inputs")
    pad_l, form_col, pad_r = st.columns([0.10, 0.80, 0.10])

    with form_col:
        inputs = {}
        for name in FEATURES:
            label = nice_label(name)
            if name == "NS":
                inputs[name] = float(st.selectbox(label, NS_OPTIONS, index=1 if 2 in NS_OPTIONS else 0))
            elif name == "BW":
                inputs[name] = float(st.selectbox(label, BW_OPTIONS, index=0))
            elif name == "BN":
                inputs[name] = float(st.selectbox(label, BN_OPTIONS, index=0))
            elif name == "FM":
                inputs[name] = float(st.selectbox(label, FM_OPTIONS, index=0))
            elif name == "TM":
                inputs[name] = float(st.selectbox(label, TM_OPTIONS, index=0))
            elif name == "IP":
                inputs[name] = float(st.selectbox(label, IP_OPTIONS, index=49))
            elif name == "IP_GS":
                inputs[name] = float(st.selectbox(label, IPGS_OPTIONS, index=49))
            elif name == "FCK":
                inputs[name] = float(st.selectbox(label, FCK_OPTIONS, index=0))
            elif name in ["AC", "AB", "rhoC", "rhoB"]:
                txt = st.text_input(label, value="0.00")
                inputs[name] = safe_float_from_text(txt, default=0.0)
            else:
                txt = st.text_input(label, value="0.00")
                inputs[name] = safe_float_from_text(txt, default=0.0)

        st.session_state["inputs_dict"] = inputs

with right:
    st.subheader("Outputs")
    run = st.button("Predict pushover curve parameters")

    if "run_state" not in st.session_state:
        st.session_state["run_state"] = False
    if run:
        st.session_state["run_state"] = True

    if st.session_state["run_state"]:
        base_row = dict(st.session_state["inputs_dict"])
        ns_base = base_row.get("NS", 2.0)
        axis_end = axis_end_mm_from_NS(ns_base)

        pred_input = predict_one(base_row)
        D2_m = float(pred_input["D2"])
        D2_mm = 1000.0 * D2_m

        row = {
            "F1 [kN]": pred_input.get("F1", np.nan),
            "K1 [kN/mm]": pred_input.get("K1", np.nan),
            "F2 [kN]": pred_input.get("F2", np.nan),
            "D2 [m]": D2_m,
            "D2 [mm]": D2_mm,
            "K23 [kN/mm]": pred_input.get("K23", np.nan),
            "Fres [kN]": pred_input.get("Fres", np.nan),
        }
        st.markdown("**Predicted parameters (Input case)**")
        st.dataframe(pd.DataFrame([row]).round(6), use_container_width=True, hide_index=True)

        scenarios = [
            ("Input", "blue", "-"),
            ("Bare frame", "black", "--"),
            ("Fully Infilled", "green", ":"),
            ("Soft story", "red", "-."),
        ]

        curves = []
        err_msgs = []

        for name, color, ls in scenarios:
            row_dict = scenario_row(base_row, name)
            pred = predict_one(row_dict)

            pts, err = curve_key_points_mm(pred)
            if err:
                err_msgs.append(f"{name}: {err}")
                continue

            x_mm, y_kN = pts
            x2, y2 = extend_to_axis_end(list(x_mm), list(y_kN), axis_end)
            curves.append((name, color, ls, x2, y2))

        if err_msgs:
            st.warning("Some scenarios could not be plotted:\n- " + "\n- ".join(err_msgs), icon="⚠️")

        st.subheader("F–D plot (scenarios)")
        fig, ax = plt.subplots(figsize=(2.6, 2.1), dpi=170)

        xs = np.linspace(0.0, axis_end, 450)
        for name, color, ls, x_mm, y_kN in curves:
            ys = np.interp(xs, x_mm, y_kN)
            ax.plot(xs, ys, linewidth=0.9, color=color, linestyle=ls, label=name)

        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Base Shear (kN)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, axis_end)

        if curves:
            all_y = np.concatenate([np.array(c[4], dtype=float) for c in curves])
            y_max = float(np.max(all_y)) * 1.05 if float(np.max(all_y)) > 0 else 1.0
        else:
            y_max = 1.0
        ax.set_ylim(0.0, y_max)

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT)
        ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=1.2,
            handletextpad=0.6,
            borderaxespad=0.3,
            fontsize=PLOT_LEGEND_FONT,
        )

        fig.tight_layout(pad=0.5)
        st.pyplot(fig, clear_figure=True, use_container_width=False)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            "Download F–D plot (PNG)",
            data=buf,
            file_name="pushover_curve_fd_scenarios.png",
            mime="image/png",
        )