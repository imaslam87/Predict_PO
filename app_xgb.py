import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt
from io import BytesIO

import matplotlib as mpl
from matplotlib import font_manager

# ============================================================
# FONT CONTROL PANEL (EDIT THESE)
# ============================================================
FONT_FAMILY = "Times New Roman"

TITLE_FONT_PX          = 22   # main title text
SECTION_HEADER_PX      = 16   # "Inputs" / section headings

INPUT_LABEL_FONT_PX    = 14   # captions like "Bay width (bw) [mm]"
INPUT_WIDGET_FONT_PX   = 14   # dropdown selected value + dropdown list + text boxes
BUTTON_FONT_PX         = 14   # Predict + Download buttons

# Plot fonts (matplotlib)
PLOT_AXIS_LABEL_PT     = 10   # x/y label font
PLOT_TICK_FONT_PT      = 10   # tick labels
PLOT_LEGEND_FONT_PT    = 8    # legend text

# Banner image size
BANNER_USE_CONTAINER_WIDTH = True
BANNER_WIDTH_PX = 560  # used only if BANNER_USE_CONTAINER_WIDTH=False

# Banner caption font size
BANNER_CAPTION_PX = 12

# ============================================================
# MODEL/UNITS
# ============================================================
D2_MODEL_IS_MM = True
K_LABEL = "kN/mm"  # labels only (no numeric scaling)

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Pushover Predictor (XGB)", page_icon="🧱", layout="wide")
ART_DIR = Path(__file__).resolve().parent

# ============================================================
# FORCE EXACT TIMES NEW ROMAN FOR MATPLOTLIB USING YOUR FONT FILES
# Put these in repo:
#   assets/fonts/times.ttf
#   assets/fonts/timesbd.ttf
#   assets/fonts/timesi.ttf
#   assets/fonts/timesbi.ttf
# ============================================================
def force_times_from_repo():
    font_dir = ART_DIR / "assets" / "fonts"
    candidates = [
        font_dir / "times.ttf",    # regular
        font_dir / "timesbd.ttf",  # bold
        font_dir / "timesi.ttf",   # italic
        font_dir / "timesbi.ttf",  # bold italic
    ]
    existing = [p for p in candidates if p.exists()]

    if not existing:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["Liberation Serif", "DejaVu Serif"]
        return "serif"

    for fp in existing:
        font_manager.fontManager.addfont(str(fp))

    base_fp = existing[0]
    font_name = font_manager.FontProperties(fname=str(base_fp)).get_name()
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["font.serif"] = [font_name]
    return font_name

PLOT_FONT_NAME = force_times_from_repo()

mpl.rcParams["axes.labelsize"] = PLOT_AXIS_LABEL_PT
mpl.rcParams["xtick.labelsize"] = PLOT_TICK_FONT_PT
mpl.rcParams["ytick.labelsize"] = PLOT_TICK_FONT_PT
mpl.rcParams["legend.fontsize"] = PLOT_LEGEND_FONT_PT

# ============================================================
# INPUT LABEL DEFINITIONS (HTML)
# ============================================================
FEATURE_UI = {
    "NS": {"label": "Number of stories", "symbol_html": "<i>NS</i>"},
    "BW": {"label": "Bay width", "symbol_html": "<i>b</i><sub>w</sub>", "unit_html": "mm"},
    "BN": {"label": "Number of bays", "symbol_html": "<i>BN</i>", "unit_html": "count"},
    "FM": {"label": "Infill strength", "symbol_html": "<i>f</i><sup>&prime;</sup><sub>m</sub>", "unit_html": "MPa"},
    "TM": {"label": "Infill thickness", "symbol_html": "<i>t</i><sub>m</sub>", "unit_html": "mm"},
    "IP": {"label": "Infill percentage", "symbol_html": "<i>IP</i>", "unit_html": "%"},
    "IP_GS": {"label": "Infill % at ground storey", "symbol_html": "<i>IP</i><sub>GS</sub>", "unit_html": "%"},
    "FCK": {"label": "Concrete strength", "symbol_html": "<i>f</i><sub>ck</sub>", "unit_html": "MPa"},
    "AC": {"label": "Area of column", "symbol_html": "<i>A</i><sub>c</sub>", "unit_html": "mm<sup>2</sup>"},
    "AB": {"label": "Area of beam", "symbol_html": "<i>A</i><sub>b</sub>", "unit_html": "mm<sup>2</sup>"},
    "rhoC": {"label": "Reinf. ratio (column)", "symbol_html": "&rho;<sub>c</sub>", "unit_html": "-"},
    "rhoB": {"label": "Reinf. ratio (beam)", "symbol_html": "&rho;<sub>b</sub>", "unit_html": "-"},
}

OUTPUT_HEADER_TXT = {
    "Scenario": "Scenario",
    "F1": "F1 (kN)",
    "K1": f"K1 ({K_LABEL})",
    "F2": "F2 (kN)",
    "D2_mm": "D2 (mm)",
    "K23": f"K23 ({K_LABEL})",
    "Fres": "Fres (kN)",
    "EndDisp_mm": "Disp_end (mm)",
}

# ============================================================
# CSS (FORCE TIMES + SIZE CONTROL FOR EVERY UI ELEMENT)
# ============================================================
st.markdown(
f"""
<style>
html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
}}

div[data-testid="stAppViewContainer"] h1,
div[data-testid="stAppViewContainer"] h2,
div[data-testid="stAppViewContainer"] h3,
div[data-testid="stAppViewContainer"] h4,
div[data-testid="stAppViewContainer"] h5,
div[data-testid="stAppViewContainer"] h6,
div[data-testid="stAppViewContainer"] h1 *,
div[data-testid="stAppViewContainer"] h2 *,
div[data-testid="stAppViewContainer"] h3 *,
div[data-testid="stAppViewContainer"] h4 *,
div[data-testid="stAppViewContainer"] h5 *,
div[data-testid="stAppViewContainer"] h6 * {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  color: #000000 !important;
}}

.app-title {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  font-size: {TITLE_FONT_PX}px !important;
  font-weight: 800 !important;
  margin-bottom: 6px !important;
  color: #000000 !important;
  line-height: 1.15 !important;
}}

.sec-h {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  font-size: {SECTION_HEADER_PX}px !important;
  font-weight: 700 !important;
  margin: 6px 0 10px 0 !important;
  color: #000000 !important;
  line-height: 1.15 !important;
}}

.input-label {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  font-size: {INPUT_LABEL_FONT_PX}px !important;
  color: #000000 !important;
  margin-bottom: 0.12rem !important;
}}

div[data-testid="stSelectbox"] * {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  color: #000000 !important;
  font-size: {INPUT_WIDGET_FONT_PX}px !important;
}}

div[data-baseweb="popover"] * {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  color: #000000 !important;
  font-size: {INPUT_WIDGET_FONT_PX}px !important;
}}

div[data-testid="stTextInput"] input {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  color: #000000 !important;
  font-size: {INPUT_WIDGET_FONT_PX}px !important;
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}}

button, button * {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  color: #000000 !important;
  font-size: {BUTTON_FONT_PX}px !important;
}}

a, a * {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  color: #000000 !important;
  font-size: {BUTTON_FONT_PX}px !important;
}}

div[data-testid="stSelectbox"], div[data-testid="stTextInput"] {{
  max-width: 170px !important;
}}

/* Banner captions */
.banner-cap {{
  font-family: "{FONT_FAMILY}", Times, serif !important;
  font-style: italic !important;
  font-size: {BANNER_CAPTION_PX}px !important;
  text-align: center !important;
  color: #000 !important;
  line-height: 1.05 !important;
  margin-top: -6px !important;
}}
</style>
""",
unsafe_allow_html=True
)

# ============================================================
# Helpers
# ============================================================
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
    sig_parts = []
    for f in files:
        p = ART_DIR / f
        try:
            sig_parts.append(f"{f}:mtime={int(os.path.getmtime(p))}:size={os.path.getsize(p)}")
        except Exception:
            sig_parts.append(f"{f}:missing")
    return "|".join(sig_parts)

def axis_end_mm_from_NS(ns: float) -> float:
    ns_i = int(round(float(ns)))
    fixed = {2: 150.0, 4: 300.0, 8: 600.0, 12: 900.0}
    return fixed.get(ns_i, 75.0 * float(ns_i))

def nice_input_label_html(key: str) -> str:
    ui = FEATURE_UI.get(key, {})
    label = ui.get("label", key)
    sym = ui.get("symbol_html", key)
    unit = ui.get("unit_html", "")
    if unit:
        return f'<div class="input-label">{label} ({sym}) [{unit}]</div>'
    return f'<div class="input-label">{label} ({sym})</div>'

def scenario_row(base_row, kind: str):
    r = dict(base_row)
    infill_keys = ["FM", "TM", "IP", "IP_GS"]
    if kind == "Input":
        return r
    if kind == "Bare frame":
        for k in infill_keys:
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

def make_integer_ticks_include_end(x_end: float, n: int = 7):
    if x_end <= 0:
        return [0]
    raw = np.linspace(0, x_end, n)
    ticks = np.unique(np.round(raw).astype(int))
    if ticks[-1] != int(round(x_end)):
        ticks = np.append(ticks, int(round(x_end)))
    ticks = np.unique(ticks)
    return ticks.tolist()

def ceil_to_nice_int(v: float, step: int = 50) -> int:
    if v <= 0:
        return step
    return int(np.ceil(v / step) * step)

# ============================================================
# Load artifacts
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

meta, Xsc, Ysc, models = load_artifacts(ART_SIG)

FEATURES = meta.get("FEATURES")
YVARS = meta.get("YVARS")
cfg = meta.get("cfg", {})

log_X = bool(cfg.get("log_transform_X", True))
log_Y = bool(cfg.get("log_transform_Y", True))

# ============================================================
# Preprocessing + prediction
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
# Curve construction (plotted in mm)
# ============================================================
def curve_key_points_mm(pred):
    F1 = float(pred["F1"])
    K1 = float(pred["K1"])
    F2 = float(pred["F2"])
    D2_mm = float(pred["D2"]) if D2_MODEL_IS_MM else float(pred["D2"]) * 1000.0
    K23 = float(pred["K23"])
    Fres = float(pred["Fres"])

    if abs(K1) < 1e-12 or abs(K23) < 1e-12:
        return None, "K1 or K23 too close to zero."

    D1_mm = F1 / K1
    D3_mm = D2_mm + (F2 - Fres) / K23

    x = [0.0, D1_mm, D2_mm, D3_mm]
    y = [0.0, F1, F2, Fres]
    pts = sorted(zip(x, y), key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts]

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
st.markdown(
    "<div class='app-title'>Pushover Curve predictor for multi-story masonry infilled reinforced concrete frames (XGB)</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([0.45, 0.55], gap="small")

if "inputs_dict" not in st.session_state:
    st.session_state["inputs_dict"] = {}

# Dropdown options
NS_OPTIONS = list(range(1, 13))
BW_OPTIONS = [4000, 6000]
BN_OPTIONS = list(range(1, 9))
FM_OPTIONS = [3, 6, 9, 15]
TM_OPTIONS = [115, 230]
IP_OPTIONS = list(range(1, 101))
IPGS_OPTIONS = list(range(1, 101))
FCK_OPTIONS = [30, 40]

with left:
    st.markdown("<div class='sec-h'>Inputs</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="small")

    inputs = {}
    for i, name in enumerate(FEATURES):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(nice_input_label_html(name), unsafe_allow_html=True)
            wkey = f"in_{name}"

            if name == "NS":
                inputs[name] = float(st.selectbox(" ", NS_OPTIONS, index=1, label_visibility="collapsed", key=wkey))
            elif name == "BW":
                inputs[name] = float(st.selectbox(" ", BW_OPTIONS, index=0, label_visibility="collapsed", key=wkey))
            elif name == "BN":
                inputs[name] = float(st.selectbox(" ", BN_OPTIONS, index=3, label_visibility="collapsed", key=wkey))
            elif name == "FM":
                inputs[name] = float(st.selectbox(" ", FM_OPTIONS, index=0, label_visibility="collapsed", key=wkey))
            elif name == "TM":
                inputs[name] = float(st.selectbox(" ", TM_OPTIONS, index=1, label_visibility="collapsed", key=wkey))
            elif name == "IP":
                inputs[name] = float(st.selectbox(" ", IP_OPTIONS, index=24, label_visibility="collapsed", key=wkey))
            elif name == "IP_GS":
                inputs[name] = float(st.selectbox(" ", IPGS_OPTIONS, index=24, label_visibility="collapsed", key=wkey))
            elif name == "FCK":
                inputs[name] = float(st.selectbox(" ", FCK_OPTIONS, index=0, label_visibility="collapsed", key=wkey))
            else:
                txt = st.text_input(" ", value="0.00", label_visibility="collapsed", key=wkey)
                inputs[name] = safe_float_from_text(txt, default=0.0)

    st.session_state["inputs_dict"] = inputs

with right:
    # (Removed "Outputs" heading as per your earlier request)

    # Title above figure
    st.markdown("<div class='sec-h' style='margin-top:2px;'>Infill configuration</div>", unsafe_allow_html=True)

    # Banner image
    banner_path = ART_DIR / "assets" / "images" / "frame_banner.png"
    if banner_path.exists():
        if BANNER_USE_CONTAINER_WIDTH:
            st.image(str(banner_path), use_container_width=True)
        else:
            st.image(str(banner_path), width=BANNER_WIDTH_PX)
    else:
        st.warning("Missing banner: assets/images/frame_banner.png")

    # Captions under each configuration (6 frames assumed)
    cap_cols = st.columns(6, gap="small")
    caps = [
        ("0", "0"),
        ("25", "25"),
        ("50", "50"),
        ("75", "75"),
        ("56", "0"),
        ("100", "100"),
    ]
    for i, (ip, ipgs) in enumerate(caps):
        with cap_cols[i]:
            st.markdown(
                f"""
                <div class="banner-cap">
                  <i>IP</i> = {ip}%<br/>
                  <i>IP</i><sub>GS</sub> = {ipgs}%
                </div>
                """,
                unsafe_allow_html=True
            )

    run = st.button("Predict pushover curve parameters", key="run_btn")

    if "run_state" not in st.session_state:
        st.session_state["run_state"] = False
    if run:
        st.session_state["run_state"] = True

    if st.session_state["run_state"]:
        base_row = dict(st.session_state["inputs_dict"])
        ns_base = base_row.get("NS", 2.0)
        axis_end = axis_end_mm_from_NS(ns_base)

        scenarios = [
            ("Input", "blue", "-"),
            ("Bare frame", "black", "--"),
            ("Fully Infilled", "green", ":"),
            ("Soft story", "red", "-."),
        ]

        curves = []
        scenario_rows = []

        for sc_name, color, ls in scenarios:
            row_dict = scenario_row(base_row, sc_name)
            pred = predict_one(row_dict)

            d2_mm = float(pred["D2"]) if D2_MODEL_IS_MM else float(pred["D2"]) * 1000.0

            scenario_rows.append({
                OUTPUT_HEADER_TXT["Scenario"]: sc_name,
                OUTPUT_HEADER_TXT["F1"]: float(pred["F1"]),
                OUTPUT_HEADER_TXT["K1"]: float(pred["K1"]),
                OUTPUT_HEADER_TXT["F2"]: float(pred["F2"]),
                OUTPUT_HEADER_TXT["D2_mm"]: d2_mm,
                OUTPUT_HEADER_TXT["K23"]: float(pred["K23"]),
                OUTPUT_HEADER_TXT["Fres"]: float(pred["Fres"]),
                OUTPUT_HEADER_TXT["EndDisp_mm"]: axis_end,
            })

            x_mm, y_kN = curve_key_points_mm(pred)
            x2, y2 = extend_to_axis_end(x_mm, y_kN, axis_end)
            curves.append((sc_name, color, ls, x2, y2))

        st.markdown("<div class='sec-h'>F–D plot (scenarios)</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(3.0, 2.5), dpi=170)

        xs = np.linspace(0.0, axis_end, 450)
        for name, color, ls, x_mm, y_kN in curves:
            ys = np.interp(xs, x_mm, y_kN)
            ax.plot(xs, ys, linewidth=1.0, color=color, linestyle=ls, label=name)

        ax.set_xlabel("Displacement (mm)", fontname=PLOT_FONT_NAME, fontsize=PLOT_AXIS_LABEL_PT)
        ax.set_ylabel("Base Shear (kN)", fontname=PLOT_FONT_NAME, fontsize=PLOT_AXIS_LABEL_PT)

        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, axis_end)

        all_y = np.concatenate([np.array(c[4], dtype=float) for c in curves]) if curves else np.array([1.0])
        y_max = float(np.max(all_y)) * 1.05
        y_max_int = ceil_to_nice_int(y_max, step=50)
        ax.set_ylim(0.0, y_max_int)

        xticks = make_integer_ticks_include_end(axis_end, n=7)
        yticks = make_integer_ticks_include_end(y_max_int, n=6)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_PT)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontname(PLOT_FONT_NAME)
            t.set_fontsize(PLOT_TICK_FONT_PT)

        leg = ax.legend(loc="upper right", frameon=False, fontsize=PLOT_LEGEND_FONT_PT)
        for txt in leg.get_texts():
            txt.set_fontname(PLOT_FONT_NAME)

        fig.tight_layout(pad=0.6)
        st.pyplot(fig, clear_figure=True, use_container_width=False)

        df_out = pd.DataFrame(scenario_rows)

        st.download_button(
            "Download scenario predictions (CSV)",
            data=df_out.to_csv(index=False),
            file_name="scenario_predictions.csv",
            mime="text/csv",
            key="dl_csv",
        )